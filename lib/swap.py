"""Экспорт пресетов models.ini в конфиг llama-swap.

Зачем: llamastery — источник правды (validate + budget + замеры идут
напрямую к llama-server), llama-swap — только рантайм-прокси: один порт,
хот-свап, ttl. Поэтому здесь только генерация YAML и опрос статуса,
а НЕ загрузка через swap (иначе timings и VRAM-факт искажаются прокси).

Покрывает и роутерные сборки (faks/upstream), и single (ik): секция всегда
раскладывается в single-argv через server.preset_to_argv — матрешка
«swap -> router -> instance» запрещена осознанно.
"""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from . import builds, paths, server

# Порт по умолчанию. 8080 на этой машине занят контейнером searxng, поэтому
# дефолт — 8087, и он же первый кандидат при поиске свободного.
SWAP_PORT = 8087
SWAP_HOST = "127.0.0.1"
SWAP_LISTEN = f"{SWAP_HOST}:{SWAP_PORT}"
PORT_SCAN_SPAN = 200      # сколько портов перебрать, прежде чем сдаться

# ключи models.ini, которые НЕ являются флагами llama-server,
# а управляют роутером/комментариями — в cmd их не несём
SKIP_KEYS = {"models-dir", "models-max", "models-preset", "models-autoload",
             "alias", "host", "port", "tags"}


def find_binary() -> Path | None:
    """llama-swap в PATH или None."""
    p = shutil.which("llama-swap")
    return Path(p) if p else None


def default_output() -> Path:
    return paths.home() / ".config" / "llama-swap" / "config.yaml"


def _quote(s: str) -> str:
    """Минимальное YAML-цитирование для cmd-строк."""
    s = str(s)
    if any(c in s for c in "\"'\n:#{}[],&*?|-<>=!%@`"):
        return json.dumps(s)
    if not s or s.strip() != s:
        return json.dumps(s)
    return s


def section_to_cmd(section_name: str, pairs: dict, build: builds.Build) -> tuple[str, list[str], dict]:
    """Секция -> (cmd-строка для swap, предупреждения, env).

    Всегда single-режим: --port ${PORT} подставляет сам swap.
    """
    from . import schema
    flags, meta = schema.load(build.server_bin)
    if meta.get("error"):
        return "", [f"схема флагов недоступна: {meta['error']}"], {}
    clean = {k: v for k, v in pairs.items() if k.lower() not in SKIP_KEYS}
    argv, warns = server.preset_to_argv(clean, flags)
    parts = [str(build.server_bin)] + argv + ["--port", "${PORT}"]
    env = dict(build.env or {})
    # LD_LIBRARY_PATH для shared-сборок (faks/upstream — thin-бинари + .so рядом)
    bindir = str(build.server_bin.parent)
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = bindir
    # вся команда — одна YAML-строка: несколько quoted-сегментов подряд
    # YAML не принимает ("did not find expected key"), поэтому shlex внутри,
    # json.dumps снаружи
    return json.dumps(" ".join(shlex.quote(p) for p in parts)), warns, env


def export_yaml(ini_path: str | None = None, build_name: str | None = None,
                only: list[str] | None = None, ttl: int = 0) -> tuple[str, list[str]]:
    """Генерирует текст llama-swap.yaml. Ничего не пишет."""
    from .inifile import IniFile
    ini = IniFile.load(ini_path or str(paths.default_ini()))
    build = builds.get(build_name) if build_name else None
    if build is None and build_name:
        return "", [f"сборка {build_name!r} не в реестре"]
    if build is None:
        # дефолт: первая роутерная собранная, иначе любая собранная
        for b in builds.all_builds().values():
            if b.router and b.server_bin.exists():
                build = b
                break
        if build is None:
            for b in builds.all_builds().values():
                if b.server_bin.exists():
                    build = b
                    break
    if build is None:
        return "", ["нет собранной сборки в реестре"]
    names = [n for n in ini.names() if n != "*"]
    if only:
        want = {o.lower() for o in only}
        names = [n for n in names if n.lower() in want or any(w in n.lower() for w in want)]
    if not names:
        return "", ["ни одна секция не подошла под --only"]
    warns: list[str] = []
    lines = [
        f"# сгенерировано llamastery из {ini.path} build={build.name}",
        "# источник правды — models.ini: правится там, сюда — только export",
        "# validate/budget/measure/probe идут НАПРЯМУЮ к llama-server, не через swap",
        "",
        "healthCheckTimeout: 900",
        "",
        "models:",
    ]
    for name in names:
        sec = ini.section(name)
        pairs = sec.pairs()
        cmd, w, env = section_to_cmd(name, pairs, build)
        for x in w:
            warns.append(f"{name}: {x}")
        if not cmd:
            warns.append(f"{name}: пустой cmd — секция пропущена")
            continue
        lines.append(f"  {_quote(name)}:")
        lines.append(f"    cmd: {cmd}")
        if env:
            # swap ждёт env СПИСКОМ строк KEY=VALUE, мапу не ест
            # (cannot unmarshal !!map into []string)
            lines.append("    env:")
            for k, v in env.items():
                lines.append(f"      - {_quote(f'{k}={v}')}")
        if ttl:
            lines.append(f"    ttl: {ttl}")
        # alias = имя секции по умолчанию и так; явные aliases не дублируем
    lines.append("")
    return "\n".join(lines), warns


def running_models(swap_url: str = "http://127.0.0.1:8080",
                   timeout: float = 5.0) -> list[dict]:
    """Что swap держит запущенным: [{model, state, proxy_port}].

    Нужно `measure`: под swap бэкенд живёт на своём порту (`${PORT}`), а не
    на 8099, поэтому прямой сервер не видит ни одной модели и замер по
    прямому `/models` concludes «в VRAM ничего не загружено».
    """
    st = status(swap_url, timeout=timeout)
    if not st["up"]:
        return []
    data = st.get("running")
    if isinstance(data, dict):
        data = data.get("running")
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        proxy = str(item.get("proxy") or "")
        port = None
        if ":" in proxy:
            tail = proxy.rsplit(":", 1)[-1].split("/")[0]
            port = int(tail) if tail.isdigit() else None
        out.append({"model": item.get("model") or item.get("id") or "?",
                    "state": item.get("state") or "unknown",
                    "proxy": proxy, "proxy_port": port})
    return out


def status(swap_url: str = "http://127.0.0.1:8080", timeout: float = 5.0) -> dict:
    """Опрос прокси: /health + /running. Не требует запущенного сервера.

    Адрес приводится к схеме: в `swap status` без --url подставляется
    SWAP_LISTEN («host:port»), и без http:// urlopen падал с
    «unknown url type».
    """
    swap_url = swap_url if "://" in swap_url else f"http://{swap_url}"
    out: dict = {"url": swap_url, "up": False, "running": [], "error": None}
    try:
        with urllib.request.urlopen(f"{swap_url.rstrip('/')}/health",
                                    timeout=timeout) as r:
            out["up"] = r.status < 400
    except Exception as exc:  # noqa: BLE001 — статусная диагностика
        out["error"] = str(exc)
        return out
    try:
        with urllib.request.urlopen(f"{swap_url.rstrip('/')}/running",
                                    timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}
        out["running"] = data
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


# ── занятость порта ──
def parse_listen(text: str) -> tuple[str, int]:
    """Разбирает --listen/URL в (host, port).

    Принимает '0.0.0.0:8080', '127.0.0.1:8090', ':8080', '8080',
    'http://127.0.0.1:8080/'.
    """
    s = (text or "").strip()
    s = s.split("://", 1)[-1].rstrip("/")
    if ":" in s:
        host, _, port = s.rpartition(":")
    else:
        host, port = "", s
    return host or "0.0.0.0", int(port)


# ── демон: своими руками, как роутер ──
def _state() -> Path:
    d = paths.state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file() -> Path:
    return _state() / "swap.pid"


def state_file() -> Path:
    """Что запустили и на каком порту. Без этого «а где оно?» — частый вопрос."""
    return _state() / "swap.json"


def save_state(listen: str, config: str, pid: int) -> None:
    host, port = parse_listen(listen)
    state_file().write_text(json.dumps(
        {"listen": listen, "host": host, "port": port,
         "url": f"http://{host}:{port}", "config": str(config),
         "pid": pid, "started_at": int(time.time())},
        ensure_ascii=False, indent=1), encoding="utf-8")


def load_state() -> dict:
    try:
        return json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def clear_state() -> None:
    try:
        state_file().unlink()
    except OSError:
        pass


def find_free_port(host: str = SWAP_HOST, preferred: int = SWAP_PORT,
                   span: int = PORT_SCAN_SPAN) -> int:
    """Первый свободный порт начиная с preferred.

    Ручной подбор порта — это то, что человек всегда забывает сделать,
    поэтому подбор автоматический: занят 8087 — берём 8088 и говорим об этом.
    """
    probe = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    for port in range(preferred, preferred + span):
        if not is_port_busy(probe, port):
            return port
    return preferred


def resolve_listen(listen: str | None = None, auto: bool = True) -> tuple[str, str]:
    """Возвращает (listen, причина).

    Порядок: явный --listen → порт уже запущенного swap → свободный порт
    от дефолта. `auto=False` означает «не смещать молча».
    """
    if listen:
        host, port = parse_listen(listen)
        if auto and is_port_busy(host, port):
            if daemon_pid():
                return listen, f"уже запущен на {listen}"
            return (f"{host}:{find_free_port(host, port)}",
                    f"порт {port} занят ({describe_owner(port_owner(port))}) — "
                    f"выбран свободный")
        return listen, ""
    st = load_state()
    if st.get("listen"):
        host, port = parse_listen(st["listen"])
        if not is_port_busy(host, port):
            return st["listen"], "порт из прошлого запуска"
        if not auto:
            return st["listen"], f"порт {port} занят, а он записан у нас"
    host = st.get("host") or SWAP_HOST
    preferred = int(st.get("port") or SWAP_PORT)
    free = find_free_port(host, preferred)
    if free == preferred:
        return f"{host}:{preferred}", ""
    return (f"{host}:{free}",
            f"порт {preferred} занят ({describe_owner(port_owner(preferred))}) — "
            f"выбран свободный")


def log_file() -> Path:
    return _state() / "swap.log"


def daemon_pid() -> int | None:
    try:
        pid = int(pid_file().read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def up(config: str | None = None, listen: str | None = None,
       timeout: int = 20, auto_port: bool = True) -> dict:
    """Поднимает llama-swap демоном: свой pid-файл, лог, проверка /health.

    Отдельный демон, а не `&` в терминале: иначе никто не знает, кто его
    запустил, и остановить его можно только руками. Роутер поднимается
    ровно так же, поэтому поведение знакомое.
    """
    binary = find_binary()
    if binary is None:
        return {"ok": False, "message": "бинарь llama-swap не найден "
                                        "(llamastery swap install)"}
    cfg = Path(config).expanduser() if config else default_output()
    if not cfg.exists():
        return {"ok": False, "message": f"нет конфига: {cfg}\n"
                                        f"  собери его: llamastery swap export "
                                        f"-o {cfg}"}
    pid = daemon_pid()
    if pid:
        st = load_state()
        at = st.get("listen") or listen or SWAP_LISTEN
        return {"ok": True, "pid": pid, "url": f"http://{at}",
                "message": f"уже работает (pid {pid}) на {at}"}
    listen, why = resolve_listen(listen, auto=auto_port)
    host, port = parse_listen(listen)
    if is_port_busy(host, port):
        return {"ok": False,
                "message": f"порт {port} занят: {describe_owner(port_owner(port))}"
                           + (f"\n  освободи его или укажи другой: "
                              f"--listen {host}:{port + 1}"
                              if not auto_port else "")}
    log = open(log_file(), "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} llama-swap "
              f"--config {cfg} --listen {listen}\n")
    log.flush()
    argv = [str(binary), "--config", str(cfg), "--listen", listen]
    try:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return {"ok": False, "message": f"не запустился: {exc}"}
    pid_file().write_text(str(proc.pid), encoding="utf-8")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "pid": proc.pid,
                    "message": f"процесс умер (код {proc.returncode}). "
                               f"лог: {log_file()}"}
        if status(f"http://{host}:{port}", timeout=2.0)["up"]:
            save_state(listen, str(cfg), proc.pid)
            return {"ok": True, "pid": proc.pid, "url": f"http://{listen}",
                    "message": (f"swap поднят на {listen} (pid {proc.pid})"
                                + (f"\n  {why}" if why else ""))}
        time.sleep(0.5)
    return {"ok": False, "pid": proc.pid,
            "message": f"не ответил за {timeout} с. лог: {log_file()}"}


def next_steps(url: str) -> list[str]:
    """Что делать сразу после старта — иначе «поднял и забыл»."""
    return [
        f"проверить:      curl {url}/health",
        f"список моделей: curl {url}/v1/models",
        f"веб-интерфейс:  {url}/ui",
        f"OpenAI base_url: {url}/v1  (model = имя секции models.ini)",
        "остановить:     llamastery swap down",
        "статус:         llamastery swap status",
    ]


def down(timeout: int = 20) -> dict:
    """Останавливает демон swap. Модели в VRAM уходят вместе с ним."""
    pid = daemon_pid()
    if not pid:
        return {"ok": True, "message": "swap не запущен (или не нашим pid-файлом)"}
    try:
        server._terminate(pid, graceful=True)
    except OSError as exc:
        return {"ok": False, "message": f"не удалось завершить: {exc}"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and daemon_pid():
        time.sleep(0.3)
    if daemon_pid():
        try:
            server._terminate(pid, graceful=False)
        except OSError:
            pass
        time.sleep(0.5)
    for f in (pid_file(), state_file()):
        try:
            f.unlink()
        except OSError:
            pass
    return {"ok": True, "message": f"swap остановлен (pid {pid})"}


def is_port_busy(host: str, port: int, timeout: float = 1.0) -> bool:
    """Слушает ли кто-то порт. Только stdlib, без ss/lsof."""
    probe = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    try:
        with socket.create_connection((probe, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _tcp_listen_inodes(port: int) -> set[str]:
    """Inode сокетов в LISTEN на порту (IPv4+IPv6). Пусто — не Linux."""
    want = f"{port:04X}"
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            if len(f) < 10:
                continue
            try:
                local_port, state, inode = f[1].rsplit(":", 1)[1], f[3], f[9]
            except IndexError:
                continue
            if local_port.upper() == want and state == "0A":
                inodes.add(inode)
    return inodes


def port_owner(port: int) -> dict | None:
    """Кто держит порт: pid/exe/cmd, плюс docker-контейнер если он пробросил порт.

    Best-effort: нет прав или не Linux — вернёт None или часть полей.
    """
    owner: dict = {"port": port, "pid": None, "exe": None,
                   "cmd": None, "container": None}
    inodes = _tcp_listen_inodes(port)
    if inodes:
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
            except (OSError, PermissionError):
                continue
            hit = False
            for fd in fds:
                try:
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    hit = True
                    break
            if not hit:
                continue
            try:
                owner["exe"] = os.readlink(f"/proc/{pid}/exe")
            except (OSError, PermissionError):
                pass
            try:
                raw = Path(f"/proc/{pid}/cmdline").read_bytes()
                owner["cmd"] = raw.replace(b"\0", b" ").decode(
                    "utf-8", "replace").strip()[:200]
            except OSError:
                pass
            owner["pid"] = int(pid)
            break
    if shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Ports}}"],
                               capture_output=True, text=True, timeout=10)
            for ln in r.stdout.splitlines():
                if f":{port}->" in ln or f":{port}/" in ln or f"->{port}/" in ln:
                    owner["container"] = ln.strip()[:160]
                    break
        except (OSError, subprocess.SubprocessError):
            pass
    if owner["pid"] is None and owner["container"] is None:
        return None
    return owner


def describe_owner(owner: dict | None) -> str:
    """Одна строка для печати: кто занял порт."""
    if not owner:
        return "владелец не виден (чужой пользователь или не Linux)"
    bits = []
    if owner.get("container"):
        bits.append(f"docker: {owner['container']}")
    if owner.get("pid"):
        bits.append(f"pid {owner['pid']}")
    if owner.get("exe"):
        bits.append(owner["exe"])
    if owner.get("cmd"):
        bits.append(f"[{owner['cmd']}]")
    return "; ".join(bits) if bits else "владелец не виден"
