"""Жизненный цикл сервера llama.cpp: свой, без внешних менеджеров.

Раньше `llamastery` делегировал start/load/stop скриптам `llama`, `llama-faks`,
`llama-ik`. Это работало только на одной машине: у другого человека этих
скриптов нет, а ставить их отдельно ради скилла — лишний шаг. Поэтому здесь
реализовано то же самое напрямую:

  * роутер: llama-server --models-preset <ini> --host --port, свой pid-файл
    и лог в каталоге состояния;
  * API роутера: /health, /models, /models/load, /models/unload;
  * сборки без роутера (ik_llama): пресет транслируется в argv и сервер
    запускается как single-модельный;
  * окружение берётся из реестра сборок, а не из констант здесь.

Взаимодействие с API подтверждено по работающему менеджеру и по
tools/server/README.md: POST /models/load с телом {"model": "<id>"}.
"""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import builds, paths

STATUS_LABEL = {
    "loaded": "загружена (VRAM)",
    "loading": "грузится",
    "unloaded": "выгружена",
    "sleeping": "спит (выгрузится при нехватке VRAM)",
    "failed": "ОШИБКА",
    "downloading": "скачивается",
}


_active_build: str | None = None


def use_build(name: str | None) -> None:
    """Запомнить сборку, к которой относятся дальнейшие обращения.

    Нужна для сборок на своём порту: адрес сервера нужен не только при старте,
    но и во всех запросах после — /models, /completion, /tokenize. Раньше порт
    жил только в переменной окружения, поэтому реестр сборок молча его
    игнорировал, и вторая сборка без роутера падала с «couldn't bind to
    server socket» — без внятной причины.
    """
    global _active_build
    _active_build = name


def active_build() -> str | None:
    return _active_build


def port_number(build_name: str | None = None) -> int:
    env = os.environ.get("LLAMA_SERVER")
    if env:
        try:
            return int(env.rsplit(":", 1)[-1].split("/")[0] or 8099)
        except ValueError:
            return 8099
    name = build_name or _active_build
    if name:
        b = builds.get(name)
        if b is not None and b.port:
            return int(b.port)
    return 8099


def server_url(build_name: str | None = None) -> str:
    """Адрес сервера: явный LLAMA_SERVER → порт сборки → дефолтный 8099."""
    env = os.environ.get("LLAMA_SERVER")
    if env:
        return env.rstrip("/")
    return f"http://127.0.0.1:{port_number(build_name)}"


def state_dir() -> Path:
    return _state()


def _state() -> Path:
    d = paths.home() / ".local" / "state" / "llamastery"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file() -> Path:
    return _state() / "router.pid"


def log_file() -> Path:
    return _state() / "router.log"


def single_state() -> Path:
    return _state() / "single.json"


# ── HTTP ──
def http(method: str, url: str, body: dict | None = None,
         timeout: float = 10.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"error": exc.code, "body": exc.read().decode("utf-8", "replace")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"error": str(exc)}
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {"raw": raw}


def healthy(build_name: str | None = None, timeout: float = 3.0) -> bool:
    """Отвечает ли сервер.

    Первым идёт имя сборки, а не таймаут: во всех вызовах передавалось имя,
    а параметр стоял на первом месте — тихо уезжавшее в timeout имя ломало
    status с невнятным TypeError от socket.
    """
    base = server_url(build_name)
    j = http("GET", f"{base}/health", timeout=timeout)
    if isinstance(j, dict) and j.get("status") == "ok":
        return True
    if isinstance(j, dict) and "error" not in j:
        return True
    # старый single-сервер ik_llama отдаёт /health plain-текстом
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            return r.status == 200 and "ok" in r.read().decode("utf-8", "replace").lower()
    except (urllib.error.URLError, OSError, ValueError):
        return False


def models(reload: bool = False) -> list[dict]:
    url = f"{server_url()}/models" + ("?reload=1" if reload else "")
    j = http("GET", url)
    if not isinstance(j, dict) or "data" not in j:
        return []
    out = []
    for m in j.get("data", []):
        st = m.get("status")
        status, failed = "unknown", False
        if isinstance(st, dict):
            status = st.get("value", "unknown")
            failed = bool(st.get("failed"))
        out.append({"id": m.get("id", "?"), "status": status, "failed": failed})
    return out


def api_load(model_id: str) -> dict:
    return http("POST", f"{server_url()}/models/load", {"model": model_id})


def api_unload(model_id: str) -> dict:
    return http("POST", f"{server_url()}/models/unload", {"model": model_id})


def is_router() -> bool:
    """Роутер ли это. У одиночного сервера эндпоинта /models нет."""
    if http("GET", f"{server_url()}/models", timeout=3.0):
        return True
    owner = identify(pid_on_port())
    if owner:
        b = builds.get(owner)
        if b is not None:
            return bool(b.router)
    return False


def loaded_models() -> list[dict]:
    """Что загружено.

    Роутер сообщает состояние сам. У одиночного сервера модель всегда одна и
    всегда загружена, а `/models` либо отсутствует, либо отдаёт её без поля
    status — поэтому для таких сборок ответ синтетический. Режим определяется
    по реестру: у сборки router = False.
    """
    owner = identify(pid_on_port())
    b = builds.get(owner) if owner else None
    if b is not None and not b.router:
        sp = single_state()
        name = None
        if sp.exists():
            try:
                name = json.loads(sp.read_text(encoding="utf-8")).get("preset")
            except (OSError, json.JSONDecodeError):
                name = None
        if not name:
            lst = models()
            name = lst[0]["id"] if lst else "single"
        return [{"id": name, "status": "loaded", "failed": False,
                 "single": True}]
    lst = models()
    if lst:
        return lst
    sp = single_state()
    if sp.exists():
        try:
            name = json.loads(sp.read_text(encoding="utf-8")).get("preset")
        except (OSError, json.JSONDecodeError):
            return []
        if name:
            return [{"id": name, "status": "loaded", "failed": False,
                     "single": True}]
    return []


def wait_status(model_id: str, want: tuple[str, ...], timeout: int = 900,
                poll: float = 2.0) -> str:
    """Ждёт, пока модель придёт в одно из состояний. Возвращает статус."""
    deadline = time.monotonic() + timeout
    last = "?"
    while time.monotonic() < deadline:
        for m in models():
            if m["id"] == model_id:
                last = m["status"]
                if last in want:
                    return last
                if last == "failed":
                    return "failed"
        time.sleep(poll)
    return last


# ── владение портом ──
def pid_on_port(port: int | None = None) -> int | None:
    port = port or port_number()
    try:
        r = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    import re
    m = re.search(r"pid=(\d+)", r.stdout)
    return int(m.group(1)) if m else None


def proc_exe(pid: int) -> str | None:
    try:
        return str(Path(f"/proc/{pid}/exe").resolve())
    except (OSError, PermissionError):
        return None


def identify(pid: int | None) -> str | None:
    """Имя зарегистрированной сборки по pid."""
    if not pid:
        return None
    exe = proc_exe(pid)
    if not exe:
        return None
    for name, b in builds.all_builds().items():
        try:
            if b.server_bin.resolve() == Path(exe):
                return name
        except OSError:
            continue
    return None


def proc_argv(pid: int | None) -> list[str]:
    """Аргументы живого процесса — по ним видно режим запуска."""
    if not pid:
        return []
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\0") if a]


def read_pid() -> int | None:
    try:
        return int(pid_file().read_text().strip())
    except (OSError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)


# ── окружение сборки ──
def build_env(build: builds.Build) -> dict[str, str]:
    """LD_LIBRARY_PATH + переменные, объявленные в реестре сборки.

    Реестр — единственное место, где знают про форковые оптимизации
    (например GGML_CUDA_REGISTER_HOST у Faks), чтобы здесь не было констант.
    """
    env = dict(os.environ)
    bindir = str(build.server_bin.parent)
    env.setdefault("LD_LIBRARY_PATH", bindir)
    for k, v in (build.env or {}).items():
        env.setdefault(k, str(v))
    return env


def router_argv(build: builds.Build, ini: str, host: str, port: int) -> list[str]:
    return [str(build.server_bin), "--models-preset", ini,
            "--host", host, "--port", str(port)]


# ── пресет -> argv (для сборок без роутера) ──
# ВНИМАНИЕ: правило «роутер управляет сам» тут НЕ действует. Оно про
# per-model пресеты роутера, где секция описывает запись каталога. Здесь секция
# превращается в командную строку одиночного сервера, поэтому `model` и
# `mmproj` — обязательные аргументы, а выкидываются только те, что задаются
# нами же при сборке команды.
SINGLE_SKIP = {"models-dir", "models-max", "models-preset", "models-autoload",
               "alias", "host", "port"}


TRUE_WORDS = {"1", "true", "yes", "on", "enabled"}
FALSE_WORDS = {"0", "false", "no", "off", "disabled"}


def _normalize_enum(text: str, enum: list[str], warns: list[str],
                    key: str) -> str:
    """Приводит значение к тому, что эта сборка принимает.

    Один и тот же пресет должен работать на разных форках, а формы записи
    булевых у них разные: faks понимает `fa = true`, ik_llama просит
    `fa = on` или `fa = 1`. Поэтому true/on/yes приводится к тому варианту,
    который есть в enum этой сборки.
    """
    low = [e.lower() for e in enum]
    v = text.strip().lower()
    if v in low:
        return text
    if v in TRUE_WORDS:
        for cand in ("on", "true", "1", "yes", "enabled"):
            if cand in low:
                return cand
    if v in FALSE_WORDS:
        for cand in ("off", "false", "0", "no", "disabled"):
            if cand in low:
                return cand
    if v == "auto" and "auto" in low:
        return "auto"
    if v in ("all", "none") and v in low:
        return v
    warns.append(f"{key}: {text!r} не из списка {enum} — передам как есть")
    return text


def preset_to_argv(pairs: dict, flags: dict, defaults: dict | None = None,
                   extra: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Переводит секцию пресета в argv одиночного llama-server.

    Требуется схема флагов: длинная форма берётся из неё, короткая запись
    в INI переводится в каноническую. Возвращает (argv, предупреждения).
    """
    from . import schema
    warns: list[str] = []
    args: list[str] = []
    d = defaults or {}
    seen: set[str] = set()

    for key, val in pairs.items():
        if key.lower() in SINGLE_SKIP:
            continue
        f = schema.resolve(flags, key)
        if f is None:
            warns.append(f"флаг {key!r} неизвестен этой сборке — пропущен")
            continue
        canon = f.canonical
        if canon.lstrip("-") in SINGLE_SKIP:
            continue
        if canon in seen:
            continue
        seen.add(canon)
        if val is None or str(val).strip() == "":
            if f.kind == "flag":
                args.append(canon)
            continue
        text = str(val).strip()
        # булевы флаги (в т.ч. --no-mmproj-offload) значения не принимают:
        # в INI они либо пустые, либо «true» — в обоих случаях сам факт
        # присутствия ключа означает «включить»
        if f.kind == "flag":
            args.append(canon)
            continue
        if f.kind == "int" or f.kind == "float":
            try:
                float(text)
            except ValueError:
                warns.append(f"{key}: {text!r} не число — пропущен")
                continue
        if f.kind == "enum" and f.enum:
            text = _normalize_enum(text, f.enum, warns, key)
        args += [canon, text]

    for k, v in (d or {}).items():
        f = schema.resolve(flags, k)
        canon = f.canonical if f else k
        if canon in seen or canon.lstrip("-") in SINGLE_SKIP:
            continue
        seen.add(canon)
        if str(v).strip() == "" and f and f.kind == "flag":
            args.append(canon)
        else:
            args += [canon, str(v)]

    if extra:
        args += list(extra)
    return args, warns


# ── старт / стоп ──
@dataclass
class StartResult:
    ok: bool
    pid: int | None = None
    message: str = ""
    argv: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def start(build_name: str | None = None, ini: str | None = None,
          host: str = "127.0.0.1", timeout: int = 90,
          preset: str | None = None) -> StartResult:
    """Поднимает сервер. Для сборок без роутера нужен аргумент preset."""
    from . import schema
    ini = ini or str(paths.default_ini())
    build = builds.get(build_name) if build_name else None
    if build_name and build is None:
        return StartResult(False, message=f"сборка {build_name!r} не в реестре")
    if build is None:
        build = next((b for b in builds.all_builds().values()
                      if b.router and b.server_bin.exists()), None)
        if build is None:
            return StartResult(False, message="нет собранной сборки с роутером")
    if not build.server_bin.exists():
        return StartResult(False, message=f"бинарь не собран: {build.server_bin}")

    # порт берём у сборки: вторая сборка без роутера не встанет на 8099,
    # если он уже занят другой. Сборку запоминаем: дальше все обращения
    # (модели, проба, замер) должны идти на её адрес, а не на дефолтный.
    use_build(build.name)
    port = port_number(build.name)
    owner_pid = pid_on_port(port)
    if owner_pid:
        owner = identify(owner_pid) or "чужой процесс"
        if owner != build.name:
            return StartResult(False, pid=owner_pid,
                               message=(f"порт {port} уже занят: "
                                        f"{owner} (pid {owner_pid}). "
                                        f"Останови: llamastery runtime stop"))
        if healthy(build.name):
            return StartResult(True, pid=owner_pid,
                               message=f"уже работает: {build.name} "
                                       f"(pid {owner_pid})")
        _terminate(owner_pid)

    env = build_env(build)
    if build.router:
        argv = router_argv(build, ini, host, port)
    else:
        from .inifile import IniFile
        try:
            ini_obj = IniFile.load(ini)
        except OSError as exc:
            return StartResult(False, message=f"не читается пресет: {exc}")
        sec = ini_obj.section(preset) if preset else None
        if sec is None:
            return StartResult(False,
                               message="для сборки без роутера нужен пресет: "
                                       f"llamastery runtime start --preset <секция>")
        flags, smeta = schema.load(build.server_bin)
        if smeta.get("error"):
            return StartResult(False, message=f"схема флагов: {smeta['error']}")
        extra, warns = preset_to_argv(sec.pairs(), flags)
        argv = [str(build.server_bin), *extra, "--host", host,
                "--port", str(port)]
        Path(single_state()).write_text(
            json.dumps({"preset": sec.name, "build": build.name,
                        "port": port}, ensure_ascii=False), encoding="utf-8")
        res = StartResult(False, argv=argv, warnings=warns)
        res.message = "запуск single-сервера"
        return _spawn(build, env, argv, timeout, res)

    # роутер не single-сервер: старый single.json относится к прошлому
    # владельцу порта и в status показывался как «сейчас загружено», хотя
    # порт уже занят роутером с другими моделями
    try:
        Path(single_state()).unlink()
    except OSError:
        pass
    res = _spawn(build, env, argv, timeout, StartResult(True, argv=argv))
    return res


def _spawn(build: builds.Build, env: dict, argv: list[str], timeout: int,
           res: StartResult) -> StartResult:
    try:
        log = open(log_file(), "a")
    except OSError as exc:
        res.ok = False
        res.message = f"не открыть лог: {exc}"
        return res
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"{build.name}: {' '.join(argv)}\n")
    log.flush()
    try:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
    except OSError as exc:
        res.ok = False
        res.message = f"не запустился: {exc}"
        return res
    pid_file().write_text(str(proc.pid))
    res.pid = proc.pid

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            res.ok = False
            res.message = (f"процесс умер с кодом {proc.returncode}. "
                           f"лог: {log_file()}")
            return res
        if healthy():
            res.ok = True
            res.message = (f"роутер готов: {server_url(build.name)} "
                       f"({build.name}, pid {proc.pid})")
            return res
        time.sleep(0.5)
    res.ok = False
    res.message = (f"сервер не ответил за {timeout} с. лог: {log_file()}")
    return res


def stop(timeout: int = 30) -> tuple[bool, str]:
    """Останавливает сервер: сначала pid-файл, затем владельца порта.

    Порты перебираются все, а не один: сборки живут на разных адресах, и
    остановка по одному порту оставляла вторую сборку висеть в VRAM.
    """
    pids = []
    pf = read_pid()
    if pf and alive(pf):
        pids.append(pf)
    ports = {port_number(_active_build), 8099}
    ports.update(int(b.port) for b in builds.all_builds().values() if b.port)
    for prt in sorted(ports):
        owner = pid_on_port(prt)
        if owner and owner not in pids:
            pids.append(owner)
    if not pids:
        return True, "сервер не запущен"
    msgs = []
    for pid in pids:
        who = identify(pid) or "процесс"
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            msgs.append(f"{who} (pid {pid}): не удалось SIGTERM — {exc}")
            continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and alive(pid):
            time.sleep(0.3)
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except OSError:
                pass
        msgs.append(f"{who} (pid {pid}): остановлен"
                    + ("" if not alive(pid) else " — не уходит даже по SIGKILL"))
    try:
        pid_file().unlink()
    except OSError:
        pass
    return all("остановлен" in m or "не запущен" in m for m in msgs), "; ".join(msgs)


def status() -> dict:
    # сборку знаем по pid-файлу: он переживает перезапуск процесса, тогда как
    # pid на порту ищется заново и на нестандартном порту не находится
    pf = read_pid()
    owner = identify(pf)
    if owner:
        use_build(owner)
    port = port_number(owner or _active_build)
    pid = pf if (pf and alive(pf)) else pid_on_port(port)
    up = healthy(owner or _active_build)
    info = {
        "url": server_url(owner or _active_build),
        "port": port,
        "up": up,
        "pid": pid,
        "build": owner,
        "models": models() if up else [],
        "log": str(log_file()),
        "ini": str(paths.default_ini()),
    }
    sp = single_state()
    if sp.exists():
        try:
            single = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            single = None
        if single:
            info["single"] = single
            # Порт может держать роутер, а single.json — след прошлого
            # владельца. Показывать такую запись как «сейчас загружено» нельзя:
            # такой модели в списке нет. Роутер узнаём по --models-preset
            # в argv живого процесса.
            pid = read_pid() or pid_on_port()
            is_router = up and pid and "--models-preset" in proc_argv(pid)
            info["single_stale"] = is_router
    return info