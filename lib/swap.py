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
import shlex
import shutil
import urllib.request
from pathlib import Path

from . import builds, paths, server

SWAP_PORT = 8080
SWAP_LISTEN = "0.0.0.0:8080"

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


def status(swap_url: str = "http://127.0.0.1:8080", timeout: float = 5.0) -> dict:
    """Опрос прокси: /health + /running. Не требует запущенного сервера."""
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
