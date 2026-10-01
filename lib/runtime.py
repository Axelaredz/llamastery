"""Загрузка сборки и пресета — тонкая прослойка над существующими менеджерами.

Логику роутера (порты, pid-файлы, окружение, рестарт) дублировать нельзя:
она уже отлажена в `llama` / `llama-faks` / `llama-ik`. Здесь только три
вещи, которых у них нет:

  1. единая точка входа, которая по имени сборки находит её менеджер;
  2. проверка пресета ДО загрузки (чтобы не поднимать заведомо плохое);
  3. прогноз VRAM перед загрузкой.

Все verbs менеджера поддерживаются как есть, новых не изобретается.
"""

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import builds, paths

VERBS = ("load", "unload", "start", "stop", "status", "doctor", "logs", "test")

# какие verbs поддерживает каждый менеджер; ik-шим умеет только start/stop/…
# и load у него означает «перезапустить сервер с другим пресетом»
SUPPORTED = {
    "llama": {"load", "unload", "start", "stop", "status", "doctor"},
    "llama-faks": {"load", "unload", "start", "stop", "status", "doctor"},
    "llama-ik": {"start", "stop", "status", "logs", "test"},
}


def pick_manager(manager: str | None, verb: str) -> tuple[str, str | None]:
    """Возвращает (команда, чем заменить verb), если verb не поддержан.

    Для ik-сборки `load` переводится в `llama load` — полноценный менеджер
    сам умеет переключать ik-пресет (он держит общий порт и pid-файл).
    """
    if manager is None:
        return "llama", None
    supported = SUPPORTED.get(Path(manager).name)
    if supported is None or verb in supported:
        return manager, None
    return "llama", manager


def manager_for(build_name: str | None) -> tuple[str | None, str | None]:
    """Возвращает (команда менеджера, имя сборки).

    Порядок: явное имя сборки -> текущая активная сборка -> первая
    зарегистрированная со включённым менеджером -> llama из PATH.
    """
    if build_name:
        b = builds.get(build_name)
        if b is None:
            return None, build_name
        return (b.manager or None), build_name
    active = active_build()
    if active:
        b = builds.get(active)
        if b and b.manager:
            return b.manager, active
    for name, b in builds.all_builds().items():
        if b.manager:
            return b.manager, name
    found = shutil.which("llama")
    return found, None


def router_url() -> str:
    return os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8099")


def probe(timeout: float = 2.0) -> dict:
    """Спрашивает роутер о состоянии. Не требует запущенного сервера."""
    base = router_url().rstrip("/")
    info = {"url": base, "up": False, "loaded": [], "error": None}
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            info["up"] = r.status < 400
    except urllib.error.HTTPError as exc:
        info["up"] = exc.code < 500
    except (urllib.error.URLError, OSError, ValueError) as exc:
        info["error"] = str(exc)
        return info
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        for m in data.get("data", []) or []:
            info["loaded"].append({
                "id": m.get("id"),
                "port": m.get("port") or (m.get("meta") or {}).get("port"),
                "state": (m.get("meta") or {}).get("state"),
            })
    except (urllib.error.URLError, OSError, ValueError) as exc:
        info["error"] = str(exc)
    return info


def active_build() -> str | None:
    """Какая сборка держит порт роутера.

    Определяется по тому, чей бинарь слушает порт: pid ищется через
    /proc, а имя процесса сверяется с реестром.
    """
    url = router_url()
    port = int(url.rsplit(":", 1)[-1].split("/")[0] or 0)
    if not port:
        return None
    try:
        out = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    blob = out.stdout
    if "pid=" not in blob:
        return None
    import re
    m = re.search(r"pid=(\d+)", blob)
    if not m:
        return None
    pid = int(m.group(1))
    try:
        exe = Path(f"/proc/{pid}/exe").resolve()
    except (OSError, PermissionError):
        return None
    for name, b in builds.all_builds().items():
        try:
            if b.server_bin.resolve() == exe:
                return name
        except OSError:
            continue
    return None


def pick_preset_name(query: str, ini) -> tuple[str | None, list[str]]:
    """Находит секцию по имени или подстроке. Возвращает (точное?, кандидаты)."""
    names = [n for n in ini.names() if n != "*"]
    if query in names:
        return query, [query]
    low = query.lower()
    exact = [n for n in names if n.lower() == low]
    if len(exact) == 1:
        return exact[0], exact
    partial = [n for n in names if low in n.lower()]
    if len(partial) == 1:
        return partial[0], partial
    return None, partial


def preflight(name: str, pairs: dict, build_name: str | None) -> tuple[bool, str]:
    """Проверяет пресет перед загрузкой. Возвращает (можно, отчёт)."""
    from . import schema, validate
    out = []
    ok = True

    binary, src = builds.resolve_binary(build_name)
    flags, meta = schema.load(binary)
    if meta.get("error"):
        out.append(f"! схема флагов недоступна: {meta['error']}")
    else:
        rep = validate.Report()
        validate.validate_section(name, pairs, flags, rep, check_paths=True)
        for f in rep.findings:
            out.append(f.line())
        if rep.count("error"):
            ok = False

    if not ok:
        return False, "\n".join(out)

    from . import budget, gguf
    mp = pairs.get("model") or pairs.get("m") or ""
    if mp and gguf.is_gguf(mp):
        try:
            model = gguf.probe(mp)
            mm = None
            mpp = pairs.get("mmproj")
            if mpp and gguf.is_gguf(mpp):
                mm = gguf.probe(mpp)
            est = budget.estimate(pairs, model, mmproj_meta=mm)
            total = budget.gpu_total_mib()
            out.append(f"  VRAM: {est.total_gb:.2f} GiB"
                       + (f" из {total / 1024:.2f} GiB, запас "
                          f"{(total / 1024) - est.total_gb:+.2f} GiB" if total else ""))
        except gguf.GGUFError as exc:
            out.append(f"! {exc}")
    return True, "\n".join(out)


def delegate(manager: str, verb: str, *args: str,
             dry_run: bool = False) -> int:
    """Передаёт команду менеджеру. Ничего не выдумывает."""
    if verb not in VERBS:
        raise ValueError(f"менеджер не знает команду {verb!r}")
    manager, replaced = pick_manager(manager, verb)
    if replaced:
        print(f"у {replaced} нет команды {verb!r} — делегирую в {manager}")
    cmd = [manager, verb, *[a for a in args if a]]
    if dry_run:
        print("было бы выполнено:", " ".join(cmd))
        return 0
    exe = shutil.which(manager) or manager
    return subprocess.call([exe, verb, *[a for a in args if a]])