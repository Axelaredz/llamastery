"""Помощники запуска пресета: поиск секции, активная сборка, предпроверка.

Раньше модуль был прослойкой над внешними менеджерами (`llama`, `llama-faks`),
но делегирование заменено собственным жизненным циклом сервера в lib/server.py,
и от той прослойки остались только живые функции: pick_preset_name,
active_build, preflight и router_url.
"""

import os
import subprocess
from pathlib import Path

from . import builds

def router_url() -> str:
    return os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8099")


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