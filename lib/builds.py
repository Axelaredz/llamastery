"""Реестр сборок/форков llama.cpp.

Смысл: на машине может быть несколько форков, у каждого свой каталог,
свои патчи и разные возможности (роутер есть/нет, спец-флаги). Держать
это в голове и в четырёх разных менеджерах — источник ошибок, поэтому
реестр один, в JSON, переживающий репосты.
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths

# Форматы строк версий у сборок разные:
#   ik:      "version: 104 (32cddbf)"
#   upstream/faks: "version: 0.5.0-dev (build 126, commit 19e28a27)"
# Коммит нужен всегда: по нему видно, что бинарь старее дерева.
VERSION_RE = re.compile(r"version:\s*(\S+)\s*\(([^)]+)\)")
_COMMIT_IN_B = re.compile(r"commit\s+([0-9a-f]{7,40})")


@dataclass
class Build:
    name: str
    path: str                       # корень исходников форка
    remote: str = ""                # URL репозитория
    router: bool = True             # поддерживает ли --models-preset
    note: str = ""
    manager: str = ""               # команда управления (необязательна)
    extra_flags: list[str] = field(default_factory=list)  # форковые флаги
    env: dict = field(default_factory=dict)   # переменные окружения сборки
    port: int | None = None        # свой порт, если сборка живёт не на 8099
    built: bool | None = None       # заполняется status

    # ── производные пути ──
    @property
    def server_bin(self) -> Path:
        return Path(self.path).expanduser() / "build" / "bin" / "llama-server"

    @property
    def bench_bin(self) -> Path:
        return Path(self.path).expanduser() / "build" / "bin" / "llama-bench"

    def status(self) -> dict:
        d = asdict(self)
        binp = self.server_bin
        d["server_bin"] = str(binp)
        d["bin_exists"] = binp.exists()
        d["bench_exists"] = self.bench_bin.exists()
        d["version"] = None
        d["git"] = {}
        if d["bin_exists"]:
            try:
                out = subprocess.run([str(binp), "--version"],
                                     capture_output=True, text=True, timeout=30)
                m = VERSION_RE.search(out.stdout + out.stderr)
                if m:
                    d["version"] = f"{m.group(1)} ({m.group(2)})"
            except (OSError, subprocess.SubprocessError):
                pass
        root = Path(self.path).expanduser()
        if (root / ".git").exists():
            for args, key in ((["rev-parse", "--abbrev-ref", "HEAD"], "branch"),
                              (["rev-parse", "--short", "HEAD"], "commit"),
                              (["status", "--porcelain"], "dirty")):
                try:
                    r = subprocess.run(["git", "-C", str(root), *args],
                                       capture_output=True, text=True, timeout=30)
                    val = r.stdout.strip()
                    if key == "dirty":
                        d["git"][key] = bool(val)
                    else:
                        d["git"][key] = val
                except (OSError, subprocess.SubprocessError):
                    pass
        return d


def freshness(name: str, fetch: bool = True) -> dict:
    """Актуальность сборки: отстаёт ли дерево от своего апстрима и бинарь — от дерева.

    Зачем. Сборка три дня не пересобиралась, а в форках новые коммиты могут
    менять не только скорость, но и набор флагов: пресет, который сегодня
    валиден, завтра может получить новый флаг — или наоборот, потерять старый,
    и об этом лучше узнать до замеров, а не после.

    Различает три разных состояния, которые легко спутать:
      - дерево отстаёт от апстрима (fetch не принёс ничего нового);
      - дерево ahead — есть локальные коммиты, которых нет в апстриме
        (форк или патч, и пересборка их не потеряет, но и не добавит ничего);
      - бинарь отстаёт от дерева (HEAD двигался, а сборка не пересобрана).
    """
    b = get(name)
    if b is None:
        return {"name": name, "error": "сборка не зарегистрирована"}
    root = Path(b.path).expanduser()
    info: dict = {"name": name, "path": str(root),
                  "remote": b.remote, "bin_exists": b.server_bin.exists()}

    def g(*args: str, timeout: int = 60) -> tuple[int, str]:
        try:
            r = subprocess.run(["git", "-C", str(root), *args],
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, str(exc)

    if not (root / ".git").exists():
        info["error"] = "каталог не под git — актуальность не проверить"
        return info

    _, head = g("rev-parse", "--short", "HEAD")
    _, branch = g("rev-parse", "--abbrev-ref", "HEAD")
    info.update(head=head, branch=branch)
    _, dirty = g("status", "--porcelain")
    info["dirty"] = bool(dirty)

    fetched, note = False, ""
    if fetch:
        rc, out = g("fetch", "--quiet", "--prune")
        fetched = rc == 0
        if not fetched:
            note = f"fetch не удался ({out.splitlines()[0] if out else 'нет сети'}) — " \
                   "сравнение с апстримом может быть неполным"
    info["fetched"] = fetched
    info["note"] = note

    # откуда считать «апстрим»: явный remote из реестра, иначе его upstream
    remote = b.remote or ""
    rc, up = g("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    tracking = up if rc == 0 else ""
    info["tracking"] = tracking
    if not tracking and remote:
        _, head_ref = g("rev-parse", "--abbrev-ref", "HEAD")
        _, exists = g("rev-parse", "--verify", "--quiet", remote)
        if exists:
            tracking = remote
            info["tracking"] = remote
    if not tracking:
        info["stale"] = None
        info["why"] = "нет ветки отслеживания — не с чем сравнивать"
        return info

    rc, behind = g("rev-list", "--count", f"HEAD..{tracking}")
    rc2, ahead = g("rev-list", "--count", f"{tracking}..HEAD")
    info["behind"] = int(behind) if rc == 0 and behind.isdigit() else None
    info["ahead"] = int(ahead) if rc2 == 0 and ahead.isdigit() else None

    info["new_commits"] = []
    if info.get("behind"):
        _, log = g("log", "--oneline", "--no-decorate",
                  f"HEAD..{tracking}", timeout=60)
        info["new_commits"] = [ln for ln in log.splitlines() if ln.strip()][:12]
        # могли ли прийти новые флаги: они живут в arg.cpp и описании сервера
        _, files = g("diff", "--name-only", f"HEAD...{tracking}")
        hot = [f for f in files.splitlines()
               if f.endswith(("arg.cpp", "server.cpp", "server-context.cpp",
                              "server-settings.cpp"))]
        info["flags_may_change"] = bool(hot)
        info["flag_files"] = hot[:6]
        if hot:
            _, dstat = g("diff", "--stat", f"HEAD...{tracking}", "--", *hot)
            info["flag_diffstat"] = dstat.strip().splitlines()[-1:] or []
    else:
        info["flags_may_change"] = False

    # бинарь против дерева
    out = ""
    if info["bin_exists"]:
        try:
            r = subprocess.run([str(b.server_bin), "--version"],
                               capture_output=True, text=True, timeout=30)
            out = r.stdout + r.stderr
        except (OSError, subprocess.SubprocessError):
            pass
    m = VERSION_RE.search(out)
    ver = m.group(2) if m else ""
    # коммит внутри скобок: либо «build N, commit abcdef», либо сам короткий hash
    bin_commit = None
    inner = _COMMIT_IN_B.search(ver)
    if inner:
        bin_commit = inner.group(1)
    elif re.fullmatch(r"[0-9a-f]{7,40}", ver.strip()):
        bin_commit = ver.strip()
    info["bin_commit"] = bin_commit
    info["bin_version"] = f"{m.group(1)} ({ver})" if m else None
    behind_by_commit = bool(
        bin_commit and info.get("head")
        and not bin_commit.startswith(info["head"][:7]))
    info["bin_behind_tree"] = behind_by_commit

    # Коммита в бинаре может не быть (сборка без вшитого hash). Тогда
    # сравниваем время: бинарь, собранный раньше последнего коммита, заведомо
    # не содержит изменений этого коммита.
    if not bin_commit:
        try:
            _, when = g("log", "-1", "--format=%ct")
            bin_mtime = b.server_bin.stat().st_mtime
            info["bin_mtime"] = int(bin_mtime)
            info["head_mtime"] = int(when) if when.isdigit() else None
            info["bin_older_than_head"] = bool(
                info["head_mtime"] and bin_mtime < info["head_mtime"])
        except (OSError, ValueError):
            info["bin_older_than_head"] = None
        info["bin_behind_tree"] = info["bin_older_than_head"]

    info["stale"] = bool((info.get("behind") or 0) > 0) or info["bin_behind_tree"]
    return info


def _load() -> dict:
    p = paths.builds_file()
    if not p.exists():
        return {"version": 1, "builds": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict) -> Path:
    p = paths.builds_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


def all_builds() -> dict[str, Build]:
    data = _load()
    out = {}
    for name, b in data.get("builds", {}).items():
        known = {f for f in Build.__dataclass_fields__}
        out[name] = Build(**{k: v for k, v in b.items() if k in known})
    return out


def get(name: str) -> Build | None:
    return all_builds().get(name)


def upsert(build: Build) -> Build:
    data = _load()
    data.setdefault("builds", {})[build.name] = asdict(build)
    _save(data)
    return build


def remove(name: str) -> bool:
    data = _load()
    if name in data.get("builds", {}):
        del data["builds"][name]
        _save(data)
        return True
    return False


def resolve_binary(name: str | None) -> tuple[Path | None, str]:
    """Возвращает (путь к llama-server, имя сборки).

    Приоритет: явное имя → LLAMA_SERVER_BIN → первая зарегистрированная
    сборка с готовым бинарём → llama-server из PATH.
    """
    if name:
        b = get(name)
        if b and b.server_bin.exists():
            return b.server_bin, name
        cand = Path(name).expanduser()
        if cand.exists():
            return cand, cand.name
        return None, name

    import os
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p, "env"
    for bname, b in all_builds().items():
        if b.router and b.server_bin.exists():
            return b.server_bin, bname
    from shutil import which
    p = which("llama-server")
    return (Path(p), "PATH") if p else (None, "")


# ── bootstrap: угадать сборки по типичным каталогам ──
CANDIDATES = [
    # (имя, путь, remote, router, manager)
    ("faks", "~/git/llama-faks", "https://github.com/Faks/llama.cpp", True, ""),
    ("upstream", "~/llama-upstream", "https://github.com/ggml-org/llama.cpp", True, ""),
    ("ik", "~/llama-ik", "https://github.com/ik_llama.cpp/ik_llama.cpp", False, ""),
]

# форковые оптимизации: знание о них живёт в реестре, а не в коде
KNOWN_ENV = {
    "faks": {"GGML_CUDA_REGISTER_HOST": "1", "GGML_SCHED_PREFETCH_EXPERTS": "1"},
}


def detect() -> list[Build]:
    """Находит форки в типичных местах. Ничего не записывает."""
    found = []
    for name, path, remote, router, manager in CANDIDATES:
        p = Path(path).expanduser()
        if not p.is_dir():
            continue
        # каталог исходников считаем форком, если внутри есть common/ или tools/
        if not ((p / "common").is_dir() or (p / "tools").is_dir()):
            continue
        real_remote = remote
        if (p / ".git").exists():
            try:
                r = subprocess.run(["git", "-C", str(p), "remote", "get-url",
                                    "origin"], capture_output=True, text=True,
                                   timeout=20)
                if r.returncode == 0 and r.stdout.strip():
                    real_remote = r.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        found.append(Build(name=name, path=str(p), remote=real_remote,
                           router=router, manager=manager,
                           env=dict(KNOWN_ENV.get(name, {})),
                           built=(p / "build" / "bin" / "llama-server").exists()))
    return found
