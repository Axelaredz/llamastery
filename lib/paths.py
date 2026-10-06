"""Портативные пути. Переопределяются переменными окружения."""

import os
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent


def _x(name: str, default: Path) -> Path:
    """Переменная окружения, задающая путь.

    Инструмент назывался llama-preset-ops, и префикс переменных менялся вместе
    с ним. Старый префикс намеренно не поддерживается: держать его — значит
    обещать совместимость с чужими скриптами, которых у нас нет.
    """
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def home() -> Path:
    # Windows: HOME есть не всегда (в PowerShell его historically нет),
    # USERPROFILE — канонический путь профиля.
    return Path(os.environ.get("USERPROFILE")
                or os.environ.get("HOME", "~")).expanduser()


def config_dir() -> Path:
    """Каталог конфигурации инструмента (реестр сборок, кэш схемы).

    XDG на Unix, %APPDATA% на Windows — как принято у CLI-инструментов.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(home() / "AppData" / "Roaming")
        default = Path(base) / "llamastery"
    else:
        default = home() / ".config" / "llamastery"
    return _x("LLAMASTERY_CONFIG_DIR", default)


def state_dir() -> Path:
    """Состояние: кэш схемы флагов, калибровки, история прогонов."""
    if os.name == "nt":
        # LOCALAPPDATA — для данных, которые не должны ездить между машинами
        base = os.environ.get("LOCALAPPDATA") or str(home() / "AppData" / "Local")
        default = Path(base) / "llamastery"
    else:
        default = home() / ".local" / "state" / "llamastery"
    return _x("LLAMASTERY_STATE_DIR", default)


def cache_dir() -> Path:
    if os.name == "nt":
        # на Windows отдельного кэша нет — всё в LOCALAPPDATA рядом с состоянием
        default = state_dir() / "cache"
    else:
        default = home() / ".cache" / "llamastery"
    return _x("LLAMASTERY_CACHE_DIR", default)


def builds_file() -> Path:
    return _x("LLAMASTERY_BUILDS", config_dir() / "builds.json")


def calibration_file() -> Path:
    return state_dir() / "calibration.json"


def crashes_file() -> Path:
    """Журнал падений: пресет, который роняет CUDA, должен это помнить.

    Без журнала пресет-мина остаётся в наборе как ни в чём не бывало: внешне
    он валиден, а на глубоком контексте роняет сервер, и выясняется это
    через полчаса замеров.
    """
    return state_dir() / "crashes.json"


def schema_cache(binary: str | None = None) -> Path:
    """Кэш схемы — на каждый бинарь свой: у форков разные флаги."""
    if binary is None:
        return cache_dir() / "flag-schema.json"
    import hashlib
    key = hashlib.sha256(str(binary).encode()).hexdigest()[:12]
    stem = Path(binary).name
    return cache_dir() / f"schema-{stem}-{key}.json"


def default_ini() -> Path:
    """Пресет роутера по умолчанию.

    Сначала пробуем путь, который задаёт их менеджер `llama`
    (LLAMA_MODELS_INI), потом каноничный XDG-путь llama.cpp.
    """
    env = os.environ.get("LLAMA_MODELS_INI")
    if env:
        return Path(env).expanduser()
    return home() / ".config" / "llama" / "models.ini"


def ensure_dirs() -> None:
    for d in (config_dir(), state_dir(), cache_dir()):
        d.mkdir(parents=True, exist_ok=True)
