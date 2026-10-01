"""Журнал падений: какие пресеты роняют сервер и при каких условиях.

Зачем. Пресет может загрузиться, пройти валидацию по схеме и упасть только
на глубоком контексте — так роняет CUDA связка «ubatch 2048 + ngram-mod»
на 114688. Внешне такой пресет ничем не отличается от рабочего, и узнать о
падении можно только потеряв полчаса замеров. Здесь падения запоминаются, а
`validate` и `presets annotate` показывают их как предупреждение и как строку
в блоке пресета.

Запись делается автоматически при неудачной загрузке и при падении инстанса
во время пробы. Формат — обычный JSON в состоянии, ключ по имени пресета.
"""

import json
import re
import time
from pathlib import Path

from . import paths

# причины, по которым сервер поднимается белым и это интересно помнить
_CRASH_PATTERNS = (
    ("cuda_oom", re.compile(r"out of memory|ggml_cuda_error|CUDA error: out of memory",
                            re.I), "CUDA: не хватило видеопамяти в середине запроса"),
    ("segfault", re.compile(r"segmentation fault|SIGSEGV", re.I), "падение SIGSEGV"),
    ("assert", re.compile(r"assertion failed|GGML_ASSERT", re.I), "сработал GGML_ASSERT"),
    ("kv_too_small", re.compile(r"kv cache|context shift|cannot allocate", re.I),
     "не хватило места под KV-кэш"),
    ("mmproj", re.compile(r"mmproj|multimodal|vision", re.I), "ошибка загрузки mmproj"),
)


def _load() -> dict:
    p = paths.crashes_file()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    paths.ensure_dirs()
    Path(paths.crashes_file()).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def classify(text: str) -> tuple[str, str] | None:
    """Что именно сломалось по тексту лога."""
    for kind, rx, why in _CRASH_PATTERNS:
        if rx.search(text or ""):
            return kind, why
    return None


def note(preset: str, build: str, text: str, *, depth: int | None = None,
         log: str | None = None) -> dict | None:
    """Запомнить падение пресета. None — если это не падение."""
    hit = classify(text)
    if not hit:
        return None
    kind, why = hit
    data = _load()
    entry = data.get(preset) or {"preset": preset, "count": 0, "first": None}
    entry.update({
        "build": build,
        "kind": kind,
        "why": why,
        "count": int(entry.get("count", 0)) + 1,
        "last": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": log,
    })
    if depth:
        entry["depth"] = int(depth)
    entry.setdefault("first", entry["last"])
    data[preset] = entry
    _save(data)
    return entry


def known(preset: str) -> dict | None:
    """Падал ли этот пресет раньше."""
    return _load().get(preset)


def all_entries() -> dict[str, dict]:
    return _load()


def forget(preset: str) -> bool:
    """Сбросить историю падений — после успешного замера."""
    data = _load()
    if preset in data:
        del data[preset]
        _save(data)
        return True
    return False


def tail_log(path: str | Path | None, lines: int = 200) -> str:
    """Хвост лога — из него вытаскивается причина падения."""
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])