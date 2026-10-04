"""Оси тюнинга: реестр, автообнаружение новых флагов, приоритет по замерам.

Зачем этот файл. Тюнер (`tools/tune_models.py`) перебирает параметры, и
порядок этого перебора определяет, сколько бюджета ты потратишь впустую.
Раньше порядок был зашит в тюнер константой, а список флагов — в словаре
каждой сборки. Обе вещи устаревают: форк обновили, приехали новые флаги,
а реальный прирост от `n-cpu-moe` на твоём железе может отличаться от
приортета «на глаз».

Здесь три вещи:

  1. `AXES` — реестр осей: флаг, зачем, ожидаемый прирост (1–5), как получить
     значения. Единственный источник правды и для тюнера, и для тюнера-док.
  2. `discover()` — флаги этой сборки, которых нет в реестре, но которые
     похожи на ручки производительности. Не молчаливый включённый поиск, а
     предложение: сначала `validate` и `budget`, потом ось.
  3. `learn()` — приоритет, посчитанный по твоим же прогонам: если измеренный
     эффект оси заметно больше (или меньше) приортета, порядок меняется.

Приоритет по умолчанию — константа `PRIOR`, но он не «священный»: функция
`order()` накладывает на него измеренное и отдаёт пересортированный список.
"""

import json
import re
import time
from pathlib import Path

from . import paths

# ── реестр осей ──
# gain 1..5 — ожидаемый прирост, отсортированы по убыванию.
# flag — как ось выглядит в этой сборке; alias — другие написания.
# probe — подстрока для `schema --grep`, чтобы проверить наличие.
# values — "enum" (перечислимые значения) или "int" (нужен генератор значений).
AXES = [
    {
        "key": "moe",
        "flag": "--n-cpu-moe",
        "alias": ["-ncmoe"],
        "probe": "cpu-moe",
        "values": "int",
        "gain": 5,
        "why": "сколько слоёв MoE-экспертов живёт в RAM: ~260 MiB VRAM на слой, "
               "главный рычаг на тесной карте",
    },
    {
        "key": "kv_cache_type",
        "flag": "--cache-type-k",
        "alias": ["-ctk", "--cache-type-v"],
        "probe": "cache-type",
        "values": "enum:q8_0,q5_0,q4_0,f16",
        "gain": 4,
        "why": "тип KV-кэша: экономит VRAM дешевле по скорости, часто "
               "позволяет уменьшить n-cpu-moe",
    },
    {
        "key": "ubatch",
        "flag": "--ubatch-size",
        "alias": ["-ub"],
        "probe": "ubatch",
        "values": "int",
        "gain": 4,
        "why": "размер физического батча: скорость префилла и объём compute "
               "buffer (на 12 ГБ 2048 — OOM)",
    },
    {
        "key": "fa",
        "flag": "--flash-attn",
        "alias": ["-fa"],
        "probe": "flash-attn",
        "values": "enum:on,off,auto",
        "gain": 3,
        "why": "flash attention: заметно влияет на префилл",
    },
    {
        "key": "threads",
        "flag": "--threads",
        "alias": ["-t"],
        "probe": "threads",
        "values": "int",
        "gain": 3,
        "why": "потоки генерации: при экспертах на CPU это узкое место",
    },
    {
        "key": "spec",
        "flag": "--spec-type",
        "probe": "spec-type",
        "values": "enum:none,ngram-mod,ngram-simple",
        # приоритет 5 — не из общих соображений, а из твоего же A/B:
        # docs/ru/tuning.md, Qwen3.8 MiniPlus, 114688: на повторяющемся тексте
        # 84-92 t/s против 26 (x3.2), на уникальном 30 против 28.5 (x1.05).
        # То есть это самый большой рычаг из всех, но только на том тексте,
        # для которого он и предназначен.
        "gain": 5,
        "measured": {"repetitive": 3.2, "unique": 1.05,
                     "src": "docs/ru/tuning.md, Qwen3.8 MiniPlus, 114688"},
        # Не ось линейного поиска: эффект знакопеременный — на повторяющемся
        # тексте ×3-4, на уникальном около нуля. В общую сетку его соваливать
        # бессмысленно (в твоих прогонах 0 замеров именно потому, что тюнер
        # исключает ускоритель), поэтому он идёт отдельным A/B.
        "mode": "ab",
        "why": "спекулятивный декодер: ×3–4 на повторяющемся тексте, "
               "на уникальном около нуля — мерить A/B на обоих текстах",
    },
    {
        "key": "b",
        "flag": "--batch-size",
        "alias": ["-b"],
        "probe": "batch-size",
        "values": "int",
        "gain": 1,
        "why": "логический батч: осмыслен только больше ubatch; в новых "
               "сборках n_batch клампится в n_ubatch",
    },
]

BY_KEY = {a["key"]: a for a in AXES}


def known_flags() -> set[str]:
    """Все флаги, которые реестр уже знает (включая алиасы)."""
    out: set[str] = set()
    for a in AXES:
        out.add(a["flag"])
        out.update(a.get("alias", []))
    return out


# ── автообнаружение новых флагов ──
# Подстроки, по которым флаг считается «ручкой производительности».
# Намеренно грубо: лучше показать лишнее, чем промолчать про новый флаг.
PERF_HINTS = (
    "-moe", "-batch", "ubatch", "-threads", "cache-type", "flash",
    "-gqa", "spec", "ngram", "draft", "offload", "-poll", "prio",
    "numa", "repack", "no-host", "op-offload", "fit", "swa", "lookahead",
    "kv", "-split", "main-gpu", "tensor-split", "-ctx", "image-",
    "mtmd", "parallel", "lora", "no-mmap", "mlock", "load-mode",
)


# Флаги, которые похожи на ручки, но осями не являются: либо пресеты,
# либо устаревшие, либо документированныеknob'ы вне линий тюнинга. Без этого
# списка «новые флаги» превращались в шум из 84 позиций.
KNOWN_NOT_AXES = {
    # готовые пресеты и удобства, не параметры
    "--fim-qwen-1.5b-default", "--fim-qwen-3b-default", "--fim-qwen-7b-default",
    "--fim-qwen-7b-spec", "--fim-qwen-14b-spec", "--fim-qwen-30b-default",
    "--spec-default", "--gpt-oss-20b-default", "--gpt-oss-120b-default",
    "--vision-gemma-4b-default", "--vision-gemma-12b-default",
    "--embd-gemma-default", "--fim-qwen-30b-spec",
    # объявленные удалёнными в новых версиях
    "--draft", "--draft-min", "--draft-max", "--spec-ngram-size-n",
    "--spec-ngram-size-m", "--spec-ngram-min-hits",
    # документированные knob'ы, но не оси линейного поиска
    "--cpu-moe", "--kv-unified", "--kv-offload", "--kv-unified-per-slot",
    "--ctx-checkpoints", "--swa-checkpoints", "--ctx-checkpoint-min-step",
    "--checkpoint-min-step", "--cache-ram", "--cache-reuse", "--cache-prompt",
    "--cache-idle-slots", "--fit", "--fit-ctx", "--fit-target",
    "--load-mode", "--lazy-mode", "--repack", "--no-host", "--op-offload",
    "--cont-batching", "--context-shift", "--no-context-shift",
    "--image-min-tokens", "--image-max-tokens", "--mtmd-batch-max-tokens",
    "--mmproj-offload", "--no-mmproj-offload", "--override-tensor",
    "-lora", "--lora-scaled", "--parallel", "-np", "--lookahead",
    "--no-cpu-moe", "--swa-full", "--no-mmap", "--mlock", "--poll",
    "--poll-batch", "--prio", "--prio-batch", "--numa", "--main-gpu",
    "--split-mode", "--tensor-split", "--gpu-layers", "--n-gpu-layers",
    "--ctx-size", "-c", "-b", "--batch-size", "-ub", "--ubatch-size",
    "--threads", "-t", "--threads-batch", "-tb", "--cache-type-k", "-ctk",
    "--cache-type-v", "-ctv", "--flash-attn", "-fa", "--n-cpu-moe", "-ncmoe",
    # ручки драфт-модели: они к `--spec-draft-model`, а не к линии поиска
    "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min",
    "--spec-draft-p-split", "--spec-draft-type-k", "--spec-draft-type-v",
    "--spec-draft-cpu-mask", "--spec-draft-cpu-mask-batch",
    "--spec-draft-cpu-range", "--spec-draft-cpu-range-batch",
    "--spec-draft-cpu-strict", "--spec-draft-cpu-strict-batch",
    "--spec-draft-device", "--spec-draft-ngl", "--spec-draft-n-cpu-moe",
    "--spec-draft-override-tensor", "--spec-draft-threads",
    "--spec-draft-threads-batch", "--spec-draft-prio", "--spec-draft-prio-batch",
    "--spec-draft-poll", "--spec-draft-poll-batch", "--spec-draft-hf",
    "--spec-draft-model", "--spec-draft-cache-type-k", "--spec-draft-cache-type-v",
    # привязка к CPU и LoRA — не про производительность на GPU
    "--cpu-mask", "--cpu-mask-batch", "--cpu-range", "--cpu-range-batch",
    "--cpu-strict", "--cpu-strict-batch", "--lora-init-without-apply",
    "--override-kv", "--override-tensor-draft",
}


def discover(binary, flags: dict | None = None) -> list[str]:
    """Флаги сборки, похожие на ручки скорости, но не описанные в реестре.

    flags — результат `schema.load()`; если None, читается сама схема.
    Возвращает отсортированный список ИМЁН (без ведущих дефисов).
    """
    if flags is None:
        from . import schema
        flags, meta = schema.load(binary)
        if meta.get("error"):
            return []
    known = {f.lstrip("-") for f in known_flags()}
    skip = {f.lstrip("-") for f in KNOWN_NOT_AXES}
    out = []
    for name in flags:
        n = name.lstrip("-")
        if n in known or n in skip:
            continue
        low = name.lower()
        if any(h in low for h in PERF_HINTS):
            out.append(name)
    return sorted(out)


def flags_snapshot(binary) -> dict:
    """Снимок набора флагов — чтобы узнать, что изменилось при обновлении."""
    from . import schema
    flags, meta = schema.load(binary)
    return {
        "count": len(flags),
        "names": sorted(flags),
        "mtime": meta.get("mtime"),
        "at": int(time.time()),
    }


def snapshot_path(build_name: str) -> Path:
    return paths.state_dir() / f"flags-{build_name}.json"


def save_snapshot(build_name: str, snap: dict) -> Path:
    p = snapshot_path(build_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_snapshot(build_name: str) -> dict:
    try:
        return json.loads(snapshot_path(build_name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def diff_snapshot(build_name: str) -> dict:
    """Что появилось и исчезло с прошлого снимка флагов этой сборки.

    Пустой результат — не «обновлений не было», а «снимка нет». Это разные
    вещи, и путать их опасно: молчаливое «всё ок» при первом запуске.
    """
    from . import builds
    b = builds.get(build_name)
    if b is None:
        return {"error": f"сборка {build_name!r} не в реестре"}
    prev = load_snapshot(build_name)
    now = flags_snapshot(b.server_bin)
    if not prev:
        save_snapshot(build_name, now)
        return {"first": True, "added": [], "removed": [],
                "count": now["count"], "saved": True}
    old = set(prev.get("names", []))
    new = set(now["names"])
    return {
        "first": False,
        "added": sorted(new - old),
        "removed": sorted(old - new),
        "count": now["count"],
        "unknown_perf": discover(b.server_bin, {n: None for n in new}),
    }


# ── приоритет по замерам ──
def _load_results() -> list[dict]:
    """Все успешные прогоны из tune-results/**/results.json."""
    root = paths.home() / ".config" / "llama" / "tune-results"
    out = []
    if not root.is_dir():
        return out
    for f in sorted(root.glob("**/results.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            out.extend(r for r in data if isinstance(r, dict))
    return out


# ключ оси -> как значение лежит в config прогона
CONFIG_KEY = {
    "moe": ("n-cpu-moe",),
    "kv_cache_type": ("cache-type-k", "cache-type-v"),
    "ubatch": ("ubatch-size", "ub"),
    "fa": ("fa", "flash-attn"),
    "threads": ("t", "threads"),
    "b": ("b", "batch-size"),
    "spec": ("spec-type",),
}


def _axis_values(cfg: dict) -> dict:
    """Значения осей в прогоне — по одному на ось."""
    out = {}
    for key, names in CONFIG_KEY.items():
        val = next((str(cfg[n]) for n in names if n in cfg), None)
        if val is not None:
            out[key] = val
    return out


def spec_effect(records: list[dict] | None = None) -> dict:
    """Прирост ускорителя раздельно по типам текста.

    Один вывод «ngram бесполезен» или «ngram даёт ×4» врёт: правда в том,
    что на повторах он даёт ×3-4, а на уникальном тексте — около нуля.
    Поэтому храним две оценки и решение принимает вызывающий, зная про нагрузку.
    """
    if records is None:
        records = _load_results()
    out: dict[str, dict] = {}
    for r in records:
        ab = r.get("spec_ab")
        if not isinstance(ab, dict):
            continue
        for workload, key in (("unique", "unique"), ("repetitive", "repetitive")):
            plain = (ab.get(f"tg_{key}_plain") or 0)
            spec = (ab.get(f"tg_{key}_spec") or 0)
            if plain > 0 and spec > 0:
                cur = out.setdefault(workload, {"pairs": 0, "gain": 0.0})
                cur["pairs"] += 1
                cur["gain"] = max(cur["gain"], spec / plain)
    for v in out.values():
        v["gain"] = round(v["gain"], 2)
    return out


def learn(min_pairs: int = 2) -> dict:
    """Наблюдаемый эффект каждой оси по твоим прогонам.

    Считаются только ИЗОЛИРОВАННЫЕ пары: два прогона, которые отличаются
    ровно одной осью. Иначе приписываем оси эффект, который вызван другой:
    в полном переборе разброс по t/s порождается и n-cpu-moe, и ubatch
    одновременно, и припишем его обоим (так и вышло в первой версии:
    у всех осей одинаковые 6.22).

    Возвращает {key: {pairs, values, tps_gain, vram_gain}}, где gain —
    во сколько раз лучший результат превосходит худший по этой оси.
    """
    runs = []
    for r in _load_results():
        if not r.get("ok"):
            continue
        cfg = dict(r.get("config") or {})
        cfg.update(r.get("config_extra") or {})
        tps = (r.get("short") or {}).get("gen_tps") or \
              (r.get("deep") or {}).get("gen_tps")
        vram = r.get("min_observed_free_mib")
        if tps is None and vram is None:
            continue
        runs.append({"axes": _axis_values(cfg),
                     "tps": float(tps) if tps else None,
                     "vram": float(vram) if vram else None})

    stats: dict[str, dict] = {}
    for key in CONFIG_KEY:
        tp, vp, vals = [], [], set()
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                diff = [k for k in set(a["axes"]) | set(b["axes"])
                        if a["axes"].get(k) != b["axes"].get(k)]
                if diff != [key]:
                    continue          # отличаются не только этой осью
                va, vb = a["axes"].get(key), b["axes"].get(key)
                if va is None or vb is None:
                    continue
                vals.add(va)
                vals.add(vb)
                for field, sink in (("tps", tp), ("vram", vp)):
                    x, y = a[field], b[field]
                    if x is None or y is None or min(x, y) <= 0:
                        continue
                    sink.append(max(x, y) / min(x, y))
        if not vals:
            continue
        s = stats.setdefault(key, {"pairs": 0, "tps_gain": None,
                                   "vram_gain": None, "values": len(vals)})
        s["pairs"] = len(tp) + len(vp)
        s["values"] = len(vals)
        s["tps_gain"] = round(max(tp), 2) if len(tp) >= min_pairs else None
        s["vram_gain"] = round(max(vp), 2) if len(vp) >= min_pairs else None

    # spec-ось измеряется отдельно (A/B), а не в общей сетке
    ab = spec_effect(runs)
    if "spec" in CONFIG_KEY or True:
        s = stats.setdefault("spec", {"pairs": 0, "tps_gain": None,
                                      "vram_gain": None, "values": 0,
                                      "mode": "ab"})
        if ab:
            s["unique"] = ab.get("unique", {}).get("gain")
            s["repetitive"] = ab.get("repetitive", {}).get("gain")
            s["pairs"] = sum(v["pairs"] for v in ab.values())
            s["values"] = 2 if s["repetitive"] else 0
            # в линейную сетку ось не идёт, но приоритет отражает повторы
            s["tps_gain"] = s["repetitive"]
    return stats


def order(learned: dict | None = None, present: set[str] | None = None) -> list[str]:
    """Ключи осей в порядке проверки: приоритет, скорректированный замерами.

    Правило: измеренный эффект сильнее доверяем, чем константу, но не
    позволяем одному шумному прогоду переставить всё — нужно минимум
    min_samples наблюдений и вес ограничен.
    """
    learned = learned if learned is not None else learn()
    scored = []
    for a in AXES:
        key = a["key"]
        if a.get("mode") == "ab" and present is None:
            # в сетку не идёт: эффект знакопеременный, решается A/B-прогоном
            continue
        if present is not None:
            # именно скобки: в Python `&` приоритетнее `|`, и без них
            # объединение с непустым множеством алиасов всегда истинно —
            # фильтр не отсекал ничего (проверено: order(present={...})
            # возвращал все оси)
            names = {a["flag"].lstrip("-")}
            names.update(x.lstrip("-") for x in a.get("alias", []))
            have = {f.lstrip("-") for f in present}
            if not (names & have):
                continue
        gain = float(a["gain"])
        ev = learned.get(key)
        note = ""
        if ev and ev.get("pairs", 0) >= 2 and ev["values"] > 1:
            eff = max(ev.get("tps_gain") or 1.0, 1.0) * \
                max(ev.get("vram_gain") or 1.0, 1.0) ** 0.5
            # логарифм сдерживает разброс: ×2 — это много, ×10 — неправда
            bump = max(-2.0, min(2.0, 2.0 * (eff ** 0.5 - 1.0)))
            gain += bump
            note = (f"изолированных пар: {ev['pairs']}, "
                    f"tps ×{ev.get('tps_gain')}, VRAM ×{ev.get('vram_gain')}")
        scored.append((gain, key, note))
    scored.sort(key=lambda x: -x[0])
    return [k for _, k, _ in scored]


def explain(learned: dict | None = None) -> list[dict]:
    """Человекочитаемая таблица: приоритет, причина, что говорят замеры.

    В отличие от `order()`, включает оси с `mode="ab"`: они не идут в сетку,
    но человеку их видеть надо — иначе самый большой рычаг выглядит как
    отсутствующий.
    """
    learned = learned if learned is not None else learn()
    rows = []
    grid = order(learned)
    ab_keys = [a["key"] for a in sorted(
        (a for a in AXES if a.get("mode") == "ab"),
        key=lambda a: -a["gain"])]
    for key in ab_keys + grid:
        a = BY_KEY[key]
        ev = learned.get(key) or {}
        rows.append({
            "key": key,
            "flag": a["flag"],
            "prior": a["gain"],
            "mode": a.get("mode", "grid"),
            "measured": a.get("measured"),
            "why": a["why"],
            "pairs": ev.get("pairs", 0),
            "values": ev.get("values", 0),
            "tps_gain": ev.get("tps_gain"),
            "vram_gain": ev.get("vram_gain"),
        })
    return rows