"""Сбор замеров: из прогонов tune_models.py и из комментариев models.ini.

Зачем: замеры уже существуют, но лежат в двух несвязанных местах —
в tune-results/*/results.json (машиночитаемо) и в комментариях над
секциями models.ini (только для человека). Здесь они сводятся в один
файл measurements.json, привязанный к конфигурации пресета, чтобы
дальше считать бюджет VRAM и рекомендовать пресеты по факту, а не на глаз.
"""

import hashlib
import json
import time
from pathlib import Path

from . import paths

# какие поля конфигурации реально влияют на скорость/память
SIGNIFICANT = ("c", "n-cpu-moe", "n-gpu-layers", "ngl", "b", "batch-size",
               "ubatch-size", "ub", "t", "threads", "threads-batch", "tb",
               "parallel", "np", "fa", "flash-attn", "cache-type-k", "ctk",
               "cache-type-v", "ctv", "kv-unified", "kvu", "spec-type",
               "model-draft", "mmproj", "load-mode", "image-min-tokens",
               "ctx-checkpoints", "checkpoint-min-step", "cache-reuse",
               "fit", "n-predict")


def signature(config: dict[str, str]) -> str:
    """Стабильный ключ конфигурации: нечувствителен к порядку и регистру."""
    norm = {}
    for k, v in config.items():
        if k.lower() in SIGNIFICANT:
            norm[k.lower()] = str(v).strip()
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _tune_dirs() -> list[Path]:
    """Каталоги с results.json. Ищет вглубь: tiel-apply/<ts-pid>/results.json."""
    env = paths.home() / ".config" / "llama" / "tune-results"
    if not env.is_dir():
        return []
    return sorted(f.parent for f in env.glob("**/results.json"))


def ingest_tune_results(dirs: list[Path] | None = None) -> dict:
    """Достаёт успешные прогоны из results.json. Возвращает записи по подписи."""
    dirs = dirs if dirs is not None else _tune_dirs()
    out: dict[str, dict] = {}
    for d in dirs:
        f = d / "results.json"
        if not f.exists():
            continue
        try:
            runs = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(runs, list):
            continue
        for r in runs:
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            cfg = dict(r.get("config") or {})
            cfg.update(r.get("config_extra") or {})
            if not cfg:
                continue
            sig = signature(cfg)
            rec = out.setdefault(sig, {
                "signature": sig, "config": cfg, "runs": 0,
                "short_tps": None, "deep_tps": None,
                "min_free_mib": None, "min_ram_mib": None,
                "needle_ok": None, "prefill_tps": None, "sources": [],
            })
            rec["runs"] += 1
            rec["sources"].append(str(f))
            short, deep = r.get("short") or {}, r.get("deep") or {}
            for key, src in (("short_tps", short), ("deep_tps", deep)):
                if src.get("gen_tps") is not None:
                    prev = rec[key]
                    val = float(src["gen_tps"])
                    # берём худший замер: он честнее для планирования
                    rec[key] = val if prev is None else min(prev, val)
            if src.get("prefill_tps") is not None:
                prev = rec["prefill_tps"]
                rec["prefill_tps"] = (float(src["prefill_tps"]) if prev is None
                                      else min(prev, float(src["prefill_tps"])))
            if src.get("needle_ok") is False:
                rec["needle_ok"] = False
            elif src.get("needle_ok") is True and rec["needle_ok"] is None:
                rec["needle_ok"] = True
            if r.get("min_observed_free_mib") is not None:
                v = int(r["min_observed_free_mib"])
                rec["min_free_mib"] = v if rec["min_free_mib"] is None \
                    else min(rec["min_free_mib"], v)
            if r.get("min_ram_available_mib") is not None:
                v = int(r["min_ram_available_mib"])
                rec["min_ram_mib"] = v if rec["min_ram_mib"] is None \
                    else min(rec["min_ram_mib"], v)
    return out


def from_ini_annotations(ini_path: str | Path) -> dict[str, dict]:
    """Замеры, написанные в комментариях над секциями models.ini."""
    from . import presets
    from .inifile import IniFile
    ini = IniFile.load(ini_path)
    ann = presets.annotate_from_ini(ini)
    out: dict[str, dict] = {}
    for name, rec in ann.items():
        sec = ini.section(name)
        if not sec:
            continue
        sig = signature(sec.pairs())
        out[sig] = {"signature": sig, "config": sec.pairs(),
                    "source": "models.ini:comment", **rec}
    return out


def build_store(ini_path: str | Path | None = None) -> dict:
    """Собирает сводное хранилище замеров."""
    tune = ingest_tune_results()
    ini_recs = {}
    ini_path = ini_path or paths.default_ini()
    try:
        ini_recs = from_ini_annotations(ini_path)
    except OSError:
        pass
    merged = dict(tune)
    for sig, rec in ini_recs.items():
        if sig in merged:
            for k, v in rec.items():
                if k in ("signature", "config", "source"):
                    continue
                if v is not None and merged[sig].get(k) is None:
                    merged[sig][k] = v
            merged[sig].setdefault("sources", []).append(rec.get("source", "models.ini"))
        else:
            rec.setdefault("runs", 0)
            rec.setdefault("sources", [rec.get("source", "models.ini")])
            merged[sig] = rec
    return {
        "version": 1,
        "built_at": int(time.time()),
        "ini": str(ini_path),
        "records": merged,
    }


def store_path() -> Path:
    return paths.state_dir() / "measurements.json"


def save_store(store: dict) -> Path:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_store() -> dict:
    p = store_path()
    if not p.exists():
        return {"version": 1, "records": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "records": {}}


def lookup(preset_config: dict[str, str], store: dict | None = None) -> dict | None:
    """Ищет замеры по конфигурации пресета (точное совпадение подписи)."""
    store = store or load_store()
    return store.get("records", {}).get(signature(preset_config))
