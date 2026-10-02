"""Живой замер потребления VRAM загруженной моделью.

Зачем: оценка `llamastery budget` опирается на подогнанные константы и на
12 ГБ карте всегда ошибается в меньшую сторону на десятки процентов. Единственный
надёжный источник — фактическое потребление, снятое с карты, пока пресет
загружен. Такой замер записывается в measurements.json и дальше имеет приоритет
над оценкой в `budget` и `validate`.
"""

import json
import shutil
import subprocess
import time

from . import measure, paths

# что занимает VRAM помимо модели: контекст CUDA, буферы самого сервера
MIN_OVERHEAD_MIB = 128


def _query_nvidia_smi(field: str, index: int = 0) -> int | None:
    """Один запрос к nvidia-smi; field — memory.used / memory.total и т.д."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, f"--id={index}", f"--query-gpu={field}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def gpu_used_mib(index: int = 0) -> int | None:
    return _query_nvidia_smi("memory.used", index)


def gpu_total_mib(index: int = 0) -> int | None:
    return _query_nvidia_smi("memory.total", index)


def capture_preset(index: int = 0, allow_single: bool = False) -> dict:
    """Снимает показания и привязывает их к единственной загруженной модели."""
    from . import server
    # адрес сервера зависит от сборки (у неё может быть свой порт), а каждый
    # вызов CLI — новый процесс: без этого шага замер спрашивал бы /models
    # у дефолтного 8099 и не увидел бы загруженную модель
    pf = server.read_pid()
    if pf and server.alive(pf):
        name = server.identify(pf)
        if name:
            server.use_build(name)
    exe = shutil.which("nvidia-smi")
    used = total = None
    if exe:
        try:
            r = subprocess.run(
                [exe, f"--id={index}", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().splitlines()[0].split(",")
                used = int(parts[0].strip())
                total = int(parts[1].strip()) if len(parts) > 1 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if used is None:
        used = gpu_used_mib(index)
        total = gpu_total_mib(index)
    if used is None:
        return {"ok": False, "error": "nvidia-smi недоступен"}
    loaded = [m for m in server.models()
              if m["status"] in ("loaded", "sleeping")]
    if not loaded and allow_single:
        loaded = server.loaded_models()
    return {
        "ok": True,
        "used_mib": used,
        "free_mib": (total - used) if total else None,
        "total_mib": total,
        "loaded": [m["id"] for m in loaded],
        "at": int(time.time()),
    }


def record(name: str, pairs: dict, cap: dict, index: int = 0) -> dict:
    """Записывает замер в measurements.json по подписи конфигурации."""
    sig = measure.signature(pairs)
    store = measure.load_store()
    store.setdefault("version", 1)
    store.setdefault("records", {})
    rec = store["records"].setdefault(sig, {"signature": sig,
                                            "config": pairs, "runs": 0,
                                            "sources": []})
    rec["min_free_mib"] = cap["free_mib"]
    rec["used_mib"] = cap["used_mib"]
    # «занято» = всего минус свободно — та же величина, что в annotate и в
    # сообщениях сервера, чтобы цифры в разных местах не расходились
    rec["vram_mib"] = cap["used_mib"]
    rec["model_mib"] = max(0, cap["used_mib"] - MIN_OVERHEAD_MIB)
    rec["vram_source"] = "живой замер nvidia-smi"
    rec["measured_at"] = cap["at"]
    rec["runs"] = rec.get("runs", 0) + 1
    rec.setdefault("sources", []).append(f"nvidia-smi:{cap['at']}")
    if cap.get("loaded"):
        rec["model_ids"] = cap["loaded"]
    p = measure.save_store(store)
    rec["store"] = str(p)
    return rec


def calibrate_from_measurement(pairs: dict, meta_used_mib: int,
                              structural_gb: float) -> dict | None:
    """Пересчитывает compute buffer по живому замеру конкретного пресета."""
    from . import budget
    used_gb = max(0.0, (meta_used_mib - MIN_OVERHEAD_MIB) / 1024.0)
    residual = used_gb - structural_gb
    if residual < 0:
        residual = 0.0
    cal = budget.load_calibration()
    cur = cal.get("compute_gb")
    # берём консервативное из двух: занизить compute опаснее, чем завысить
    new = residual if cur is None else min(cur, residual)
    cal["compute_gb"] = round(new, 3)
    # константа измерена при СВОЁМ ubatch этого пресета — опорный тоже меняем,
    # иначе последующие оценки пересчитают масштабирование второй раз
    ub = budget._flag_int(pairs, "ubatch-size", "ub", default=0) or 0
    if ub:
        cal["compute_ref_ubatch"] = ub
    cal["vram_total_mib"] = gpu_total_mib() or cal.get("vram_total_mib")
    cal["compute_source"] = ("живой замер пресета: "
                             f"{meta_used_mib} MiB, структура {structural_gb:.2f} GiB, "
                             f"остаток {residual:.2f} GiB при ub={ub or '?'}"
                             + (f"; было {cur:.2f} GiB, оставлен меньший"
                                if cur is not None else ""))
    p = budget.save_calibration(cal)
    return {"compute_gb": cal["compute_gb"], "residual": round(residual, 3),
            "store": str(p)}