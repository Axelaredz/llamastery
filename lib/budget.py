"""Оценка потребления VRAM и проверка, влезает ли набор пресетов.

Принципиальная оговорка: точный расчёт памяти делает внутри себя
llama.cpp, и он зависит от аллокатора CUDA, unified KV, SSM-состояния,
размера compute buffer и графиков. Поэтому здесь НЕAttempts точная
арифметика, а разложение на слагаемые, где структурные части считаются,
а compute buffer — подгоняемая константа, калибруемая по факту.

Что считается точно:
  * веса — размер GGUF минус доля, ушедшая в RAM из-за n-cpu-moe
  * KV-кэш — по слоям внимания (у гибридных архитектур вроде qwen35moe
    полное внимание только каждый full_attention_interval-й слой)
  * mmproj — размер файла проектора (или 0 при mmproj-offload = 0)
"""

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import paths
from .gguf import ModelMeta


def measure_lookup(pairs: dict[str, str]) -> dict | None:
    """Живой замер по конфигурации пресета (ленивый импорт: measure → budget)."""
    from . import measure
    try:
        return measure.lookup(pairs)
    except Exception:  # noqa: BLE001 — замеры опциональны
        return None

GIB = 2 ** 30
MIB = 2 ** 20

# Проверенный порог: связка «ускоритель + ubatch > 1024» на 12 ГБ роняет
# CUDA по compute buffer. Число, а не доля от калибровки: перекалибровка не
# должна молча отключать предупреждение о подтверждённом падении.
SPEC_UBATCH_LIMIT = 1024

# размер элемента KV-кэша в байтах по имени типа
KV_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 1.0625, "q8_1": 1.125,
            "q6_0": 0.8125, "q5_1": 0.6875, "q5_0": 0.6875, "q4_1": 0.5625,
            "q4_0": 0.5625, "iq4_nl": 0.5625, "iq4_xs": 0.5625}


def kv_bytes_for(kind: str) -> float:
    return KV_BYTES.get((kind or "f16").lower(), 2.0)


@dataclass
class Estimate:
    weights_gb: float = 0.0
    kv_gb: float = 0.0
    mmproj_gb: float = 0.0
    compute_gb: float = 0.0     # подогнанная константа, не вычисляется
    total_gb: float = 0.0
    cpu_moe: int = 0
    n_layer: int = 0
    n_attn_layer: int = 0
    ctx: int = 0
    model: str = ""
    notes: list[str] = field(default_factory=list)
    measured_min_free_mib: int | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["total_gb"] = round(self.total_gb, 2)
        for k in ("weights_gb", "kv_gb", "mmproj_gb", "compute_gb"):
            d[k] = round(d[k], 2)
        return d


def _norm(pairs: dict[str, str]) -> dict[str, str]:
    """Нижнерегистровые ключи один раз — вместо перестройки в каждом геттере."""
    return {k.lower(): v for k, v in pairs.items()}


def _get_int(low: dict[str, str], *names: str, default=None):
    for n in names:
        if n in low:
            try:
                return int(str(low[n]).strip())
            except (TypeError, ValueError):
                return default
    return default


def _get_str(low: dict[str, str], *names: str, default=""):
    for n in names:
        if n in low and str(low[n]).strip() != "":
            return str(low[n]).strip()
    return default


def _flag_int(pairs: dict[str, str], *names: str, default=None):
    return _get_int(_norm(pairs), *names, default=default)


def _flag_str(pairs: dict[str, str], *names: str, default=""):
    return _get_str(_norm(pairs), *names, default=default)


def attention_layers(meta: ModelMeta) -> int:
    """Сколько слоев реально хранят KV-кэш.

    У гибридов (qwen35moe, Nemotron-H, Jamba) часть слоев — SSM/Mamba:
    их состояние не растёт с контекстом, KV они не хранят.
    """
    n = meta.n_layer
    interval = int(meta.kv.get(f"{meta.arch}.full_attention_interval", 0) or 0)
    if interval and interval > 1 and n:
        return max(1, math.ceil(n / interval))
    return n


def _mtp_active(pairs: dict[str, str]) -> bool:
    """Встроенная MTP-голова (хвостовые nextn-слои) реально загружается?"""
    low = _norm(pairs)
    spec = _get_str(low, "spec-type", "spec", default="").lower()
    draft = _get_str(low, "model-draft", "draft", default="").strip()
    return bool(draft) or "draft-mtp" in spec


_BLK_RE = re.compile(r"blk\.(\d+)\.")


@lru_cache(maxsize=128)
def _compile_ot_entries(raw: str) -> tuple[tuple[str, str, bool], ...]:
    """Компилирует -ot один раз на уникальную строку (кэш между пресетами).

    Хранит (pattern, device, is_cpu); сам regex компилируется при
    использовании и тоже кэшируется через re-модуль. Битые паттерны
    молча пропускаются — как и раньше.
    """
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        pat, dev = chunk.split("=", 1)
        pat, dev = pat.strip(), dev.strip()
        if not pat:
            continue
        try:
            re.compile(pat)
        except re.error:
            continue
        out.append((pat, dev, dev.lower() == "cpu"))
    return tuple(out)


def override_cpu_bytes(pairs: dict[str, str], meta: ModelMeta | None,
                       _sizes: list[tuple[str, int]] | None = None
                       ) -> tuple[float, int, str]:
    """Байты весов, уходящие в RAM через override-tensor (флаг -ot).

    Семантика повторяет llama.cpp (llama-model-loader.cpp): запятые делят
    записи pattern=device, сопоставление — regex_search, на тензор действует
    ПЕРВОЕ совпавшее правило. В VRAM не считаются только device=CPU.
    Возвращает (байты, число тензоров, заметка).
    """
    if meta is None or not meta.tensors:
        return 0.0, 0, ""
    raw = _flag_str(pairs, "override-tensor", "ot", default="").strip()
    if not raw:
        return 0.0, 0, ""
    compiled = [(pat, re.compile(pat), is_cpu)
                for pat, _dev, is_cpu in _compile_ot_entries(raw)]
    entries = [(pat, rx, is_cpu) for pat, rx, is_cpu in compiled]
    if not entries:
        return 0.0, 0, ""
    cpu_bytes, matched = 0, 0
    for name, size in (_sizes if _sizes is not None else meta.tensor_sizes()):
        for pat, rx, is_cpu in entries:
            try:
                hit = rx.search(name) is not None
            except re.error:
                hit = False
            if hit:
                if is_cpu:
                    cpu_bytes += size
                    matched += 1
                break
    note = ""
    if matched:
        pats = ", ".join(f"{p}=CPU" if c else f"{p}=…" for p, _, c in entries)
        note = (f"override-tensor [{pats}] → в RAM уходит "
                f"{cpu_bytes / GIB:.2f} GiB ({matched} тензоров)")
    return float(cpu_bytes), matched, note


def expert_weight_share(meta: ModelMeta) -> float:
    """Доля весов, приходящаяся на экспертов MoE (для оценки n-cpu-moe)."""
    if not meta.is_moe or not meta.n_embd or not meta.n_layer:
        return 0.0
    a = meta.arch
    ffn = int(meta.kv.get(f"{a}.expert_feed_forward_length", 0) or 0)
    if not ffn:
        return 0.9  # нет данных — для MoE почти всё и так в экспертах
    shared = int(meta.kv.get(f"{a}.expert_shared_feed_forward_length", ffn) or ffn)
    e = meta.n_embd
    experts = meta.n_expert * 3 * e * ffn
    dense = 3 * e * shared                       # общий (shared) FFN
    attn = e * (e + 2 * meta.n_head_kv * meta.head_dim
                + meta.n_head * meta.v_head_dim)
    total = experts + dense + attn               # всё в расчёте на один слой
    return experts / total if total else 0.0


def estimate(pairs: dict[str, str], meta: ModelMeta | None,
             compute_gb: float | None = None,
             mmproj_meta: ModelMeta | None = None,
             _cal: dict | None = None) -> Estimate:
    """Оценка VRAM для одного пресета.

    _cal — уже загруженная калибровка (чтобы цикл по N пресетам не читал
    JSON-файл N раз). Если None — читается с диска, как раньше.
    """
    est = Estimate(model=meta.path.name if meta else "?")
    cal = _cal if _cal is not None else load_calibration()
    low = _norm(pairs)
    if compute_gb is None:
        compute_gb = cal.get("compute_gb", 0.0)
    # Калибровка compute_gb — это остаток (всего минус веса минус KV), а не сам
    # compute buffer, поэтому в неё уже входят постоянные накладные расходы:
    # CUDA-контекст, pinned host-буферы, запас аллокатора. Они от ubatch не
    # зависят, и реальный compute buffer тоже растёт медленно: сервер сам
    # сообщил 978 MiB при ub=1024, тогда как остаток при ub=512 — 0.99 GiB.
    # Прежняя формула (40% постоянная + 60% ∝ ubatch) давала на ub=2048
    # 2.76 GiB и объявляла заведомо рабочий пресет невмещающимся.
    # Рост теперь затухающий; точный наклон не откалиброван (одна точка), поэтому
    # формула подобрана так, чтобы остаться ближе к измеренному, а не завышать.
    ref_ub = cal.get("compute_ref_ubatch") or 0
    ub = _get_int(low, "ubatch-size", "ub", default=0) or 0
    if compute_gb and ref_ub and ub and ub != ref_ub:
        scaled = compute_gb * (0.9 + 0.1 * ub / ref_ub)
        est.notes.append(
            f"compute buffer {compute_gb:.2f} GiB при ub={ref_ub} → {scaled:.2f} GiB "
            f"при ub={ub} (рост затухающий; наклон откалиброван по одной точке)")
        compute_gb = scaled

    # Проверенный порог для связки «ускоритель + ubatch» на 12 ГБ:
    # 1024 работает, 2048 падает. Не зависит от калибровки.
    # Спекулятивный декодер расширяет граф верификации, и llama.cpp не умеет
    # расширять буфер в середине запроса: при глубоком контексте CUDA падает.
    # Проверено на Tiel-Coder NanoPlus (moe16, c=114688, mmproj): ubatch=2048
    # роняет out of memory в середине запроса, ubatch=1024 работает.
    # Порог фиксирован, а не считается от калибровки. Раньше здесь было
    # `ub > 2 * ref_ub`, и после `calibrate --from-log` (compute_ref_ubatch
    # сдвинулся с 512 на 1024) значение 2048 перестало помечаться — при том
    # что именно 2048 подтверждённо роняет сервер. Эвристика о молчании при
    # перекалибровке опаснее неточности.
    spec = _get_str(low, "spec-type", "spec", default="").lower()
    if spec and ub > SPEC_UBATCH_LIMIT:
        est.notes.append(
            f"ВНИМАНИЕ: spec-type={spec} при ubatch={ub} — на глубоком контексте "
            f"падение CUDA по compute buffer (проверено на Tiel-Coder NanoPlus, "
            f"moe16, c=114688: ubatch={SPEC_UBATCH_LIMIT} работает, выше нет)")

    if meta is None:
        est.compute_gb = compute_gb
        est.total_gb = compute_gb
        est.notes.append("метаданные модели недоступны — оценка неполная")
        return est

    est.n_layer = meta.n_layer
    est.n_attn_layer = attention_layers(meta)
    if meta.n_layer_nextn and est.n_attn_layer > meta.n_layer_nextn:
        # хвостовые MTP-слои (nextn) KV-кэш не хранят
        est.n_attn_layer -= meta.n_layer_nextn
        est.notes.append(
            f"минус {meta.n_layer_nextn} MTP-слоёв без KV: "
            f"слоёв внимания {est.n_attn_layer}")
    est.ctx = _get_int(low, "c", "ctx-size", default=meta.n_ctx_trained) or 0
    est.cpu_moe = _get_int(low, "n-cpu-moe", "ncmoe", "cmoe", default=0) or 0

    # ── веса ──
    # Структурная доля экспертов (аналитика по метаданным) завышает эффект
    # примерно в 1.3 раза: реально в RAM уезжает не вся аналитическая доля.
    # Коэффициент калибруется по замерам (см. llamastery calibrate --from-tune).
    realization = cal.get("offload_realization", 0.76)
    weights = meta.size_bytes / GIB
    # таблицу тензоров строим один раз — она нужна и MTP-скипу, и -ot
    sizes = meta.tensor_sizes() if meta.tensors else []
    if meta.n_layer_nextn and not _mtp_active(pairs) and sizes:
        # встроенная MTP-голова при выключенном MTP: загрузчик помечает
        # хвостовые nextn-слои как unused — в VRAM их нет
        first_nextn = meta.n_layer - meta.n_layer_nextn
        skip = 0
        for _name, _size in sizes:
            _m = _BLK_RE.match(_name)
            if _m and int(_m.group(1)) >= first_nextn:
                skip += _size
        if skip:
            weights -= skip / GIB
            est.notes.append(
                f"MTP выключен: хвостовые {meta.n_layer_nextn} nextn-слоёв "
                f"не грузятся (−{skip / GIB:.2f} GiB)")
    ot_bytes, ot_n, ot_note = override_cpu_bytes(pairs, meta, _sizes=sizes)
    if ot_bytes:
        # точный учёт через таблицу тензоров — вместо эвристики n-cpu-moe
        weights -= ot_bytes / GIB
        est.notes.append(ot_note)
        if est.cpu_moe:
            est.notes.append(
                "n-cpu-moe проигнорирован: override-tensor уже учтён точно")
    elif est.cpu_moe and meta.n_layer:
        share = expert_weight_share(meta)
        frac = min(1.0, (est.cpu_moe / meta.n_layer) * share * realization)
        weights *= (1.0 - frac)
        per_layer = meta.size_bytes / GIB * share * realization / meta.n_layer
        est.notes.append(
            f"n-cpu-moe={est.cpu_moe}/{meta.n_layer} → в RAM уходит "
            f"~{per_layer * est.cpu_moe:.2f} GiB ({per_layer:.3f} GiB на слой, "
            f"коэффициент {realization})")
    ngl = _get_int(low, "n-gpu-layers", "ngl", "gpu-layers")
    if ngl is not None and 0 <= ngl < meta.n_layer:
        weights *= (ngl + 1) / meta.n_layer
        est.notes.append(f"n-gpu-layers={ngl} → только {ngl + 1}/{meta.n_layer} слоёв в VRAM")
    est.weights_gb = weights

    # ── KV-кэш ──
    ctk = kv_bytes_for(_get_str(low, "cache-type-k", "ctk", default="f16"))
    ctv = kv_bytes_for(_get_str(low, "cache-type-v", "ctv", default="f16"))
    np_ = _get_int(low, "parallel", "np", default=1) or 1
    kv_unified = _get_str(low, "kv-unified", "kvu", default="")
    if meta.is_mla:
        # MLA (DeepSeek-стиль, xing4_0): KV — одна строка на слой,
        # latent (kv_lora_rank) + rope-часть. В llama.cpp весь MLA-кэш —
        # один тензор типа cache-type-k, отдельной V нет.
        per_token = (est.n_attn_layer
                     * (meta.kv_lora_rank + meta.rope_dim) * ctk)
        est.notes.append(
            f"MLA-KV: {meta.kv_lora_rank}+{meta.rope_dim} эл/токен/слой "
            f"(тип {ctk})")
    else:
        per_token = est.n_attn_layer * meta.n_head_kv * (ctk * meta.head_dim
                                                         + ctv * meta.v_head_dim)
    if np_ > 0 and kv_unified.lower() in ("off", "0", "false", "disabled"):
        slots = np_
    elif np_ < 0:
        slots = 1        # -1 = авто; при unified cache растёт не слотами
        est.notes.append("parallel=-1 (auto) — число слотов принято за 1")
    else:
        slots = 1        # unified: общий пул на весь контекст
    est.kv_gb = per_token * est.ctx * slots / GIB
    if slots > 1:
        est.notes.append(f"KV ×{slots} слотам (kv-unified выключен)")

    # ── mmproj ──
    if mmproj_meta is not None:
        offload = _get_str(low, "mmproj-offload", default="1").lower()
        if offload not in ("0", "off", "false", "disabled"):
            est.mmproj_gb = mmproj_meta.size_bytes / GIB

    est.compute_gb = compute_gb
    est.total_gb = est.weights_gb + est.kv_gb + est.mmproj_gb + est.compute_gb
    est.notes.append(
        f"compute buffer {compute_gb:.2f} GiB — подогнанная константа, "
        f"а не расчёт (см. llamastery budget calibrate)")
    return est


# ── впихнуть пресет в VRAM ──
MIN_CTX = 4096                    # ниже тюнер всё равно не работает
CTX_STEPS = (262144, 131072, 114688, 98304, 65536, 49152, 32768, 16384,
             8192, MIN_CTX)


@dataclass
class Shrink:
    """Чем пресет пришлось урезать, чтобы он влез в VRAM."""

    ctx: int = 0
    cpu_moe: int = 0
    n_layer: int = 0
    fits: bool = False
    est: Estimate = field(default_factory=Estimate)
    steps: list[str] = field(default_factory=list)


def _with(pairs: dict[str, str], ctx: int | None = None,
          moe: int | None = None) -> dict[str, str]:
    """Копия пресета с подменёнными c и n-cpu-moe (по именам, что в нём есть)."""
    p = dict(pairs)
    if ctx is not None:
        low = _norm(p)
        for k in ("c", "ctx-size"):
            if k in low:
                p[k] = str(ctx)
                break
        else:
            p["c"] = str(ctx)
    if moe is not None:
        low = _norm(p)
        key = next((k for k in ("n-cpu-moe", "ncmoe", "cmoe") if k in low), None)
        if key is not None:
            p[key] = str(moe)
        elif moe:
            p["n-cpu-moe"] = str(moe)
    return p


def shrink_to_fit(pairs: dict[str, str], meta: ModelMeta | None,
                  total_mib: int | None, reserve_mib: int = 1024,
                  mmproj_meta: ModelMeta | None = None,
                  _cal: dict | None = None, min_ctx: int = MIN_CTX) -> Shrink:
    """Подобрать c и n-cpu-moe так, чтобы пресет влез в VRAM.

    Порядок жертв: сперва контекст (до min_ctx), потом n-cpu-moe. Сознательно
    именно так: слои экспертов в RAM бьют по скорости КАЖДОГО токена, а
    короткий контекст — это просто короткий контекст, и полную глубину потом
    проверяют отдельно (`budget` + `probe --tokens <c>`).

    Числа берутся из estimate, поэтому результат совпадает с выводом
    `llamastery budget`: не может выйти «влезло по тут, а по там нет».
    Считается от estimate, а не по линейной формуле: скилл знает про
    override-tensor, n-gpu-layers и прочие сюрпризы, которые ломают арифметику.

    Живой замер важнее оценки: если пресет уже мерился и влезает по факту,
    он возвращается нетронутым — иначе прогноз, ошибающийся на 5 GiB,
    заставил бы мастер урезать рабочий пресет.
    """
    cal = _cal if _cal is not None else load_calibration()
    cur_ctx = _flag_int(pairs, "c", "ctx-size", default=0) or 0
    cur_moe = _flag_int(pairs, "n-cpu-moe", "ncmoe", "cmoe", default=0) or 0
    if meta is None or not total_mib:
        return Shrink(ctx=cur_ctx, cpu_moe=cur_moe)
    usable_mib = total_mib - reserve_mib
    max_moe = max(0, meta.n_layer - 1) if meta.is_moe else 0

    meas = measure_lookup(pairs)
    if meas and meas.get("used_mib"):
        # факт: 10800 MiB против оценки 16.06 GiB на том же пресете
        if float(meas["used_mib"]) <= usable_mib:
            return Shrink(ctx=cur_ctx, cpu_moe=cur_moe,
                          n_layer=meta.n_layer, fits=True,
                          steps=["пресет уже влезает (живой замер "
                                 f"{meas['used_mib']} MiB)"])
    elif meas and meas.get("min_free_mib") is not None:
        if float(meas["min_free_mib"]) >= reserve_mib:
            return Shrink(ctx=cur_ctx, cpu_moe=cur_moe,
                          n_layer=meta.n_layer, fits=True,
                          steps=["пресет уже влезает (живой замер: свободно "
                                 f"{meas['min_free_mib']} MiB)"])

    def need(ctx: int, moe: int) -> float:
        est = estimate(_with(pairs, ctx=ctx, moe=moe), meta,
                       mmproj_meta=mmproj_meta, _cal=cal)
        return est.total_gb * 1024.0

    def moe_for(ctx: int) -> int | None:
        """Минимальный n-cpu-moe, при котором ctx влезает (None — не влезает).

        Потребление монотонно падает с ростом moe, поэтому делим отрезок.
        """
        lo, hi, found = 0, max_moe, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if need(ctx, mid) <= usable_mib:
                found = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return found

    # контекст важнее скорости: сначала ищем максимальный, который влезает
    best: tuple[int, int] | None = None
    for ctx in CTX_STEPS:
        if cur_ctx and ctx > cur_ctx:
            continue
        if ctx < min_ctx:
            break
        moe = moe_for(ctx)
        if moe is not None:
            best = (ctx, moe)
            break
    ctx, moe = best if best is not None else (min_ctx, max_moe)

    est = estimate(_with(pairs, ctx=ctx, moe=moe), meta,
                   mmproj_meta=mmproj_meta, _cal=cal)
    out = Shrink(ctx=ctx, cpu_moe=moe, n_layer=meta.n_layer,
                 fits=est.total_gb * 1024.0 <= usable_mib, est=est)
    if ctx != cur_ctx:
        out.steps.append(f"c {cur_ctx} → {ctx}")
    if moe != cur_moe:
        out.steps.append(f"n-cpu-moe {cur_moe} → {moe}")
    if not out.steps:
        out.steps.append("пресет уже влезает")
    if not out.fits:
        out.steps.append(
            f"даже c={ctx} и n-cpu-moe={moe} не хватает: "
            f"{est.total_gb:.2f} GiB против доступных {usable_mib / 1024.0:.2f} GiB")
    return out


# ── калибровка ──
def calibration_path():
    return paths.calibration_file()


def load_calibration() -> dict:
    p = calibration_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_calibration(data: dict):
    p = calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def gpu_total_mib() -> int | None:
    """Сколько VRAM на карте (nvidia-smi)."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def fit(pairs_of: list[tuple[str, dict]], measured: list[float],
        structural: list[float]) -> float | None:
    """Подбирает compute_gb так, чтобы оценка совпала с замерами.

    pairs_of не используется, но оставлен для единообразия вызова.
    """
    diffs = [m - s for m, s in zip(measured, structural) if m > 0]
    if not diffs:
        return None
    diffs.sort()
    return max(0.0, diffs[len(diffs) // 2])
