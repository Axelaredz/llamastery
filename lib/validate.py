"""Валидация пресетов: схема флагов, пути, ловушки конкретных форков.

Проверки, которые ловят реальные ошибки, а не опечатки:
  * ключ не существует в этой сборке (флаг появился/переименован)
  * пресет переопределяет control-аргумент роутера (их роутер вырежет)
  * путь к GGUF/mmproj/model-draft не существует
  * известные грабли форков (см. KNOWN_TRAPS)
  * c > обученного контекста модели
  * n-cpu-moe > числа слоёв
  * n-predict/parallel против запаса VRAM (мягко, через budget)
"""

from dataclasses import dataclass, field

from . import budget, gguf, schema

# Аргументы, которыми роутер управляет сам. Источник истины —
# tools/server/server-models.cpp, функция unset_reserved_args().

# вырезаются из любого пресета
HARD_CONTROL = {"ssl-key-file", "ssl-cert-file", "api-key", "models-dir",
                "models-max", "models-preset", "models-autoload"}

# перезаписываются роутером при спавне дочернего процесса
ROUTER_FORCED = {"port", "host", "alias"}

# в per-model пресете эти задают САМ модель (так их и грузят из кэша),
# поэтому легальны; вырезаются только из базового пресета роутера
MODEL_DEFINING = {"model", "m", "mmproj", "hf-repo", "hf-file"}

TRUE = {"1", "true", "yes", "on", "enabled"}
FALSE = {"0", "false", "no", "off", "disabled"}


@dataclass
class Finding:
    level: str            # error | warn | info
    section: str
    key: str
    message: str
    hint: str = ""

    def line(self) -> str:
        tag = {"error": "ОШИБКА", "warn": "ВНИМАНИЕ", "info": "инфо"}[self.level]
        s = f"[{tag}] {self.section}: {self.key} — {self.message}"
        if self.hint:
            s += f"\n         подсказка: {self.hint}"
        return s


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, section: str, key: str, msg: str, hint: str = ""):
        self.findings.append(Finding(level, section, key, msg, hint))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)

    @property
    def ok(self) -> bool:
        return self.count("error") == 0


def validate_section(name: str, pairs: dict[str, str], flags: dict,
                     rep: Report, check_paths: bool = True,
                     model_cache: dict | None = None) -> gguf.ModelMeta | None:
    """Проверяет одну секцию. Возвращает метаданные модели, если нашлись."""
    low = {k.lower(): v for k, v in pairs.items()}
    meta = None

    for key, val in pairs.items():
        k = key.lower()
        if k in HARD_CONTROL:
            rep.add("error", name, key,
                    "роутер вырезает этот аргумент из пресета",
                    "убери из пресета; значение задаётся только CLI роутера")
            continue
        if k in ROUTER_FORCED:
            rep.add("warn", name, key,
                    "роутер всё равно подставит своё значение",
                    "в пресете смысла нет; лишнее только путает при чтении")
            continue
        f = schema.resolve(flags, k)
        if f is None:
            rep.add("warn", name, key, "такого флага нет в этой сборке",
                    "проверь `llama-server --help`; в форках флаги добавляют "
                    "и переименовывают")
            continue
        if f.canonical.lstrip("-") in HARD_CONTROL:
            rep.add("error", name, key,
                    "роутер вырезает этот аргумент из пресета",
                    "убери из пресета; значение задаётся только CLI роутера")
            continue
        if f.canonical.lstrip("-") in ROUTER_FORCED:
            rep.add("warn", name, key,
                    "роутер всё равно подставит своё значение",
                    "в пресете смысла нет; лишнее только путает при чтении")
            continue
        # значение против enum
        if f.enum and str(val).strip():
            v = str(val).strip().lower()
            if v not in [e.lower() for e in f.enum] and v not in TRUE | FALSE \
                    and v not in ("all", "auto"):
                rep.add("warn", name, key, f"значение {val!r} не из списка {f.enum}")
        if f.kind in ("int", "float") and str(val).strip():
            try:
                float(str(val).strip())
            except ValueError:
                rep.add("error", name, key, f"ожидалось число, а стоит {val!r}")

    # ── пути ──
    model = low.get("model") or low.get("m") or ""
    if not model:
        rep.add("error", name, "model", "не указан путь к модели",
                "пресет без model не запустится; для HF укажи hf-repo")
    elif check_paths and not model.startswith(("http://", "https://")):
        if model_cache is not None and model in model_cache:
            ok, meta = model_cache[model]
        else:
            if gguf.is_gguf(model):
                try:
                    meta = gguf.probe(model)
                except gguf.GGUFError as exc:
                    rep.add("error", name, "model", str(exc))
                    meta = None
                ok = meta is not None
            else:
                rep.add("error", name, "model", "файла нет или это не GGUF",
                        f"проверь путь: {model}")
                ok, meta = False, None
            if model_cache is not None:
                model_cache[model] = (ok, meta)

    for key in ("mmproj", "model-draft"):
        p = low.get(key)
        if p and check_paths and not p.startswith(("http://", "https://")):
            if not gguf.is_gguf(p):
                rep.add("error", name, key, "файла нет или это не GGUF", f"путь: {p}")

    if meta is None:
        return None

    # ── семантика против метаданных ──
    ctx = budget._flag_int(pairs, "c", "ctx-size")
    if ctx and meta.n_ctx_trained and ctx > meta.n_ctx_trained:
        rep.add("error", name, "c",
                f"контекст {ctx} больше обученного {meta.n_ctx_trained}",
                "модель не выучена на таком окне; качество деградирует")
    ncpu = budget._flag_int(pairs, "n-cpu-moe", "ncmoe")
    if ncpu and meta.n_layer and ncpu > meta.n_layer:
        rep.add("error", name, "n-cpu-moe",
                f"{ncpu} больше числа слоёв ({meta.n_layer})")
    ngl = budget._flag_int(pairs, "n-gpu-layers", "ngl", "gpu-layers")
    if ngl is not None and meta.n_layer and 0 <= ngl < meta.n_layer:
        rep.add("info", name, "n-gpu-layers",
                f"{ngl} из {meta.n_layer} слоёв в VRAM — часть модели в RAM",
                "медленнее генерация; осознанное решение для больших моделей")
    return meta


# ── грабли, найденные в форках ──
KNOWN_TRAPS = [
    {
        "id": "ngram-mod+draft-mtp",
        "applies": lambda p: "ngram-mod" in str(p.get("spec-type", "")) and
                             str(p.get("model-draft", "")).strip() != "",
        "message": "spec-type=ngram-mod вместе с внешним MTP-драфтом — "
                   "известный segfault на загрузке в сборке Faks",
        "hint": "убери MTP-голову из model-draft; внешний MTP сам по себе вдвое "
                "медленнее, ngram-mod — единственный рабочий ускоритель",
        "level": "error", "kind": "crash",
    },
    {
        # Порог уточнён живьём: mmproj в VRAM стоит ~0.7 GiB, и на 12 ГБ
        # при 114688 контексте этого не хватает префиллу — CUDA OOM
        # («CUDA error: out of memory» в update_slots). Наблюдалось на
        # qwen3.8-35B-A3B-miniplus-128ctx-ngram-mmproj-vram с ubatch 512.
        "id": "mmproj-vram-ubatch",
        "applies": lambda p: str(p.get("mmproj", "")).strip() != ""
                              and str(p.get("mmproj-offload", "")).strip()
                              in ("1", "true", "yes", "on")
                              and int(budget._flag_int(p, "ubatch-size", "ub") or 512) >= 512,
        "message": "mmproj в VRAM при ubatch-size >= 512 — не хватает VRAM "
                   "префиллу на 12 ГБ (проверено: CUDA OOM при 114688)",
        "hint": "либо ubatch-size = 256, либо mmproj-offload = 0 "
                "(тогда картинка кодируется на CPU в 2-3 раза дольше)",
        "level": "warn", "kind": "crash",
    },
    {
        # Порог уточнён живьём: сам по себе ubatch-size=2048 на 12 ГБ работает
        # (tiel-coder-nanoplus-128ctx-mmproj-moe16 измерен: 29.3 t/s, 1616 MiB
        # свободно). Падает он в паре с ускорителем — 2048 + ngram-mod роняет
        # CUDA по compute buffer на 110k токенах, 1024 работает.
        "id": "ubatch-2048-oom",
        "applies": lambda p: str(p.get("ubatch-size", p.get("ub", ""))) == "2048"
                              and str(p.get("spec-type", "")).strip() != "",
        "message": "ubatch-size=2048 вместе с ускорителем на 12 ГБ VRAM — "
                   "подтверждённая причина OOM",
        "hint": "снижай до 1024: проверено, что с ngram-mod при 114688 "
                "работает только 1024; без ускорителя 2048 не мешает",
        "level": "warn", "kind": "heuristic",
    },
    {
        # Тот же порог, но без ускорителя: замер показал, что 2048 достаточно
        # влезает. Остаётся информацией, а не предупреждением.
        "id": "ubatch-2048-tight",
        "applies": lambda p: str(p.get("ubatch-size", p.get("ub", ""))) == "2048"
                              and str(p.get("spec-type", "")).strip() == "",
        "message": "ubatch-size=2048 на 12 ГБ VRAM — впритык, но рабочее",
        "hint": "проверено на tiel-coder-nanoplus-128ctx-mmproj-moe16: 29.3 t/s, "
                "свободно 1616 MiB. С ускорителем это уже значение ломается",
        "level": "info", "kind": "heuristic",
    },
    {
        "id": "parallel-kv-pool",
        "applies": lambda p: (budget._flag_int(p, "parallel", "np") or 1) > 1,
        "message": "parallel > 1 кратно умножает KV-пул",
        "hint": "на 12 ГБ держи parallel=1, иначе OOM",
        "level": "warn", "kind": "heuristic",
    },
    {
        "id": "qwen-p",
        "applies": lambda p: str(p.get("temp", "")).strip() in ("0.6", "0.60") and
                             str(p.get("top-k", "")) == "40",
        "message": "температура Qwen-официальная 0.6, но top-k=40 — нет",
        "hint": "для Qwen официально temp 0.6 / top-p 0.95 / top-k 20",
        "level": "info", "kind": "style",
    },
]


def check_traps(name: str, pairs: dict[str, str], rep: Report,
                measured: dict | None = None) -> None:
    """Проверяет грабли форков.

    Если по этой конфигурации есть фактический замер, эвристика
    молчит: замер важнее общего правила.
    """
    low = {k.lower(): (v or "") for k, v in pairs.items()}
    proven = bool(measured) and measured.get("ok")
    for trap in KNOWN_TRAPS:
        try:
            hit = trap["applies"](low)
        except Exception:
            hit = False
        if not hit:
            continue
        # замер глушит только эвристики: предсказание падения он не отменяет
        if proven and trap.get("kind") == "heuristic":
            rep.add("info", name, "профиль",
                    f"{trap['message']} — но есть замер, прошедший проверку",
                    f"min free {measured.get('min_free_mib')} MiB")
            continue
        rep.add(trap["level"], name, "профиль", trap["message"], trap["hint"])


def check_crash_history(name: str, rep: Report) -> None:
    """Пресет, который ронял сервер вживую, обязан об этом предупреждать.

    Статические правила бессильны против связок вроде «ubatch 2048 + ngram-mod
    на 114688»: по схеме всё в порядке, валидатор молчит, а CUDA падает на
    первом же глубоком запросе. Единственный надёжный источник — журнал
    падений, который ведёт сам инструмент.
    """
    from . import crashes
    e = crashes.known(name)
    if not e:
        return
    when = e.get("last", "?")
    howmany = e.get("count", 1)
    rep.add("warn", name, "падение",
            f"этот пресет ронял сервер ({e.get('why')}), "
            f"{howmany} раз, последний — {when}",
            f"проверенные условия: глубина {e.get('depth', '?')} токенов. "
            f"исправь или убери; после успешного замера запись снимется сама")
