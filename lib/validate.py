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
from .i18n import t

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
                    t('роутер вырезает этот аргумент из пресета'),
                    t('убери из пресета; значение задаётся только CLI роутера'))
            continue
        if k in ROUTER_FORCED:
            rep.add("warn", name, key,
                    t('роутер всё равно подставит своё значение'),
                    t('в пресете смысла нет; лишнее только путает при чтении'))
            continue
        f = schema.resolve(flags, k)
        if f is None:
            rep.add("warn", name, key, t('такого флага нет в этой сборке'),
                    t('проверь `llama-server --help`; в форках флаги добавляют и переименовывают'))
            continue
        if f.canonical.lstrip("-") in HARD_CONTROL:
            rep.add("error", name, key,
                    t('роутер вырезает этот аргумент из пресета'),
                    t('убери из пресета; значение задаётся только CLI роутера'))
            continue
        if f.canonical.lstrip("-") in ROUTER_FORCED:
            rep.add("warn", name, key,
                    t('роутер всё равно подставит своё значение'),
                    t('в пресете смысла нет; лишнее только путает при чтении'))
            continue
        # значение против enum
        if f.enum and str(val).strip():
            v = str(val).strip().lower()
            if v not in [e.lower() for e in f.enum] and v not in TRUE | FALSE \
                    and v not in ("all", "auto"):
                rep.add("warn", name, key, t('значение {0!r} не из списка {1}', val, f.enum))
        if f.kind in ("int", "float") and str(val).strip():
            try:
                float(str(val).strip())
            except ValueError:
                rep.add("error", name, key, t('ожидалось число, а стоит {0!r}', val))

    # ── пути ──
    model = low.get("model") or low.get("m") or ""
    if not model:
        rep.add("error", name, "model", t('не указан путь к модели'),
                t('пресет без model не запустится; для HF укажи hf-repo'))
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
                rep.add("error", name, "model", t('файла нет или это не GGUF'),
                        t('проверь путь: {0}', model))
                ok, meta = False, None
            if model_cache is not None:
                model_cache[model] = (ok, meta)

    for key in ("mmproj", "model-draft"):
        p = low.get(key)
        if p and check_paths and not p.startswith(("http://", "https://")):
            if not gguf.is_gguf(p):
                rep.add("error", name, key, t('файла нет или это не GGUF'), t('путь: {0}', p))

    if meta is None:
        return None

    # ── семантика против метаданных ──
    ctx = budget._flag_int(pairs, "c", "ctx-size")
    if ctx is not None and ctx < 0:
        rep.add("error", name, "c", t('контекст отрицательный: {0}', ctx),
                t('опечатка в знаке; llama-server такое значение не примет'))
    elif ctx and meta.n_ctx_trained and ctx > meta.n_ctx_trained:
        rep.add("error", name, "c",
                t('контекст {0} больше обученного {1}', ctx, meta.n_ctx_trained),
                t('модель не выучена на таком окне; качество деградирует'))
    ncpu = budget._flag_int(pairs, "n-cpu-moe", "ncmoe")
    if ncpu is not None and ncpu < 0:
        rep.add("error", name, "n-cpu-moe", t('выгрузка отрицательная: {0}', ncpu),
                t('опечатка в знаке; столько слоёв выгрузить нельзя'))
    elif ncpu and meta.n_layer and ncpu > meta.n_layer:
        rep.add("error", name, "n-cpu-moe",
                t('{0} больше числа слоёв ({1})', ncpu, meta.n_layer))
    ngl = budget._flag_int(pairs, "n-gpu-layers", "ngl", "gpu-layers")
    if ngl is not None and meta.n_layer and 0 <= ngl < meta.n_layer:
        rep.add("info", name, "n-gpu-layers",
                t('{0} из {1} слоёв в VRAM — часть модели в RAM', ngl, meta.n_layer),
                t('медленнее генерация; осознанное решение для больших моделей'))
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
        # Ключа может не быть: дефолт у --mmproj-offload — enabled, то есть
        # mmproj в VRAM. Раньше ловушка молчала, если ключ не записан вовсе,
        # а это самый частый случай — тот самый OOM и не варнился.
        "applies": lambda p: str(p.get("mmproj", "")).strip() != ""
                              and str(p.get("mmproj-offload", "")).strip()
                              .lower() not in ("0", "false", "no", "off")
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
        # Порог «от 2048», а не ровно 2048: 4096 с ускорителем на 12 ГБ —
        # тот же OOM, только ещё глубже, и раньше он проходил без варна.
        "id": "ubatch-2048-oom",
        "applies": lambda p: int(budget._flag_int(p, "ubatch-size", "ub", default=0) or 0) >= 2048
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
                    t('{0} — но есть замер, прошедший проверку', trap['message']),
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
            t('этот пресет ронял сервер ({0}), {1} раз, последний — {2}', e.get('why'), howmany, when),
            t('проверенные условия: глубина {0} токенов. исправь или убери; после успешного замера запись снимется сама', e.get('depth', '?')))


# ── reasoning / DRY / сэмплинг: проверка актуальных флагов ──
# Источник правды — схема ЭТОЙ сборки (schema.resolve по --help бинаря),
# а не захардкоженный список: форки добавляют флаги неравномерно
# (проверено живьём: --reasoning-effort есть в upstream/xing4, но нет
# в faks/ik; --reasoning-budget-conclusion нет нигде).
REASONING_FORMAT_VALUES = ("auto", "none", "deepseek", "deepseek-legacy")
REASONING_EFFORT_VALUES = ("default", "minimal", "low", "medium",
                           "high", "xhigh", "max")
# Подстроки имени/пути модели, намекающие на reasoning-модель.
REASONING_NAME_HINTS = ("qwen3", "qwen", "deepseek", "gpt-oss", "gpt_oss",
                        "think", "reasoning", "r1", "reasoner")


def _fnum(val) -> float | None:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def check_reasoning_quality(name: str, pairs: dict[str, str], flags: dict,
                            rep: Report, meta=None) -> None:
    """Связность reasoning/DRY/сэмплинга против схемы ЭТОЙ сборки.

    Ловит то, что общая проверка «флаг существует» не ловит:
      * флаг из чужой сборки/документации (effort на faks, несуществующий
        budget-conclusion) — будет молча проигнорирован как неизвестный;
      * бюджет без сообщения (резкий обрыв мыслей) и сообщение без бюджета
        (инертный ключ);
      * DRY-ключи при выключенном мультипликаторе и нулевое окно штрафа;
      * невалидные значения reasoning-format / reasoning-effort;
      * reasoning-модель без антизацикливающей связки (info, не warn).

    Всё, что касается поддержки, сверяется через schema.resolve: если флаг
    переименуют в следующем релизе, предупреждение последует за схемой.
    """
    low = {k.lower(): (v if v is not None else "") for k, v in pairs.items()}

    def supported(key: str) -> bool:
        try:
            return schema.resolve(flags, key) is not None
        except Exception:
            return False

    def raw(*names: str) -> str:
        for n in names:
            if n in low:
                return str(low[n])
        return ""

    # ── несуществующие / неподдержанные флаги ──
    if "reasoning-budget-conclusion" in low:
        rep.add("warn", name, "reasoning-budget-conclusion",
                t('такого флага нет ни в одной из проверенных сборок (faks/upstream/ik/xing4)'),
                t('вероятно, выдуман или из чужой документации; убери — иначе ключ молча игнорируется. Проверь `llamastery schema --build <сборка> --grep reasoning`'))
    if "reasoning-effort" in low and not supported("reasoning-effort"):
        rep.add("warn", name, "reasoning-effort",
                t('эта сборка не знает --reasoning-effort (есть в upstream/xing4, нет в faks/ik)'),
                t('ключ будет проигнорирован; либо убери, либо переезжай на сборку с поддержкой, либо держи уровень через чат-шаблон'))

    # ── значения enum ──
    fmt = raw("reasoning-format").strip().lower()
    if fmt and fmt not in REASONING_FORMAT_VALUES:
        rep.add("warn", name, "reasoning-format",
                t('значение {0!r} не из списка {1}', raw('reasoning-format'), list(REASONING_FORMAT_VALUES)),
                t('для отделения мыслей от ответа нужен deepseek (мысли — в message.reasoning_content)'))
    eff = raw("reasoning-effort").strip().lower()
    if eff and eff not in REASONING_EFFORT_VALUES:
        rep.add("warn", name, "reasoning-effort",
                t('значение {0!r} не из списка {1}', raw('reasoning-effort'), list(REASONING_EFFORT_VALUES)),
                t('проверь `llama-server --help | grep reasoning-effort`'))

    # ── бюджет + сообщение ──
    b_raw = raw("reasoning-budget").strip()
    b_num = _fnum(b_raw) if b_raw else None
    has_msg = bool(raw("reasoning-budget-message").strip())
    if b_raw and b_num is not None and b_num > 0 and not has_msg:
        rep.add("warn", name, "reasoning-budget",
                t('бюджет {0} без reasoning-budget-message — мысли обрубаются резко, без красивого завершения', b_raw),
                t('добавь reasoning-budget-message = "Wrap up and give the final answer."'))
    if has_msg and (not b_raw or (b_num is not None and b_num < 0)):
        rep.add("warn", name, "reasoning-budget-message",
                t('сообщение о завершении инертно без reasoning-budget > 0'),
                t('либо задай бюджет (например 2048), либо убери сообщение'))
    if b_raw and b_num is None:
        rep.add("error", name, "reasoning-budget",
                t('ожидалось число, а стоит {0!r}', b_raw))

    # ── DRY ──
    m_raw = raw("dry-multiplier").strip()
    m_num = _fnum(m_raw) if m_raw else 0.0
    dry_extra = {k: low[k] for k in
                 ("dry-allowed-length", "dry-base", "dry-penalty-last-n",
                  "dry-sequence-breaker") if k in low}
    if m_num is not None and m_num > 0:
        w_raw = raw("dry-penalty-last-n").strip()
        w_num = _fnum(w_raw) if w_raw else None
        if w_num == 0:
            rep.add("warn", name, "dry-penalty-last-n",
                    t('окно штрафа 0 при включённом DRY — сэмплер не видит повторы и не работает'),
                    t('поставь окно под контекст (например 4096) или убери ключ'))
    elif dry_extra:
        rep.add("info", name, "dry",
                t('DRY-ключи заданы, но dry-multiplier выключен (0/нет) — они инертны'),
                t('для борьбы с зацикливанием: dry-multiplier = 0.5, dry-allowed-length = 6, dry-penalty-last-n = 4096'))

    # ── reasoning-модель без антизацикливающей связки ──
    model_id = " ".join([
        raw("model", "m"),
        getattr(meta, "name", "") or "",
        getattr(meta, "arch", "") or "",
    ]).lower()
    looks_reasoning = any(h in model_id for h in REASONING_NAME_HINTS)
    has_any_reasoning = any(k in low for k in
                            ("reasoning", "reasoning-format", "reasoning-budget",
                             "reasoning-effort", "reasoning-budget-message"))
    if (looks_reasoning or has_any_reasoning) and not has_any_reasoning:
        rep.add("info", name, "reasoning",
                t('похоже на reasoning-модель, но reasoning-флагов нет — на низких квантах (IQ4/IQ3/Q3) возможны зацикливания в мыслях'),
                t('проверь поддержку (`schema --grep reasoning`) и добавь связку: reasoning-format = deepseek, reasoning-budget = 2048 + reasoning-budget-message, DRY 0.5/6/4096, temp 0.6 / top-p 0.95 / top-k 20'))
