"""Приведение комментариев над пресетами к единому стандарту.

Принципы (docs/ru/comments.md):
  * проза переносится ДОСЛОВНО, переставляется только обёртка;
  * метрики собираются из замеров (measurements.json) и оценки (budget);
  * у каждого числа обязателен тег [замер] или [оценка];
  * нет данных — пишется «не измерено», ничего не выдумывается;
  * идемпотентно: повторный запуск не меняет файл.
"""

import re
from dataclasses import dataclass, field

from . import budget, crashes, gguf, measure
from .inifile import IniFile

RULE_ITEM = "; " + "-" * 75
RULE_GROUP = "; " + "=" * 75

# метки старого формата -> куда их
LABEL_MAP = {
    "СКОРОСТЬ": "metrics_src",
    "ПАМЯТЬ": "metrics_src",
    "VRAM": "metrics_src",
    "ЗАМЕРЫ": "provenance",
    "ПРОИСХОЖДЕНИЕ": "provenance",
    "ОСОБЕННОСТИ": "note",
    "КОГДА ИСПОЛЬЗОВАТЬ": "note",
    "НАЗНАЧЕНИЕ": "purpose",
    "ЗАЧЕМ": "purpose",
    "ЗАМЕР": "provenance",       # строка «Замер:» нового формата
    "НЮАНС": "note",             # строка «Нюанс:» нового формата
    "ЗАМЕТКА": "remark",         # строка «Заметка:» нового формата
}
# Строка метрик — это «tg 27.1 t/s [замер] …», а не любая строка, где tg
# встретился первым: иначе продолжение внутри «Замер:» вида «;        tg 27.1
# (пробы 27.1 / 27.4) …» считалось новой строкой метрик и терялось.
# Строка метрик — либо «tg 27.1 t/s [замер] …», либо «tg не измерено  vram …».
# Именно число или «не измерено»: строка вида «;        tg 27.1 (пробы 27.1 /
# 27.4) …» — это продолжение поля «Замер:», и её нельзя считать новой строкой
# метрик, иначе она теряется из блока.
METRICS_LINE_RE = re.compile(r"^;\s*tg\s+(?:\d|не\b)")
# Продолжение поля — строка с отступом за точкой с запятой. Именно отступ
# решает: LABEL_RE матчит «;        коде ускоритель вредит: …», потому что
# после отступа стоит слово с двоеточием, и такая строка получала собственную
# метку — абзац распадался на пункты вида «Нюанс: коде ускоритель вредит».
CONTINUATION_INDENT = re.compile(r"^;\s{2,}\S")
# отступ, который печатается перед продолжением поля
CONT_INDENT = " " * 9
LABEL_RE = re.compile(r"^;\s*([А-ЯЁA-Z][А-ЯЁA-Zа-яa-z ]{2,20}):\s*(.*)$")
RULE_RE = re.compile(r"^;\s*[-]{10,}\s*$")
RULE_GROUP_RE = re.compile(r"^;\s*={10,}\s*$")

TG_RE = re.compile(r"tg[^0-9]{0,12}(\d+[.,]?\d*)\s*t/s", re.I)
VRAM_RE = re.compile(r"(\d+[.,]?\d*)\s*(?:ГБ|GiB|GB|MiB|МБ)", re.I)
MINFREE_RE = re.compile(r"min free\s+(\d{3,5})\s*MiB", re.I)
NEEDLE_RE = re.compile(r"needle\s*(OK|ок|\d/\d)", re.I)
STAGE_RE = re.compile(r"(tiel[\w-]*|stage[\w-]*)\s*(\d{4}-\d{2}-\d{2})?", re.I)


@dataclass
class Block:
    title: str = ""
    purpose: list[str] = field(default_factory=list)
    note: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    metrics_src: list[str] = field(default_factory=list)   # старая проза СКОРОСТЬ/ПАМЯТЬ
    metrics_line: list[str] = field(default_factory=list)   # строка метрик нового формата
    remark: list[str] = field(default_factory=list)         # строка «Заметка:» нового формата
    unparsed: list[str] = field(default_factory=list)


def _looks_like_preset_block(run: list[str]) -> bool:
    """Отличает комментарий пресета от шапки файла/группы."""
    body = [x.lstrip("; ").strip() for x in run
            if x.strip() and not RULE_RE.match(x.strip())
            and not RULE_GROUP_RE.match(x.strip())]
    if not body:
        return False
    first = body[0]
    if "[" in first or "|" in first:
        return True
    # титул без скобок и без меток: скорее шапка
    if any(LABEL_RE.match(x) or METRICS_LINE_RE.match(x) for x in run):
        return True
    return len(body) <= 2


def parse_block(lines: list[str]) -> Block:
    """Разбирает комментарий над секцией в поля стандарта."""
    b = Block()
    cur: list[str] | None = None
    target: list[str] | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or RULE_RE.match(line):
            continue
        body = line[1:].strip() if line.lstrip().startswith(";") else line.strip()
        # пустая строка «;» внутри поля разделяет пункты. Без такого признака
        # соседние пункты одного поля не отличить от продолжения абзаца:
        # «; Нюанс: коде ускоритель вредит…» по форме неотличим от строки,
        # которую сам формат отдаёт в разбор как метку.
        if not body and line.lstrip().startswith(";") and cur and target is not None:
            target.append("\0")      # граница абзаца
            continue
        # Продолжение поля проверяем ПЕРВЫМ, до строки метрик: продолжение
        # внутри «Замер:» может начинаться с «tg 27.1 (пробы …)» — по форме
        # это строка метрик, и без порядка проверок она переезжала в шапку
        # пресета, а её значение терялось. Настоящая строка метрик идёт
        # без отступа (один пробел после «;»), продолжение — с отступом.
        if cur and target is not None and CONTINUATION_INDENT.match(line):
            if body:
                target.append(body)
            continue
        if METRICS_LINE_RE.match(line):
            # значения оттуда переиспользуются, сама строка пересобирается
            # заново. cur сбрасываем в None: следующая строка без отступа
            # обязана стать новым полем, а не продолжением метрик.
            b.metrics_line.append(body)
            cur, target = None, b.metrics_src
            continue
        m = LABEL_RE.match(line)
        if m:
            label, rest = m.group(1).strip().upper(), m.group(2).strip()
            nxt = LABEL_MAP.get(label)
            if nxt is None:
                target = b.unparsed
            elif nxt == "purpose":
                target = b.purpose
            else:
                target = getattr(b, nxt)
            # пустое поле после метки (например «Замер:» с текстом на
            # следующей строке) не должно отдавать текущему полю его старый
            # список: иначе содержимое предыдущего поля переезжает в новое.
            if rest:
                target.append(rest)
            cur = nxt
            continue
        if cur and target is not None:
            # продолжение поля: текст с отступом или пустая метка выше
            if body:
                target.append(body)
            continue
        if not b.title and not b.purpose:
            b.title = body
        else:
            b.purpose.append(body)
    return b


# ── метрики ──
def _is_table_row(text: str) -> bool:
    """Строка таблицы: несколько чисел в колонках.

    Такие строки внутри блока (сравнение сборок) нельзя читать как прозу: там
    рядом стоят tg и prefill, и число из второй колонки принималось за tg.
    """
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    return len(nums) >= 3


def _tg_from_prose(lines: list[str]) -> tuple[float, str] | None:
    """Берёт tg из старой прозы. Возвращает (значение, доказательство).

    Приоритет у значения на ПОЛНОЙ глубине: строка `deep` важнее `short`,
    иначе 4k-проба выдастся за характеристику пресета.
    """
    # строки таблиц внутри блока (сравнение сборок) выбрасываем: в них рядом
    # стоят tg и prefill, и вторая колонка принималась за tg — в шапку попадало
    # 396 t/s вместо 27.1
    lines = [ln for ln in lines if not _is_table_row(ln)]
    text = " ".join(lines)
    hits = [(m.start(), float(m.group(1).replace(",", ".")))
            for m in re.finditer(r"tg\s*(\d+(?:[.,]\d+)?)\s*t/s", text, re.I)]
    if not hits:
        m = re.search(r"~?(\d+(?:[.,]\d+)?)\s*t/s", text)
        if m:
            return float(m.group(1).replace(",", ".")), text[:110]
        return None
    marks = [m.start() for m in re.finditer(r"\bdeep\b", text, re.I)]
    after = [h for h in hits if marks and h[0] > marks[-1]]
    if after:
        return after[0][1], "deep, полный контекст"
    return hits[0][1], text[:110]


def _vram_from_prose(lines: list[str]) -> tuple[float, str] | None:
    # строки таблиц внутри блока выбрасываем: «ik 24.0 341 t/s 10419 1869
    # <- нет ngram, +1.3 ГБ» давали в шапку vram 1.3 GiB вместо 8.9
    lines = [ln for ln in lines if not _is_table_row(ln)]
    text = " ".join(lines)
    mf = MINFREE_RE.search(text)
    total = budget.gpu_total_mib()
    if mf and total:
        return (total - int(mf.group(1))) / 1024.0, text[:120]
    m = VRAM_RE.search(text)
    if m:
        v = float(m.group(1).replace(",", "."))
        return v, text[:120]
    return None


def build_metrics(name: str, pairs: dict, store: dict, cal: dict,
                  siblings: list[dict], prose: list[str] | None = None,
                  b_metrics_line: list[str] | None = None,
                  b_provenance: list[str] | None = None,
                  ) -> tuple[list[str], list[str]]:
    """Возвращает (строка метрик, сноска об источнике).

    Приоритет для tg: замер тюнера -> число из старой прозы -> «не измерено».
    Для vram: замер тюнера (свободная память) -> проза -> оценка budget.
    """
    prose = prose or []
    b_provenance = b_provenance or []
    meas = measure.lookup(pairs, store)
    # если строка метрик уже в стандартном формате — берём её значения,
    # иначе повторный запуск форматтера «съел» бы записанные замеры
    have: dict[str, str] = {}
    for pl in b_metrics_line:
        mt = re.search(r"tg\s+(\d+(?:[.,]\d+)?)\s*t/s\s+\[(замер|оценка)\]", pl)
        if mt:
            have["tg"] = f"tg {float(mt.group(1)):.1f} t/s [{mt.group(2)}]"
        mv = re.search(r"vram\s+(\d+(?:[.,]\d+)?)\s*GiB\s+\[(замер|оценка)\]", pl)
        if mv:
            have["vram"] = f"vram {float(mv.group(1)):.1f} GiB [{mv.group(2)}]"
        # n-cpu-moe — такой же факт, как ctx и ub: метрика, а не описание.
        # Без него запись «moe 24» терялась, и главный рычаг скорости в шапке
        # пресета не был виден.
        mm = re.search(r"\bmoe\s+(\d+)", pl)
        if mm:
            have["moe"] = mm.group(1)
        mc = re.search(r"\bctx\s+(\d+)", pl)
        if mc:
            have["ctx"] = mc.group(1)
        mu = re.search(r"\bub\s+(\d+)", pl)
        if mu:
            have["ub"] = mu.group(1)
    # (значение, источник). Источник пишется один раз на строку: два
    # «[замер]» подряд — «tg 27.1 t/s [замер]  vram 8.9 GiB [замер]» — читались
    # как два независимых утверждения, хотя относятся к одной строке.
    bits: list[tuple[str, str | None]] = []
    notes: list[str] = []

    meas_tps = meas.get("deep_tps") or meas.get("short_tps") if meas else None
    if "tg" in have and not meas_tps:
        bits.append((have["tg"], _marker(have["tg"])))
    elif meas and (meas.get("deep_tps") or meas.get("short_tps")):
        v = meas.get("deep_tps") or meas.get("short_tps")
        depth = "на полном контексте" if meas.get("deep_tps") else "на 4k"
        bits.append((f"tg {float(v):.1f} t/s", "замер"))
        notes.append(depth)
        if meas.get("gen_tps_repetitive"):
            notes.append(f"на тексте с повторами "
                         f"{float(meas['gen_tps_repetitive']):.1f} t/s — "
                         f"это и есть смысл ngram, на уникальном тексте его нет")
        if meas.get("needle_ok") is not None:
            notes.append("needle " + ("OK" if meas["needle_ok"] else "провален"))
    else:
        got = _tg_from_prose(prose)
        if got:
            bits.append((f"tg {got[0]:.1f} t/s", "замер"))
            if not b_provenance:
                notes.append("из комментария пресета, стадия не указана")
        else:
            bits.append(("tg не измерено", None))

    # vram: приоритет у фактического замера из тюнера
    est = None
    meta = None
    mp = pairs.get("model") or pairs.get("m") or ""
    if mp and gguf.is_gguf(mp):
        try:
            meta = gguf.probe(mp)
        except gguf.GGUFError:
            meta = None
    mm = None
    mpp = pairs.get("mmproj")
    if mpp and gguf.is_gguf(mpp):
        try:
            mm = gguf.probe(mpp)
        except gguf.GGUFError:
            mm = None
    if meta is not None:
        est = budget.estimate(pairs, meta, mmproj_meta=mm)

    vram_mib, vram_done = None, False
    meas_free = meas.get("min_free_mib") if meas else None
    if "vram" in have and meas_free is None:
        bits.append((have["vram"], _marker(have["vram"])))
        vram_done = True
    elif meas and meas.get("min_free_mib") is not None:
        total = budget.gpu_total_mib() or cal.get("vram_total_mib")
        if total:
            vram_mib = int(total) - int(meas["min_free_mib"])
            bits.append((f"vram {vram_mib / 1024:.1f} GiB", "замер"))
            vram_done = True
    if not vram_done:
        got = _vram_from_prose(prose)
        if got:
            bits.append((f"vram {got[0]:.1f} GiB", "замер"))
            if not b_provenance:
                notes.append("из комментария пресета, стадия не указана")
            vram_done = True
    if not vram_done:
        if est is not None:
            bits.append((f"vram {est.total_gb:.1f} GiB", "оценка"))
        else:
            bits.append(("vram не оценена", None))

    ctx = pairs.get("c") or pairs.get("ctx-size")
    if ctx and ("ctx" in have or any(s.get("c") != ctx for s in siblings)):
        bits.append((f"ctx {ctx}", None))
    ub = pairs.get("ubatch-size") or pairs.get("ub")
    if ub and ("ub" in have or any(s.get("ubatch-size") != ub for s in siblings)):
        bits.append((f"ub {ub}", None))
    # n-cpu-moe — главный рычаг скорости в этих пресетах, и без него шапка
    # не отвечала на вопрос «а сколько слоёв на CPU». Печатаем всегда: в отличие
    # от ctx/ub, moe почти всегда различается между секциями набора.
    moe = budget._flag_int(pairs, "n-cpu-moe")
    if moe and ("moe" in have or any(
            budget._flag_int(s, "n-cpu-moe") != moe for s in siblings)):
        bits.append((f"moe {moe}", None))

    seen, uniq = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return [_metrics_line(bits)], uniq


def _marker(text: str) -> str | None:
    """Источник значения, если он уже записан в скобках."""
    m = re.search(r"\[(замер|оценка)\]\s*$", text.strip())
    return m.group(1) if m else None


def _strip_marker(text: str) -> str:
    return re.sub(r"\s*\[(замер|оценка)\]\s*$", "", text.strip())


def _collapse_markers(line: str) -> str:
    """Схлопывает повторяющуюся пометку источника в строке комментария.

    «; tg 48.0 t/s [замер]  vram 10.6 GiB [замер]» -> одна метка в конце.
    Применяется ко всему файлу, включая блоки вне секций, которые форматтер
    не переписывает.
    """
    if line.count("[замер]") + line.count("[оценка]") < 2:
        return line
    marks = line.count("[замер]"), line.count("[оценка]")
    only = "[замер]" if marks[0] else "[оценка]"
    if marks[0] and marks[1]:
        return line          # разные источники: подписи несут смысл
    stripped = line.replace(f" {only}", "").replace(only, "")
    body = stripped.rstrip()
    if not body.endswith(";"):
        body += " "
    return f"{body}{only}"


def _metrics_line(bits: list[tuple[str, str | None]]) -> str:
    """Строка метрик с одним указанием источника.

    Пометка ставится один раз — в конце строки, и только если она относится ко
    всем помеченным значениям сразу: «tg 27.1 t/s  vram 8.9 GiB  [замер]».
    Иначе смысл теряется: при разнородных источниках подпись должна стоять
    рядом со своим значением («tg [замер]  vram [оценка]»), иначе читается
    так, будто замерены оба.

    Значения без источника (ctx, ub, moe) подпись не распространяют: они
    взяты из самого пресета, а не из измерения.
    """
    if not bits:
        return ""
    texts = [_strip_marker(t) for t, _ in bits]
    marked = [m for _, m in bits if m]
    if marked and len(set(marked)) == 1:
        return "  ".join(texts) + f"  [{marked[0]}]"
    return "  ".join(
        f"{t} [{m}]" if m else t for t, (_, m) in zip(texts, bits))


def _with_labels(rows: list[str], label: str) -> list[str]:
    """Метка на первой строке каждого абзаца, отступ на продолжениях.

    Абзацы разделены пустой строкой «;», которая при разборе становится
    маркером \0. Поэтому соседние пункты одного поля не сливаются в один
    абзац, а «Замер:» внутри блока не превращает поле в набор пунктов.
    """
    out: list[str] = []
    first = True
    for t in rows:
        if t == "\0":
            # пустая строка «;» — граница пункта. На печати она терялась, и
            # соседние пункты снова слипались в один абзац. Печатаем ровно
            # «;»: render добавляет «; » сам, иначе выходило «; ;».
            out.append("")
            first = True
            continue
        if not t.strip():
            continue
        if first:
            out.append(f"{label}: {t}")
            first = False
        else:
            out.append(CONT_INDENT + t)
    return out


def provenance_lines(b: Block, notes: list[str]) -> list[str]:
    # Если источник уже назван в блоке, замерная сноска его не дублирует.
    # Раньше при пустом блоке строка «Замер: не измерено» печаталась даже
    # тогда, когда tg только что подставлен из measurements.json: число в
    # шапке было замером, а сноска под ним утверждала обратное.
    src = list(b.provenance)
    if not src:
        src = [t for t in notes if t]
    if not src:
        return ["Замер: не измерено — прогнать: llamastery tune models.ini "
                "<секция> --build faks --extra deep"]
    # метку ставим только на первую строку поля: продолжения уже пришли с
    # своим отступом, и повторная метка превращала абзац в набор пунктов
    return _with_labels(src, "Замер")


def render(name: str, b: Block, metrics: str, prov: list[str],
           note: list[str]) -> list[str]:
    out = [RULE_ITEM]
    title = b.title or f"[{name}]"
    out.append(f"; {title}")
    if metrics:
        out.append(f"; {metrics}")
    for k, t in enumerate(b.purpose):
        out.append(f"; {'Назначение: ' + t if k == 0 else '           ' + t}")
    for p in prov:
        out.append(f"; {p}")
    for n in note:
        out.append(f"; {n}")
    out.append(RULE_ITEM)
    return out


def annotate(ini: IniFile, apply_changes: bool = False,
             store: dict | None = None) -> dict:
    """Переписывает комментарии всех пресетов. Возвращает отчёт."""
    store = store if store is not None else measure.load_store()
    cal = budget.load_calibration()
    report = {"changed": [], "unparsed": {}, "metrics": {}}

    lines = ini.text.splitlines()
    # индексы строк с заголовками секций и их комментариями
    blocks: list[tuple[int, int, str]] = []   # (start, header, name)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("[") and s.endswith("]") and s != "[*]":
            # блок пресета — непрерывный ряд `;` строк над заголовком;
            # ряд обрывается пустой строкой, ключом или другой секцией
            j = i
            while j > 0 and lines[j - 1].lstrip().startswith(";"):
                j -= 1
            run = lines[j:i]
            if run and RULE_GROUP_RE.match(run[0].strip()) and \
                    sum(1 for x in run if RULE_GROUP_RE.match(x.strip())) == 1:
                j = i          # это заголовок группы, не блок пресета
            elif run and not _looks_like_preset_block(run):
                # абзац без титула вида «[Модель] ... | режим» и без полей —
                # это шапка файла или группы, а не комментарий пресета
                j = i
            # пустую строку-разделитель в блок не берём: её нормализует
            # писатель, иначе блок никогда не совпадёт сам с собой
            blocks.append((j, i, s[1:-1]))
            i += 1
        else:
            i += 1

    # соседи по ctx — чтобы не печатать одинаковые ctx/ub
    pairs_by_name = {n: (ini.section(n).pairs() if ini.section(n) else {})
                     for _, _, n in blocks}

    out: list[str] = []
    pos = 0
    for start, header, name in blocks:
        # разделитель печатается сам, в render(): если оставить старый, он
        # задваивался, и при следующем запуске второй попадал в разбор как
        # содержимое блока
        s0 = start
        while s0 < header and not lines[s0].strip():
            s0 += 1
        if s0 < header and RULE_RE.match(lines[s0].strip()):
            start = s0
        out.extend(lines[pos:start])
        while out and not out[-1].strip():
            out.pop()
        out.append("")
        old = lines[start:header]
        clean = [x for x in old
                 if not RULE_RE.match(x.strip())
                 and not RULE_GROUP_RE.match(x.strip())]
        b = parse_block(clean)
        pairs = pairs_by_name[name]
        family = name.rsplit("-", 1)[0]
        siblings = [pairs_by_name[n] for _, _, n in blocks
                    if n != name and n.rsplit("-", 1)[0] == family]
        metrics, notes = build_metrics(name, pairs, store, cal, siblings,
                                       prose=b.metrics_src + b.provenance,
                                       b_metrics_line=b.metrics_line,
                                       b_provenance=b.provenance)
        prov = provenance_lines(b, notes)
        # Нюанс — ловушки из ОСОБЕННОСТИ/КОГДА ИСПОЛЬЗОВАТЬ.
        # Заметка — старая проза СКОРОСТЬ/ПАМЯТЬ дословно: в ней есть
        # тонкости («короткий», «на полной глубине», «prefill»), которых
        # нет в метриках, и она остаётся как есть.
        # Метку ставим только на первую строку поля: продолжения пришли
        # со своим отступом, и лишняя метка разрывала абзац на пункты.
        # Раньше префикс вешали на каждую строку — при повторном разборе
        # границы пунктов терялись навсегда.
        note = _with_labels(b.note, "Нюанс")
        remark = b.metrics_src or b.remark
        if remark:
            note += _with_labels(remark, "Заметка")
        # Проверенное падение — самая дорогая строка в файле: она экономит
        # полчаса замеров тому, кто загрузит пресет вслепую.
        crash = crashes.known(name)
        if crash:
            where = f" на глубине {crash['depth']}" if crash.get("depth") else ""
            note.append(f"Нюанс: ПАДАЕТ{where} — {crash.get('why', '')} "
                        f"(журнал падений, {crash.get('last', '?')})")
        new = render(name, b, metrics[0], prov, note)
        if b.unparsed:
            report["unparsed"][name] = b.unparsed
        report["metrics"][name] = metrics[0]
        if new != old:
            report["changed"].append(name)
        out.extend(new)
        out.append(lines[header])
        pos = header + 1
    out.extend(lines[pos:])

    # Осиротевшие блоки — те, что стоят перед разделителем группы, а не перед
    # секцией. Их форматтер не разбирает, поэтому повторяющаяся метка в них
    # оставалась: блок формально не пресет, а правило про источник действует
    # на весь файл.
    out = [_collapse_markers(ln) if ln.lstrip().startswith(";") else ln
           for ln in out]

    report["text"] = "\n".join(out) + "\n"
    if apply_changes:
        from . import presets as _p
        _p.backup(ini.path)
        ini.path.write_text(report["text"], encoding="utf-8")
    return report
