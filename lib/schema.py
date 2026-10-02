"""Схема флагов llama-server, разобранная из `llama-server --help`.

Зачем парсить help, а не держать свой список: список флагов меняется с
каждым релизом llama.cpp, а форки добавляют свои (`load-mode`,
`image-min-tokens`, `ctx-checkpoints` в Faks). Любая захардкоженная
таблица устаревает; help конкретной сборки — нет.

Формат строки в help:
    -c,    --ctx-size N                     size of the prompt context
                                            (default: 0, 0 = loaded from model)
                                            (env: LLAMA_ARG_CTX_SIZE)
    -sm,   --split-mode {none,layer,row,tensor}
    -ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers ...
"""

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths

# Строка опции: начинается с `-`, дальше формы через запятую, потом плейсхолдер.
_FORM_RE = re.compile(r"^-{1,2}[A-Za-z][\w.\-]*$")
_HINT_RE = re.compile(
    r"^(?:"
    r"\{[^}]*\}|\[[^\]]*\]|\([^)]*\)|<[^>]*>|[A-Z][A-Z0-9_]*|[A-Z0-9]+|lo-hi|[-0-9.]+"
    r"|[A-Za-z0-9_.\-]+,[A-Za-z0-9_.\-]+(?:,[A-Za-z0-9_.\-]*)*"
    r")$"
)
_ENV_RE = re.compile(r"\(env:\s*([A-Z0-9_]+)\s*\)")
_DEFAULT_RE = re.compile(r"default:\s*([^)\]]+?)(?:\)|$)")
_SET_RE = re.compile(r"\[([^]]*)\]|\{([^}]*)\}|\(([^)]*)\)")


@dataclass
class Flag:
    canonical: str            # длинная форма с двумя дефисами, если есть
    short: str | None         # короткая форма, если есть
    aliases: list[str] = field(default_factory=list)
    env: str | None = None
    value_hint: str | None = None   # N, FNAME, {a,b,c}, [on|off], lo-hi
    kind: str = "flag"        # flag | int | float | string | enum | path
    enum: list[str] = field(default_factory=list)
    default: str | None = None
    description: str = ""

    def ini_keys(self) -> list[str]:
        """Ключи, которыми этот флаг может быть записан в INI.

        llama.cpp принимает три эквивалентные формы (PR#17859):
        длинную без дефисов, короткую и имя переменной окружения.
        """
        keys = [self.canonical.lstrip("-")]
        if self.short:
            keys.append(self.short.lstrip("-"))
        keys.extend(a.lstrip("-") for a in self.aliases)
        if self.env:
            keys.append(self.env)
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out


def _tokens(line: str) -> list[str]:
    """Делит строку на токены.

    Разделители — запятые и пробелы, но: запятая внутри значения не
    разрывает токен (только если сразу за ней не идёт новый флаг), и
    запятые внутри {a,b} / [a|b] / <a|b> не трогаются вообще. Иначе
    `--spec-type none,draft-mtp,ngram-mod` распалось бы на четыре куска.
    """
    out, buf, depth = [], [], 0
    i, n = 0, len(line.strip())
    s = line.strip()
    while i < n:
        ch = s[i]
        if ch in "{[<":
            depth += 1
        elif ch in "}]>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            j = i + 1
            while j < n and s[j] == " ":
                j += 1
            if j >= n or s[j] != "-":      # запятая внутри значения
                buf.append(ch)
                i += 1
                continue
        if ch in ", \t" and depth == 0:
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _split_line(line: str) -> tuple[list[str], str | None, str]:
    """Делит строку help на (формы флагов, плейсхолдер значения, описание).

    Формы идут подряд в начале, разделённые запятыми; дальше может стоять
    плейсхолдер, дальше — описание. Иногда плейсхолдер вообще отсутствует
    (`-kvu, --kv-unified, -no-kvu, --no-kv-unified`).
    """
    tokens = _tokens(line)
    forms, idx = [], 0
    while idx < len(tokens) and _FORM_RE.match(tokens[idx]):
        forms.append(tokens[idx])
        idx += 1
    if not forms:
        return [], None, ""
    value_hint, desc = None, ""
    if idx < len(tokens):
        tok = tokens[idx]
        head = " ".join(forms)
        if _HINT_RE.match(tok):
            value_hint = tok
            m = re.search(re.escape(tok) + r"\s{2,}(.*)$", line)
            desc = m.group(1).strip() if m else ""
        else:
            m = re.search(re.escape(tok) + r"\s+(.*)$", line)
            desc = m.group(1).strip() if m else ""
    return forms, value_hint, desc


def _classify(value_hint: str | None, desc: str) -> tuple[str, list[str]]:
    """Возвращает (kind, enum_values) по плейсхолдеру значения."""
    if not value_hint:
        return "flag", []
    m = _SET_RE.search(value_hint)
    if m:
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        vals = [v.strip() for v in re.split(r"[|,]", raw) if v.strip()]
        if vals:
            return "enum", vals
    if "," in value_hint:            # голый список: none,draft-mtp,ngram-mod
        vals = [v.strip() for v in value_hint.split(",") if v.strip()]
        if len(vals) > 1:
            return "enum", vals
    if value_hint.strip("<>") == "N":
        return "int", []
    low = value_hint.lower()
    if low in ("fname", "path", "file", "dir", "filename"):
        return "path", []
    if re.fullmatch(r"[0-9]*\.?[0-9]+", value_hint):
        return "float", []
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", value_hint):
        return "string", []
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", value_hint):
        return "string", []
    return "flag", []


def parse_help(text: str) -> dict[str, Flag]:
    """Разбирает вывод --help в {canonical: Flag}."""
    lines = text.splitlines()
    flags: dict[str, Flag] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        # свежие llama.cpp печатают опции от нуля, ik_llama — с отступом
        if not line.lstrip().startswith("-"):
            i += 1
            continue
        forms, value_hint, inline = _split_line(line.lstrip())
        if not forms:
            i += 1
            continue

        # описание: остаток строки + все последующие отступные строки
        desc_parts = [inline] if inline else []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if nxt.lstrip().startswith("-") and _split_line(nxt.lstrip())[0]:
                break              # это следующая опция, а не продолжение
            if nxt.strip():
                desc_parts.append(nxt.strip())
            else:
                desc_parts.append("")
            j += 1
        desc = " ".join(p for p in desc_parts if p).strip()

        shorts = [f for f in forms if not f.startswith("--")]
        longs = [f for f in forms if f.startswith("--")]
        if longs:
            canonical, short, alias = longs[0], (shorts[0] if shorts else None), longs[1:]
        else:
            canonical, short, alias = shorts[0], shorts[0], []

        env = None
        em = _ENV_RE.search(desc)
        if em:
            env = em.group(1)
        default = None
        dm = _DEFAULT_RE.search(desc)
        if dm:
            default = dm.group(1).strip().strip("'\"")

        kind, enum = _classify(value_hint, desc)
        f = Flag(canonical=canonical, short=short, aliases=alias, env=env,
                 value_hint=value_hint, kind=kind, enum=enum,
                 default=default, description=desc)
        flags[canonical] = f
        for a in [short] + alias:
            if a:
                flags.setdefault(a, f)
        i = j
    return flags


def run_help(binary: str | Path, timeout: int = 60) -> str:
    p = subprocess.run([str(binary), "--help"], capture_output=True,
                       text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def load(binary: str | Path | None, use_cache: bool = True,
         refresh: bool = False) -> tuple[dict[str, Flag], dict]:
    """Возвращает (флаги, метаданные источника). Кэш — по mtime бинаря."""
    meta: dict = {}
    if binary is None:
        cached = paths.schema_cache()
        if cached.exists() and not refresh:
            blob = json.loads(cached.read_text())
            return {k: Flag(**v) for k, v in blob["flags"].items()}, blob["meta"]
        return {}, {"error": "не указан бинарь и нет кэша схемы"}

    binary = Path(binary).expanduser()
    if not binary.exists():
        return {}, {"error": f"нет такого бинаря: {binary}"}
    st = binary.stat()
    mtime = st.st_mtime
    meta = {"binary": str(binary), "mtime": mtime,
            "size": st.st_size}

    cached = paths.schema_cache(binary)
    if use_cache and cached.exists() and not refresh:
        try:
            blob = json.loads(cached.read_text())
            if blob.get("meta", {}).get("mtime") == mtime:
                return ({k: Flag(**v) for k, v in blob["flags"].items()},
                        blob["meta"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # битый кэш — перечитаем

    try:
        text = run_help(binary)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {}, {"error": f"не удалось запустить {binary} --help: {exc}"}

    flags = parse_help(text)
    meta["count"] = len({id(f) for f in flags.values()})
    meta["parsed_at"] = time.time()
    cached.parent.mkdir(parents=True, exist_ok=True)
    unique = {f.canonical: asdict(f) for f in
              {id(v): v for v in flags.values()}.values()}
    cached.write_text(json.dumps({"meta": meta, "flags": unique},
                                 ensure_ascii=False, indent=1))
    return flags, meta


def resolve(flags: dict[str, Flag], key: str) -> Flag | None:
    """Ищет флаг по любой из форм записи в INI (длинная/короткая/env).

    Регистр значим: в llama.cpp `-c` (--ctx-size) и `-C` (--cpu-mask) —
    разные флаги, поэтому сначала ищем точно, и только потом без учёта
    регистра (на случай, если пресет написан с заглавной).
    """
    if key in flags:
        return flags[key]
    k = key.lstrip("-")
    if k in flags:
        return flags[k]

    # один проход по уникальным флагам; ini_keys считаем один раз на флаг,
    # а не дважды (иначе O(ключи × флаги × формы) на каждый пресет)
    uniq = list({id(v): v for v in flags.values()}.values())
    forms = [(f, [x.lstrip("-") for x in f.ini_keys()]) for f in uniq]
    for f, stripped in forms:          # точно, с дефисами и без
        if f.canonical.lstrip("-") == k:
            return f
    for f, stripped in forms:          # любая форма записи, регистр значим
        if k in stripped:
            return f
    low = k.lower()
    for f, stripped in forms:          # последний рубеж — без регистра
        for x in stripped:
            if x.lower() == low:
                return f
    return None


# ── флаги, которыми роутер управляет сам: пресет не может их переопределить ──
CONTROL_ARGS = {
    "port", "host", "alias", "model", "m", "mmproj", "hf-repo", "hf-repo",
    "models-dir", "models-max", "models-preset", "api-key",
}
