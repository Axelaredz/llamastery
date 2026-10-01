"""INI с сохранением комментариев и порядка.

Зачем: models.ini у пользователя — это 27 КБ, где половина объёма это
комментарии с замерами. Стандартный configparser их теряет, а rewrite
уничтожил бы эти знания. Поэтому здесь свой парсер: файл режется на
блоки (преамбула + секции), каждый хранит сырой текст, и запись
собирает файл обратно, трогая только нужные строки.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")


@dataclass
class Entry:
    """Одна строка key = value внутри секции."""
    raw: str                 # исходная строка целиком
    key: str | None          # None для комментариев/пустых строк
    value: str | None
    line_no: int             # 1-based, для сообщений об ошибках
    is_global: bool = False  # строка до первой секции


@dataclass
class Section:
    name: str
    header_raw: str
    entries: list[Entry] = field(default_factory=list)
    line_no: int = 0
    dirty: set[int] = field(default_factory=set)   # индексы изменённых entries

    def pairs(self) -> dict[str, str]:
        """Ключи секции. Дубликаты: последний побеждает (как в llama.cpp).

        Значение перечитывается из raw, а не берётся из поля value: иначе
        после set() текст файла и содержимое секции разойдутся.
        """
        out: dict[str, str] = {}
        for e in self.entries:
            if e.key is None:
                continue
            val = e.value
            eq = e.raw.find("=")
            if eq >= 0:
                val = re.split(r"\s+[;#]", e.raw[eq + 1:], maxsplit=1)[0].strip()
            out[e.key] = val or ""
        return out

    def set(self, key: str, value: str) -> bool:
        """Меняет значение существующего ключа. False — ключа нет."""
        key_l = key.lower()
        for e in self.entries:
            if e.key and e.key.lower() == key_l:
                if (e.value or "") != value:
                    e.raw = f"{_quote(key, e.raw)} = {value}"
                    self.dirty.add(self.entries.index(e))
                return True
        return False

    def rename(self, old: str, new: str) -> bool:
        for e in self.entries:
            if e.key and e.key.lower() == old.lower():
                e.raw = e.raw.replace(e.key, new, 1)
                self.dirty.add(self.entries.index(e))
                return True
        return False

    def add(self, key: str, value: str, comment: str | None = None) -> None:
        block = ""
        if comment:
            block = "".join(f"; {ln}\n" for ln in comment.splitlines())
        self.entries.append(Entry(raw=f"{block}{key} = {value}", key=key,
                                  value=value, line_no=0, is_global=False))

    def get(self, key: str, default=None):
        return self.pairs().get(key.lower(), default)

    def drop(self, key: str) -> bool:
        keep, dropped = [], False
        for e in self.entries:
            if e.key and e.key.lower() == key.lower():
                dropped = True
                continue
            keep.append(e)
        self.entries = keep
        return dropped


def _quote(key: str, like: str) -> str:
    """Сохраняет исходное написание ключа (регистр) при перезаписи."""
    m = re.match(r"^(\s*)([A-Za-z0-9_.\-]+)", like)
    return m.group(2) if m else key


class IniFile:
    def __init__(self, path: str | Path, text: str):
        self.path = Path(path)
        self.text = text
        self.preamble: list[Entry] = []
        self.sections: dict[str, Section] = {}
        self._parse()

    # ── чтение ──
    @classmethod
    def load(cls, path: str | Path) -> "IniFile":
        p = Path(path).expanduser()
        return cls(p, p.read_text(encoding="utf-8", errors="replace"))

    def _parse(self) -> None:
        cur: Section | None = None
        for i, line in enumerate(self.text.splitlines(), 1):
            m = _SECTION_RE.match(line)
            if m:
                name = m.group("name").strip()
                cur = Section(name=name, header_raw=line, line_no=i)
                self.sections[name] = cur
                continue
            stripped = line.strip()
            if not stripped or stripped[0] in ";#":
                entry = Entry(raw=line, key=None, value=None, line_no=i,
                              is_global=cur is None)
            else:
                eq = line.find("=")
                if eq < 0:
                    entry = Entry(raw=line, key=None, value=None, line_no=i,
                                  is_global=cur is None)
                else:
                    key = line[:eq].strip()
                    val = line[eq + 1:]
                    # `;` после значения — инлайн-комментарий, не часть значения
                    val = re.split(r"\s+[;#]", val, maxsplit=1)[0].strip()
                    entry = Entry(raw=line, key=key, value=val, line_no=i,
                                  is_global=cur is None)
            if cur is None:
                self.preamble.append(entry)
            else:
                cur.entries.append(entry)

    # ── доступ ──
    def globals(self) -> dict[str, str]:
        """Значения по умолчанию: строки до первой секции + секция [*]."""
        out: dict[str, str] = {}
        for e in self.preamble:
            if e.key is not None:
                out[e.key] = e.value or ""
        star = self.sections.get("*")
        if star is not None:
            out.update(star.pairs())
        return out

    def section(self, name: str) -> Section | None:
        if name in self.sections:
            return self.sections[name]
        low = name.lower()
        for k, v in self.sections.items():
            if k.lower() == low:
                return v
        return None

    def has(self, name: str) -> bool:
        return self.section(name) is not None

    def names(self) -> list[str]:
        return list(self.sections.keys())

    # ── запись ──
    def dumps(self) -> str:
        out: list[str] = []
        for e in self.preamble:
            out.append(e.raw)
        for name, sec in self.sections.items():
            if out and out[-1].strip():
                out.append("")
            out.append(sec.header_raw)
            out.extend(e.raw for e in sec.entries)
        text = "\n".join(out)
        return text if text.endswith("\n") else text + "\n"

    def save(self, dest: str | Path | None = None) -> Path:
        p = Path(dest).expanduser() if dest else self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.dumps(), encoding="utf-8")
        return p


def read_sections(path: str | Path) -> dict[str, dict[str, str]]:
    """Быстрое чтение без сохранения комментариев (для чужих/чужих форматов)."""
    ini = IniFile.load(path)
    return {name: sec.pairs() for name, sec in ini.sections.items()}


def read_globals(path: str | Path) -> dict[str, str]:
    return IniFile.load(path).globals()
