"""Операции над пресетами: список, показ, импорт, экспорт, слияние.

Поддерживаемые источники импорта:
  * локальный путь к .ini
  * git-репозиторий + путь внутри (`repo#ref:path/to/presets.ini`)
  * HTTP(S)-URL (raw-файл с хоста, доступного без токена)

Слияние никогда не пишет в целевой файл молча: по умолчанию --dry-run,
а перед реальной записью делается резервная копия.
"""

import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .inifile import IniFile, Section

REMOTE_RE = re.compile(r"^(?P<repo>[^#]+)#(?P<ref>[^:]*):(?P<inner>.+)$")
# ключи, которые имеет смысл переносить между пресетами
CARRY_KEYS_HINT = ("model", "mmproj", "model-draft", "hf-repo", "hf-file")


@dataclass
class ImportResult:
    added: list[str]
    updated: list[str]
    skipped: list[str]
    conflicts: list[tuple[str, str, str]]   # (секция, ключ, было -> стало)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated)


def parse_source(src: str) -> tuple[str, dict]:
    """Разбирает источник. Возвращает (kind, payload)."""
    if re.match(r"^https?://", src):
        return "url", {"url": src}
    m = REMOTE_RE.match(src)
    if m:
        return "git", {"repo": m.group("repo"), "ref": m.group("ref") or "HEAD",
                       "inner": m.group("inner")}
    if src.startswith("git@") or src.endswith(".git"):
        return "git", {"repo": src, "ref": "HEAD", "inner": ""}
    return "file", {"path": src}


def fetch_source(src: str, timeout: int = 60) -> tuple[str, str]:
    """Возвращает (текст, описание источника)."""
    kind, payload = parse_source(src)
    if kind == "file":
        p = Path(payload["path"]).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"нет файла: {p}")
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    if kind == "url":
        req = urllib.request.Request(payload["url"],
                                     headers={"User-Agent": "llamastery/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace"), payload["url"]
    # git
    with tempfile.TemporaryDirectory() as td:
        cmd = ["git", "clone", "--depth", "1", "--quiet"]
        if payload["ref"] not in ("", "HEAD"):
            cmd += ["--branch", payload["ref"]]
        cmd += [payload["repo"], td]
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        root = Path(td) / Path(payload["repo"]).name
        if payload["inner"]:
            p = root / payload["inner"]
            if not p.exists():
                matches = list(root.glob(f"**/{payload['inner']}"))
                if not matches:
                    raise FileNotFoundError(
                        f"в репозитории нет {payload['inner']}")
                p = matches[0]
        else:
            cands = [p for p in root.glob("*.ini")]
            if not cands:
                cands = list(root.glob("**/*.ini"))
            if not cands:
                raise FileNotFoundError("в репозитории нет ни одного .ini")
            p = cands[0]
        return (p.read_text(encoding="utf-8", errors="replace"),
                f"{payload['repo']}#{payload['ref']}:{p.relative_to(root)}")


def backup(path: str | Path) -> Path:
    """Резервная копия с меткой времени. Возвращает путь к копии."""
    src = Path(path).expanduser()
    dst = src.with_suffix(src.suffix + f".bak-{int(__import__('time').time())}")
    shutil.copy2(src, dst)
    return dst


def merge_into(target: IniFile, source: IniFile, only: list[str] | None = None,
               rename: dict[str, str] | None = None,
               on_conflict: str = "skip") -> ImportResult:
    """Вливает секции источника в цель.

    on_conflict: skip (оставить как есть) | overwrite | new (создать копию
    с суффиксом -imported).
    """
    res = ImportResult([], [], [], [])
    rename = rename or {}
    wanted = [s.lower() for s in only] if only else None

    for name, ssec in source.sections.items():
        if name == "*":
            continue
        if wanted and name.lower() not in wanted:
            res.skipped.append(name)
            continue
        pairs = ssec.pairs()
        if not any(k.lower() in CARRY_KEYS_HINT for k in pairs):
            # секция без указания модели — скорее всего мусор или не пресет
            res.skipped.append(name)
            continue
        target_name = rename.get(name, name)
        tsec = target.section(target_name)
        if tsec is None:
            target.sections[target_name] = Section(
                name=target_name,
                header_raw=f"[{target_name}]",
                line_no=0)
            tsec = target.sections[target_name]
            for k, v in pairs.items():
                tsec.add(k, v)
            res.added.append(target_name)
            continue

        existing = tsec.pairs()
        existing_low = {e.lower(): e for e in existing}
        # политика конфликта действует на КЛЮЧИ, а не на секцию целиком:
        # непротиворечивые ключи всё равно полезны
        applied = False
        for k, v in pairs.items():
            orig = existing_low.get(k.lower())
            if orig is not None:
                if existing[orig] == v:
                    continue
                res.conflicts.append((target_name, k,
                                      f"{existing[orig]} -> {v}"))
                if on_conflict == "skip":
                    continue
            if tsec.set(k, v):
                applied = True
            else:
                tsec.add(k, v)
                applied = True
        if applied:
            res.updated.append(target_name)
        else:
            res.skipped.append(target_name)
    return res


def export_sections(ini: IniFile, names: list[str] | None = None,
                    include_globals: bool = True) -> str:
    """Собирает новый INI с выбранными секциями (для публикации/обмена)."""
    out: list[str] = []
    if include_globals:
        g = ini.globals()
        if g:
            out.append("[*]")
            out.extend(f"{k} = {v}" for k, v in g.items())
            out.append("")
    for name, sec in ini.sections.items():
        if name == "*":
            continue
        if names and name not in names:
            continue
        out.append(f"[{name}]")
        for k, v in sec.pairs().items():
            out.append(f"{k} = {v}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def annotate_from_ini(ini: IniFile) -> dict[str, dict]:
    """Достаёт из комментариев над секцией заявленные замеры.

    Комментарии у пользователя — источник знаний (tg, VRAM, приёмка),
    но они прошиты текстом. Здесь вытаскиваются грубые регулярками, чтобы
    положить их в measurements.json и дальше считать по ним.
    """
    out: dict[str, dict] = {}
    for name, sec in ini.sections.items():
        if name == "*":
            continue
        text = "\n".join(e.raw for e in sec.entries if e.key is None)
        # комментарии перед заголовком секции
        idx = ini.text.find(f"[{name}]")
        if idx > 0:
            head = ini.text[:idx].rsplit("\n\n", 1)[-1]
            text = head + "\n" + text
        rec: dict = {}
        m = re.search(r"tg\s*[:=]?\s*([\d.]+)", text, re.I)
        if m:
            rec["gen_tps"] = float(m.group(1))
        m = re.search(r"VRAM[^\d]*(\d{3,5})\s*(?:MiB|МБ|MB)", text, re.I)
        if m:
            rec["vram_mib"] = int(m.group(1))
        m = re.search(r"min free\s*(\d{3,5})\s*MiB", text, re.I)
        if m:
            rec["min_free_mib"] = int(m.group(1))
        m = re.search(r"prefill\s*[:=]?\s*([\d.]+)", text, re.I)
        if m:
            rec["prefill_tps"] = float(m.group(1))
        if re.search(r"needle OK\s*3/3|needle_ok", text, re.I):
            rec["needle_ok"] = True
        if re.search(r"КАНДИДАТ|провалил|не прошёл|регресс", text, re.I):
            rec["rejected"] = True
        if rec:
            out[name] = rec
    return out
