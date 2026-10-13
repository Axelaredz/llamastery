#!/usr/bin/env python3
"""Извлечь русские строки из кода — каркасы каталогов перевода.

Инструмент разработчика, не часть рантайма. Ключ перевода — сам русский
текст (см. `lib/i18n.py`), поэтому этот скрипт решает две задачи:

1. найти строки, которые ещё не обёрнуты в `t()`, — их надо обернуть;
2. собрать ключи в каркасы каталогов, чтобы переводчику не приходилось
   выписывать ключи вручную и ошибаться в них.

Разбирает код через `ast`, а не регулярками: регулярка не отличает строку от
комментария, не видит склейки в f-строках и не знает, где заканчивается
докстринг из тройных кавычек.

Запуск:

    python3 tools/extract_i18n.py                 # отчёт и каркасы в stdout
    python3 tools/extract_i18n.py --skeleton en  # записать lib/locale/en.json
    python3 tools/extract_i18n.py --check         # ненайденные ключи в каталогах
"""

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCALE_DIR = ROOT / "lib" / "locale"

TARGETS = ("bin/llamastery", *sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "lib").glob("*.py")))


def _has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id() узлов, которые являются докстрингами.

    Докстринги не переводятся: они читаются разработчиком в исходнике, а не
    пользователем в выводе. Переводить их — значит держать две копии
    документации, которые разъедутся при первом же правке кода.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            out.add(id(body[0].value))
    return out


def scan(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Вернуть (обёрнутые, голые, f-строки) для одного файла."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as e:
        print(f"! пропущен {path}: {e}", file=sys.stderr)
        return [], [], []
    docs = _docstring_nodes(tree)
    wrapped, bare, fstrs = [], [], []

    for node in ast.walk(tree):
        # уже обёрнутые: t("...") / _("...")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("t", "_") and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                wrapped.append({
                    "key": a0.value, "line": a0.lineno, "kind": "wrapped"})
            continue
        # f-строки требуют ручной работы: подстановки надо назвать
        if isinstance(node, ast.JoinedStr):
            if id(node) in docs:
                continue
            parts = [v.value for v in node.values
                     if isinstance(v, ast.Constant) and _has_cyrillic(v.value)]
            if not parts:
                continue
            holes = [ast.unparse(v.value) for v in node.values
                     if isinstance(v, ast.FormattedValue)]
            fstrs.append({"line": node.lineno, "parts": parts, "holes": holes})
            continue
        # обычные строки
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docs or not _has_cyrillic(node.value):
                continue
            bare.append({"key": node.value, "line": node.lineno, "kind": "bare"})

    return wrapped, bare, fstrs


def collect() -> tuple[dict, list[dict]]:
    """Собрать по всему проекту: ключи → источники, и список f-строк.

    Различаем «обёрнуто» и «голое», иначе отчёт невозможно читать: цифра
    обёрнутых строк нужна, чтобы понимать, сколько работы осталось.
    """
    keys: dict[str, dict] = {}
    fstrs: list[dict] = []
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            continue
        w, b, f = scan(p)
        for rec in w + b:
            e = keys.setdefault(rec["key"], {"sites": [], "wrapped": 0})
            e["sites"].append(f"{rel}:{rec['line']}")
            e["wrapped"] += rec["kind"] == "wrapped"
        for rec in f:
            fstrs.append({"file": rel, **rec})
    return keys, fstrs


def main() -> int:
    ap = argparse.ArgumentParser(description="извлечь русские строки")
    ap.add_argument("--skeleton", metavar="LANG",
                    help="записать каркас каталога (только непереведённые)")
    ap.add_argument("--keep", action="store_true",
                    help="не перезаписывать существующие значения каталога")
    ap.add_argument("--check", action="store_true",
                    help="показать ключи в каталогах, которых больше нет в коде")
    a = ap.parse_args()

    keys, fstrs = collect()
    LOCALE_DIR.mkdir(parents=True, exist_ok=True)

    if a.skeleton:
        tgt = LOCALE_DIR / f"{a.skeleton}.json"
        data = {}
        if a.keep and tgt.exists():
            try:
                data = json.loads(tgt.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        # в каталог идут только строки, реально обёрнутые в t(): остальные —
        # это сравнения, ключи на диске и regex-образцы, их переводить нельзя,
        # и в каталоге им не место
        wanted = [k for k, e in keys.items() if e["wrapped"]]
        added = 0
        for k in wanted:
            if k not in data:
                data[k] = ""
                added += 1
        tgt.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        print(f"{tgt}: ключей {len(data)}, добавлено {added} "
              f"(всего обёрнуто {len(wanted)} из {len(keys)})")
        return 0

    if a.check:
        stale = 0
        for p in sorted(LOCALE_DIR.glob("*.json")):
            if p.stem == "ru":
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                print(f"! {p.name}: не читается")
                continue
            missing = [k for k in data if k not in keys]
            empty = [k for k, v in data.items() if not v]
            print(f"{p.name}: {len(data)} ключей, пустых {len(empty)}, "
                  f"устаревших {len(missing)}")
            stale += len(missing)
        print(f"\nвсего ключей в коде: {len(keys)}")
        return 0

    wrapped = sum(1 for e in keys.values() if e["wrapped"])
    bare = sum(1 for e in keys.values() if not e["wrapped"])
    print(f"уникальных русских строк:            {len(keys)}")
    print(f"  уже обёрнуты в t():                {wrapped}")
    print(f"  голые (надо обернуть):              {bare}")
    print(f"f-строк с кириллицей (ручная работа): {len(fstrs)}")
    print(f"\nфайлов просканировано: {len(TARGETS)}")
    if fstrs:
        print("\nf-строки (подстановки надо превратить в именованные):")
        for rec in fstrs:
            joined = " + ".join(repr(p) for p in rec["parts"])
            holes = ", ".join(rec["holes"])
            print(f"  {rec['file']}:{rec['line']}")
            print(f"      текст: {joined}")
            if holes:
                print(f"      подстановки: {holes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())