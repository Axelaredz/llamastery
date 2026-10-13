#!/usr/bin/env python3
"""Обернуть пользовательский вывод в `t()` — автоматически и безопасно.

Инструмент разработчика, парный к `tools/extract_i18n.py`.

Почему нельзя обернуть вообще всё. Русская строка в этом коде бывает
четырёх разных видов, и переводить можно не всё:

* вывод пользователю — переводится;
* сравнение с входными данными (`text.startswith("падает")`) — нельзя:
  переведём, и разберётся уже непереведённый текст, то есть проверка молча
  перестанет срабатывать;
* ключ в JSON, раздел INI, имя секции — это формат на диске, а не текст для
  человека;
* образец для regex — тем более.

Поэтому автоматика трогает только то, что физически уходит в пользовательский
вывод: строковые аргументы `print(...)`, `sys.stderr.write(...)`, `*stream.write`.
Всё остальное остаётся для ручного разбора.

f-строки превращаются в `t("... {0} ... {1}", выр0, выр1)`: подстановки
получают позиционные номера, поэтому переводчик может переставить их — для
немецкого и английского порядок слов в предложении другой.

Запуск:

    python3 tools/apply_i18n.py bin/llamastery            # показать, что изменится
    python3 tools/apply_i18n.py bin/llamastery --write    # применить
    python3 tools/apply_i18n.py bin/llamastery --dry-run  # только число правок
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def _docstring_nodes(tree: ast.AST) -> set[int]:
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


def _is_user_output(node: ast.AST) -> bool:
    """Строка уходит пользователю, а не в логику и не на диск."""
    # print(...)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id == "print":
        return True
    # stream.write(...) — sys.stderr, log, out и подобное
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
            node.func.attr == "write":
        return True
    # input("...") — приглашение мастера
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id == "input":
        return True
    return False


# Приёмники текста, которые не печатают, а копят сообщения для отчёта.
# Совпадение строго по получателю: `rep.add(...)` собирает предупреждения
# валидатора, а `dirty.add(x)` и `seen.add(x)` — обычные `set.add`, и их
# аргументы переводить нельзя. Поэтому правило не «метод add», а конкретный
# получатель; число — первый индекс переводимого аргумента.
#
#   rep.add(level, section, key, msg, hint="")
#            0       1      2    3     4
SINK_ARG_FROM = {
    ("rep", "add"): 3,
    ("report", "add"): 3,
    ("wizard", "note"): 0,
    ("wizard", "warn"): 0,
}


def _sink_start(node: ast.AST, file: str):
    """Индекс первого переводимого аргумента, если вызов — приёмник текста."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return SINK_ARG_FROM.get((f.value.id, f.attr))
    # note(...) / warn(...) внутри самого мастера: там это локальные функции
    if file.endswith("wizard.py") and isinstance(f, ast.Name):
        return SINK_ARG_FROM.get(("wizard", f.id))
    return None


def _rewrite_fstring(node: ast.JoinedStr) -> str:
    """ast.JoinedStr → t("текст {0} {1}", выр0, выр1).

    Плейсхолдеры в каталоге получают номера, а не выражения: переводчик
    переставляет `{0}` и `{1}` сам, а движок не тащит синтаксис Python
    в файл переводов.
    """
    pieces: list[str] = []
    holes: list[str] = []
    for val in node.values:
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            pieces.append(val.value)
        elif isinstance(val, ast.FormattedValue):
            # И `!r`, и `:.2f` — часть вывода, а не украшение.
            # «нет секции 'имя» показывает имя, которое копируют в команду;
            # «0.88 GiB» не превращается в «0.883375 GiB». Оба переносятся
            # в плейсхолдер: str.format понимает `{0!r}` и `{1:.2f}` так же,
            # как f-строка.
            idx = len(holes)
            spec = "{" + str(idx)
            if val.format_spec is not None:
                spec += ":" + ast.unparse(val.format_spec)
            if val.conversion == ord("r"):
                spec += "!r"
            elif val.conversion == ord("s"):
                spec += "!s"
            spec += "}"
            pieces.append(spec)
            holes.append(ast.unparse(val.value))
    text = "".join(pieces)
    if not holes:
        return f"t({text!r})"
    return f"t({text!r}, " + ", ".join(holes) + ")"


def plan(path: Path) -> list[tuple[int, int, int, int, str]]:
    """Список правок (начало_строка, колонка, конец_строка, колонка, текст)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    docs = _docstring_nodes(tree)
    edits: list[tuple[int, int, int, int, str]] = []
    # аргументы уже занятые правкой выше: вложенные вызовы не трогаем дважды
    taken: set[tuple[int, int]] = set()

    for node in ast.walk(tree):
        start = None
        if _is_user_output(node):
            start = 0
        else:
            start = _sink_start(node, str(path))
        if start is None:
            continue
        for arg in node.args[start:]:
            if id(arg) in docs or (arg.lineno, arg.col_offset) in taken:
                continue
            # f-строка с кириллицей
            if isinstance(arg, ast.JoinedStr):
                text = "".join(v.value for v in arg.values
                               if isinstance(v, ast.Constant))
                if not _has_cyrillic(text):
                    continue
                edits.append((arg.lineno, arg.col_offset,
                              arg.end_lineno, arg.end_col_offset,
                              _rewrite_fstring(arg)))
                taken.add((arg.lineno, arg.col_offset))
            # обычная строка
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and _has_cyrillic(arg.value):
                edits.append((arg.lineno, arg.col_offset,
                              arg.end_lineno, arg.end_col_offset,
                              f"t({arg.value!r})"))
                taken.add((arg.lineno, arg.col_offset))
    return edits


def apply_edits(src: str, edits) -> str:
    """Применить правки с конца файла к началу.

    Так смещения не «уезжают»: замена в конце не сдвигает позиции начальных
    правок, и пересчитывать ничего не нужно.

    Про колонки. У `ast` смещения узлов заданы в **байтах UTF-8**, а не в
    символах. Для кириллицы это разные числа, и подстановка по байтовому
    смещению разрезает строку пополам — ровно так и вышло на первой же русской
    строке. Поэтому каждый байтовый столбец переводится в символьный.
    """
    lines = src.splitlines(keepends=True)
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    def pos(lineno: int, col: int) -> int:
        raw = lines[lineno - 1].encode("utf-8")
        charcol = len(raw[:col].decode("utf-8", errors="ignore"))
        return starts[lineno - 1] + charcol

    for el, ec, el2, ec2, text in sorted(edits, key=lambda e: -pos(e[0], e[1])):
        a, b = pos(el, ec), pos(el2, ec2)
        src = src[:a] + text + src[b:]
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description="обернуть вывод в t()")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write", action="store_true", help="применить к файлам")
    ap.add_argument("--dry-run", action="store_true", help="только количество")
    a = ap.parse_args()

    total = 0
    for rel in a.files:
        p = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        edits = plan(p)
        total += len(edits)
        if a.dry_run:
            print(f"{rel}: правок {len(edits)}")
            continue
        if not edits:
            continue
        new = apply_edits(p.read_text(encoding="utf-8"), edits)
        if a.write:
            p.write_text(new, encoding="utf-8")
            print(f"{rel}: применено {len(edits)}")
        else:
            # показать первые строки будущих изменений для ревью
            try:
                ast.parse(new)
            except SyntaxError as e:
                print(f"! {rel}: результат не парсится ({e}) — не применяю")
                return 1
            print(f"{rel}: было бы {len(edits)} правок, код после — "
                  f"валиден, запиши --write")
    if a.dry_run:
        print(f"\nвсего правок: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())