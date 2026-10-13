"""Интерактивный слой мастера: стрелки, очистка, новое состояние.

Зачем: построчный ввод (номер + Enter) заставляет держать в голове весь
прошлый вывод. Здесь каждый ответ очищает консоль и рисует заново: сверху
хлебные крошки (что уже выбрано), дальше текущий вопрос. Выбор — стрелками
клавиатуры, не номерами.

Только стандартная библиотека. Без TTY слой выключен, и вызывающий идёт
старым построчным путём — там же остаются тесты, пайпы и перенаправления.
"""

import os
import select
import shutil
import sys

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover — Windows без termios
    _HAS_TERMIOS = False

UP = "\x1b[A"
DOWN = "\x1b[B"
LEFT = "\x1b[D"
RIGHT = "\x1b[C"
ENTER_KEYS = ("\r", "\n")
ESC = "\x1b"

BACK = "back"
PICK = "pick"
PASTE = "paste"

REV_ON = "\033[7m"
REV_OFF = "\033[0m"

STAR = "★"


def enabled() -> bool:
    """Интерактив возможен: TTY с обеих сторон и не dumb-терминал."""
    return (_HAS_TERMIOS
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.environ.get("TERM", "") != "dumb")


def clear() -> None:
    """Очистить консоль. Вне TTY — ничего не делает."""
    if enabled():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


class _Raw:
    """Сырой режим на время чтения одной клавиши, потом всё назад."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.fd = stream.fileno()
        self.saved = None

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def _read_char(fd: int) -> str:
    """Один символ с fd, без буферизации.

    sys.stdin.read() сюда не годится: TextIOWrapper читает вперёд, и select
    после Esc видит пустоту, даже когда продолжение уже пришло, — стрелки
    превращались бы в Esc. Поэтому читаем сырые байты и декодируем UTF-8
    сами (важно для «д»/«н» в вопросе да/нет: это двухбайтовые символы).
    """
    first = os.read(fd, 1)
    if not first:
        raise EOFError
    head = first[0]
    if head < 0x80:
        extra = 0
    elif head & 0xE0 == 0xC0:
        extra = 1
    elif head & 0xF0 == 0xE0:
        extra = 2
    elif head & 0xF8 == 0xF0:
        extra = 3
    else:
        return first.decode("utf-8", "replace")
    buf = first
    while extra:
        more = os.read(fd, extra)
        if not more:
            break
        buf += more
        extra -= len(more)
    return buf.decode("utf-8", "replace")


def read_key() -> str:
    """Одна клавиша. Стрелки возвращаются целиком (например "\\x1b[A").

    Одиночный Esc отличается от стрелки паузой: после ESC ждём продолжение
    последовательности 50 мс, нет продолжения — это Esc.
    """
    fd = sys.stdin.fileno()
    with _Raw(sys.stdin):
        ch = _read_char(fd)
        if ch != ESC:
            return ch
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            return ESC
        seq = os.read(fd, 2).decode("ascii", "replace")
        return ESC + seq


def _height() -> int:
    try:
        return shutil.get_terminal_size().lines
    except OSError:
        return 24


def render_crumbs(out, crumbs: list[tuple[str, str]]) -> None:
    """Хлебные крошки: что уже отвечено на этом проходе."""
    for i, (q, a) in enumerate(crumbs, 1):
        out.write(f"  {i} {q} → {a}\n")
    if crumbs:
        out.write("\n")


def _window(cursor: int, total: int, max_rows: int) -> tuple[int, int]:
    """Видимое окно пунктов: курсор всегда внутри, начало не дальше нужды."""
    if total <= max_rows:
        return 0, total
    start = min(max(cursor - max_rows // 2, 0), total - max_rows)
    return start, start + max_rows


def menu_select(title: str, sub: str, items: list[tuple[str, str, bool]],
                cursor: int, crumbs: list[tuple[str, str]],
                footer: str, key_reader=read_key,
                out=None, height: int = 0,
                allow_paste: bool = False) -> tuple[str, int | None]:
    """Меню со стрелками. Возвращает (PICK, idx), (BACK, None) или (PASTE, None).

    items: [(подпись, пояснение, звезда-рекомендация)].
    Клавиши: ↑/↓ — ходить, Enter — выбрать, цифры — быстрый выбор,
    0/Esc/← — назад, "/" — вставить свой текст (только при allow_paste=True).
    Ctrl-C — выход как раньше (130).
    key_reader/out/height — для тестов: подсовывают клавиши и ловят кадр.
    """
    out = sys.stdout if out is None else out
    height = height or _height()
    cursor = max(0, min(cursor, len(items) - 1)) if items else 0
    while True:
        max_rows = max(3, height - 10)
        start, end = _window(cursor, len(items), max_rows)
        if out is sys.stdout:
            clear()
        render_crumbs(out, crumbs)
        out.write(f"── {title}\n")
        if sub:
            out.write(f"  {sub}\n")
        if start > 0:
            out.write("    …\n")
        for i in range(start, end):
            label, desc, star = items[i]
            mark = f" {STAR}" if star else "   "
            tail = f"  {desc}" if desc else ""
            row = f"  {i + 1}.{mark} {label}{tail}"
            if i == cursor:
                out.write(f"{REV_ON}► {row} {REV_OFF}\n")
            else:
                out.write(f"  {row}\n")
        if end < len(items):
            out.write("    …\n")
        out.write(f"\n  {footer}\n")
        out.flush()
        try:
            key = key_reader()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if key == "\x03":
            raise SystemExit(130)
        if key in (UP, "k"):
            cursor = (cursor - 1) % len(items) if items else 0
        elif key in (DOWN, "j"):
            cursor = (cursor + 1) % len(items) if items else 0
        elif key in ENTER_KEYS:
            return PICK, cursor
        elif key in ("0", ESC, LEFT):
            return BACK, None
        elif key == "/" and allow_paste:
            return PASTE, None
        elif key.isdigit() and items:
            idx = int(key) - 1
            if 0 <= idx < len(items):
                return PICK, idx


def confirm_select(title: str, sub: str, crumbs: list[tuple[str, str]],
                   recommended_yes: bool, key_reader=read_key,
                   out=None, height: int = 0) -> tuple[str, bool | None]:
    """Да/нет стрелками. Возвращает (PICK, ответ) или (BACK, None).

    Безопасность как в построчном режиме: курсор стартует на «нет», Enter
    соглашается только с подсвеченным. y/д подсвечивает «да», n/н — «нет»,
    ←/→ двигают. Пустой Enter = «нет». 0/Esc = назад.
    """
    out = sys.stdout if out is None else out
    height = height or _height()
    cur = 0  # 0 = нет, 1 = да. Дефолт всегда «нет», что бы ни рекомендовали.
    while True:
        if out is sys.stdout:
            clear()
        render_crumbs(out, crumbs)
        out.write(f"── {title}\n")
        if sub:
            out.write(f"  {sub}\n")
        if recommended_yes:
            out.write("  ★ рекомендовано: да (но выбираешь ты)\n")
        cells = []
        for i, word in enumerate(("нет", "да")):
            cells.append(f"{REV_ON} {word} {REV_OFF}" if i == cur
                         else f"  {word} ")
        out.write(f"\n  {' '.join(cells)}\n")
        out.write("\n  ←/→ выбор, Enter — подтвердить "
                  "(подсвеченное), y/д/n/н — быстро, 0/Esc — назад\n")
        out.flush()
        try:
            key = key_reader()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if key == "\x03":
            raise SystemExit(130)
        low = key.lower()
        if key in (LEFT, RIGHT):
            cur = 1 - cur
        elif low in ("y", "yes", "д", "да"):
            cur = 1
        elif low in ("n", "no", "н", "нет"):
            cur = 0
        elif key in ENTER_KEYS:
            return PICK, bool(cur)
        elif key in ("0", ESC):
            return BACK, None
