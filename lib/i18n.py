"""Языки вывода: каталоги, определение языка, подстановка.

Зачем. Инструмент писался по-русски, и весь вывод — сотни строк в
`bin/llamastery` и `lib/wizard.py` — зашит на русском. Кто-то из пользователей
читает по-английски быстрее, чем по-русски, и для него инструмент
неудобен не из-за логики, а из-за языка.

Ключ перевода — сам русский текст, а не искусственный идентификатор.
Это сделано намеренно:

* каталог не может «разъехаться» с кодом — ключ всегда рядом с местом
  использования, и переименование строки автоматически ломает старый перевод
  в тесте, а не молча оставляет его мёртвым;
* непереведённая строка показывает русский оригинал, а не падает и не
  печатает `KeyError` в вывод утилиты о VRAM;
* `grep` по русскому тексту находит и код, и его переводы.

Языки лежат в `lib/locale/<код>.json`, словарь ключ → строка или словарь
форм множественного числа (`one`/`few`/`many`).

Определение языка: `--lang`, потом `LLAMASTERY_LANG`, потом `LANG`/`LC_ALL`,
иначе русский. Порядок именно такой: переменная окружения может быть унаследована
от сессии, а `--lang` — осознанный выбор человека, он и должен побеждать.
"""

import json
import os
from pathlib import Path

LOCALE_DIR = Path(__file__).resolve().parent / "locale"

#: Язык, на котором написан исходный код. Базовый: каталога для него нет,
#: неизвестный ключ показывается как есть.
SOURCE_LANG = "ru"

#: Языки с наличием каталога, в порядке вывода в `llamastery lang`.
BUILTIN = ("ru", "en", "de", "zh")

_cache: dict = {}
_current: str = SOURCE_LANG


def _catalog_path(code: str) -> Path:
    return LOCALE_DIR / f"{code}.json"


def _load(code: str) -> dict:
    """Каталог языка; пустой словарь, если файла нет или он битый.

    Битый каталог не должен ронять инструмент: причина, по которой
    переводы могут не работать, — у пользователя перед глазами, а вот
    падение с `JSONDecodeError` означало бы, что `budget` вообще не работает.
    """
    if code in _cache:
        return _cache[code]
    data: dict = {}
    p = _catalog_path(code)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    _cache[code] = data
    return data


def available() -> list[str]:
    """Языки, которыми инструмент умеет говорить.

    `ru` всегда первый и всегда есть: исходник на нём, и даже при потере всех
    каталогов инструмент продолжит работать, просто по-русски.

    Остальные попадают в список по факту наличия файла каталога. Язык из
    `BUILTIN` с пустым или битым файлом не показывается: переключиться на него
    можно, но говорить на нём будет нечем, и в списке это только вводит в
    заблуждение.
    """
    found = {"ru"}
    try:
        for p in LOCALE_DIR.glob("*.json"):
            code = p.stem
            if code == SOURCE_LANG:
                continue
            # непустой перевод — говорить есть чем. Каталог, где все значения
            # пустые, не считается: переключиться на него можно, но на экране
            # окажется русский, и в списке это вводит в заблуждение.
            cat = _load(code)
            if any(v for v in cat.values()):
                found.add(code)
    except OSError:
        pass
    order = [c for c in BUILTIN if c in found]
    return order + sorted(found - set(order))


def _from_env() -> str | None:
    raw = (os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES")
           or os.environ.get("LANG") or "")
    # LANG выглядит как "de_DE.UTF-8" — берём только код до "_" или ".".
    code = raw.split(".")[0].split("_")[0].split("@")[0].strip().lower()
    return code or None


def detect(explicit: str | None = None) -> str:
    """Какой язык выбрать.

    `explicit` — значение `--lang`. Переменные окружения — за ними, потому что
    сессия терминала тоже про язык: человек, у которого `LANG=de_DE.UTF-8`,
    почти наверняка хочет читать по-немецки и не обязан это повторять флагом в
    каждой команде.
    """
    for cand in (explicit, os.environ.get("LLAMASTERY_LANG"), _from_env()):
        if not cand:
            continue
        code = cand.split(".")[0].split("_")[0].strip().lower()
        if code in BUILTIN or _load(code):
            return code
    return SOURCE_LANG


def set_lang(code: str) -> str:
    """Зафиксировать язык для процесса. Возвращает применённый код."""
    global _current
    _current = code
    return _current


def current() -> str:
    return _current


def _plural_index(n: int, code: str) -> str:
    """Форма множественного числа.

    Русский и украинский различают одну/несколько/много: «1 коммит»,
    «2 коммита», «5 коммитов». Английский и немецкий — только одну/остальные,
    китайский не склоняет число вовсе. Ошибка здесь заметна сразу и выглядит
    как «2 коммитов», поэтому правило написано явно, а не через общий
    англоязычный `%d` хак.
    """
    if code in ("ru", "uk", "be"):
        if n % 10 == 1 and n % 100 != 11:
            return "one"
        if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            return "few"
        return "many"
    return "one" if n == 1 else "other"


def _render(entry, code: str, n: int | None) -> str:
    """Достать из записи каталога строку нужной формы."""
    if isinstance(entry, dict):
        if n is not None:
            idx = _plural_index(n, code)
            if idx in entry:
                return entry[idx]
            # формы может не быть — берём любую имеющуюся, лучше один
            # нечитаемый перевод, чем падение или пустая строка
            for fallback in ("other", "many", "one", "few"):
                if fallback in entry:
                    return entry[fallback]
            return ""
        return ""
    return entry if isinstance(entry, str) else ""


def _fill(text: str, args: tuple, kw: dict) -> str:
    """Подставить параметры, не падая ни на одной стороне.

    `str.format` требует, чтобы каждая `{` в тексте была частью плейсхолдера.
    В выводе этого инструмента встречаются строки с фигурными скобками
    не-плейсхолдерами — примеры JSON, `repr` словарей, куски `models.ini`.
    Такая строка приходит в перевод как есть и должна уйти на вывод как есть,
    а не превращать `budget` в трассировку. Поэтому подстановка «мягкая»
    с обеих сторон: и кривой перевод, и кривой оригинал дают исходный текст.
    """
    if not args and not kw:
        return text
    try:
        return text.format(*args, **kw)
    except (KeyError, IndexError, ValueError):
        return text


def t(msg: str, /, *args, n: int | None = None, **kw) -> str:
    """Перевести строку и подставить параметры.

    Русский текст в коде остаётся рабочим без каталогов вообще, поэтому
    вызов безопасен всегда и не обязан быть готовым заранее.

        t("поручено")
        t("требуют внимания: {names}", names=", ".join(stale))
        t("{0} коммит", 3)              # → «3 коммита»
        t("{n} коммит", n=3)            # → «3 коммита»

    Позиционные плейсхолдеры `{0}`, `{1}` удобнее при автообёртке: выражения
    подставляются как есть, без придумывания имён, а переводчик всё равно
    переставляет их по номерам. Именованные удобнее руками, когда важно
    читаемость каталога. `n` выбирает форму множественного числа и потому
    зарезервирован: обычный аргумент с таким именем означал бы «а заодно ещё
    и число для склонения».
    """
    if kw:
        n = kw.get("n", n)
    if _current == SOURCE_LANG:
        entry = _render(_load(SOURCE_LANG).get(msg), SOURCE_LANG, n) or msg
    else:
        entry = _render(_load(_current).get(msg), _current, n)
        if not entry:
            # нет перевода — показываем исходник, а не пустоту и не ошибку
            return _fill(msg, args, kw)
    return _fill(entry, args, kw)


#: привычная короткая форма — как в gettext
_ = t


def init(explicit: str | None = None) -> str:
    """Определить и применить язык. Вызывается один раз при старте CLI."""
    return set_lang(detect(explicit))