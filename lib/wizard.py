"""Интерактивный мастер: голый запуск llamastery ведёт за руку.

Зачем: флагов десятки, а эффективный маршрут один — мастер показывает его
сразу и везде предлагает дефолт ★ (Enter = самый эффективный вариант).

Маршрут: состояние → «что делаем» (меню строится по факту: роутер поднят
или нет, что загружено, жив ли swap) → дальше только релевантные шаги.

Правила мастера:
  * чтение — молча и само (реестр, файлы моделей, замеры, порты);
  * запись/загрузка/сеть — только после явного «да»;
  * меню показывает только то, что сейчас возможно: нельзя предложить
    «остановить» то, что не запущено;
  * запуск шагов — через собственный CLI подпроцессом, чтобы не дублировать
    логику команд.
"""

import subprocess
import sys
from pathlib import Path

from . import builds, measure, paths, server, swap, vram

STAR = "★"
ARROW = "│"
ISVALORUM = "https://huggingface.co/collections/IsValorum/abliterated"


# ── оформление: вопрос и ответ видно сразу ──
_qcount = 0


def head(title: str, sub: str = "") -> None:
    print()
    print(f"── {title} " + "─" * max(0, 60 - len(title)))
    if sub:
        print(f"  {sub}")


def qnum() -> str:
    """Следующий номер вопроса.

    Раньше номера были вписаны в заголовки текстом, и при пропуске вопроса
    (например «картинки» не спрашивались — mmproj уже есть) последовательность
    путалась: 5, потом сразу 7. Счётчик считает заданные вопросы, а не строки.
    """
    global _qcount
    _qcount += 1
    return str(_qcount)


def reset_questions() -> None:
    global _qcount
    _qcount = 0


def note(text: str) -> None:
    print(f"  {text}")


def warn(text: str) -> None:
    print(f"  ! {text}")


BACK = "◄ назад"
MAX_STEPS = 500          # предохранитель от бесконечного «назад»


class Nav:
    """Экран за экраном: 0 = вернуться на шаг назад и переспросить.

    Ответы хранятся по ключу, поэтому при возврате значения после него
    помечаются как несвежие: зависеть от них дальше нельзя, их либо
    переспрашивают, либо пересчитывают.
    """

    def __init__(self) -> None:
        self.i = 0
        self.answers: dict = {}
        self.stale: set = set()

    def walk(self, steps: list[tuple]) -> dict | None:
        """steps: [(ключ, функция(answers))].

        Функция возвращает значение или BACK. Возврат BACK уводит на
        предыдущий экран; если он первый — возвращается None, и вызывающий
        код показывает главное меню заново.
        """
        self.i = 0
        self.stale = set()
        # предохранитель: экран, который всегда отвечает «назад», иначе
        # увел бы мастера в бесконечный цикл (и тесты — тоже)
        for _ in range(MAX_STEPS):
            if not 0 <= self.i < len(steps):
                return self.answers if self.i >= len(steps) else None
            key, fn = steps[self.i]
            val = fn(self.answers)
            if val is BACK or val == BACK:
                if self.i == 0:
                    return None
                self.i -= 1
                self.stale.add(steps[self.i][0])
                continue
            self.stale.discard(key)
            self.answers[key] = val
            self.i += 1
        warn("слишком много переходов назад — возвращаюсь в меню")
        return None


def _read(prompt: str) -> str:
    try:
        return input(f"  {ARROW} {prompt} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)


def ask(title: str, options: list[tuple[str, str]], default: int,
        sub: str = "") -> int:
    """Вопрос с нумерованным списком. Печатает вопрос, ответ и итог.

    options: [(подпись, пояснение)]. Пункт с пометкой ★ — рекомендация.
    """
    head(title, sub)
    for i, (label, desc) in enumerate(options, 1):
        mark = f" {STAR}" if i == default else "  "
        tail = f"  {desc}" if desc else ""
        print(f"    {i}.{mark} {label}{tail}")
    raw = _read(f"ответ (Enter = {default}, 0 = {BACK}) ▸")
    if raw == "0":
        print(f"    {ARROW} ответ: {BACK}")
        return BACK
    return clamp_choice(raw, len(options), default) - 1


def ask_pick(title: str, options: list[tuple[str, str]], default: int,
             sub: str = "") -> str:
    """Вопрос со списком → возвращает выбранную метку."""
    idx = ask(title, options, default, sub)
    if idx == BACK:
        return BACK
    label = options[idx][0]
    print(f"    {ARROW} ответ: {label}")
    return label


def ask_yn(title: str, default_yes: bool, sub: str = "") -> bool:
    head(title, sub)
    hint = "Y/n" if default_yes else "y/N"
    raw = _read(f"{hint} {STAR} ответ (Enter = "
                f"{'да' if default_yes else 'нет'}, 0 = {BACK}) ▸").lower()
    if raw == "0":
        print(f"    {ARROW} ответ: {BACK}")
        return BACK
    yes = default_yes if not raw else raw in ("y", "yes", "д", "да")
    print(f"    {ARROW} ответ: {'да' if yes else 'нет'}")
    return yes


def ask_text(title: str, default: str, sub: str = "") -> str:
    head(title, sub)
    raw = _read(f"Enter = {default}, 0 = {BACK} ▸")
    if raw == "0":
        print(f"    {ARROW} ответ: {BACK}")
        return BACK
    val = raw or default
    print(f"    {ARROW} ответ: {val}")
    return val


def clamp_choice(raw: str, n: int, default: int) -> int:
    """Номер пункта 1..n, мусор = дефолт."""
    try:
        v = int(raw)
        if 1 <= v <= n:
            return v
    except (ValueError, TypeError):
        pass
    return default


# ── чистые функции (покрыты тестами) ──
def recommend_build(reg: dict) -> str | None:
    """Самая прогрессивная сборка: faks, иначе первая роутерная собранная."""
    if "faks" in reg:
        b = reg["faks"]
        if b.server_bin.exists():
            return "faks"
    for name, b in reg.items():
        if b.router and b.server_bin.exists():
            return name
    for name, b in reg.items():
        if b.server_bin.exists():
            return name
    return None


def ngram_default(pairs: dict) -> bool:
    """ngram по умолчанию — да, но НЕ с mmproj (несовместимы)."""
    low = {k.lower(): v for k, v in pairs.items()}
    return "mmproj" not in low


def mtp_default(name: str, pairs: dict) -> bool:
    """MTP по умолчанию — да, если пресет уже про MTP/черновик."""
    low = {k.lower() for k in pairs}
    blob = (name + " " + " ".join(low)).lower()
    return ("mtp" in blob or "draft" in blob or "spec" in blob)


def find_mmproj_sibling(names: list[str], current: str) -> str | None:
    """Секция-близнец с mmproj для текущего пресета."""
    base = current.lower().replace("-mmproj", "")
    for n in names:
        ln = n.lower()
        if ln != current.lower() and "mmproj" in ln and (
                base in ln or ln.replace("-mmproj", "") == base):
            return n
    return None


def menu_for(router_up: bool, loaded: list[str], swap_up: bool) -> list[tuple[str, str]]:
    """Меню строится по состоянию: нельзя предложить «остановить» то, что не живо.

    Возвращает [(подпись, пояснение)]; первый пункт — рекомендация.
    """
    m: list[tuple[str, str]] = []
    if router_up:
        m.append(("Загрузить или сменить пресет", "сервер уже работает"))
        m.append(("Перезапустить роутер", "свежий pid, настройки перечитаны"))
        m.append(("Остановить роутер", "VRAM освободится полностью"))
    else:
        m.append(("Запустить роутер", "поднять сервер для загрузки моделей"))
        m.append(("Запустить роутер и загрузить пресет", "сразу рабочая модель"))
    if swap_up:
        m.append(("Остановить llama-swap", "модели в VRAM уйдут вместе с ним"))
    else:
        m.append(("Запустить llama-swap", "хот-свап моделей на одном порту"))
    m.append(("Настроить заново", "сборка → пресет → контекст → план"))
    m.append(("Только показать состояние", "ничего не менять"))
    return m


def menu_default(m: list[tuple[str, str]], router_up: bool) -> int:
    """Дефолт — то, что чаще нужно при данном состоянии."""
    return 2 if router_up else 1


# ── исполнение ──
def _bin() -> str:
    return str(Path(__file__).resolve().parent.parent / "bin" / "llamastery")


def _run_cli(*args: str) -> int:
    print()
    print(f"  {ARROW} выполняю: llamastery {' '.join(args)}")
    try:
        return subprocess.call([_bin(), *args])
    except OSError as exc:
        warn(f"не запустилось: {exc}")
        return 1


def _state_lines(swap_url: str) -> dict:
    """Состояние одним экраном: роутер, swap, VRAM."""
    st = server.status()
    sw = swap.status(swap_url, timeout=3.0)
    used, total = vram.gpu_used_mib(), vram.gpu_total_mib()
    running = []
    if sw.get("up"):
        running = [r["model"] for r in swap.running_models(swap_url, timeout=3.0)]
    return {"router": st, "swap": sw, "swap_models": running,
            "vram_used": used, "vram_total": total}


def print_state(s: dict, swap_url: str) -> None:
    head("состояние", "читается молча, ничего не меняет")
    r = s["router"]
    note(f"роутер   {r['url']}  "
         f"{'работает' if r['up'] else 'не запущен'}"
         + (f" (pid {r['pid']}, сборка {r['build']})" if r["up"] else ""))
    if r.get("models"):
        for m in r["models"]:
            label = server.STATUS_LABEL.get(m["status"], m["status"])
            note(f"           [{m['status']}] {m['id']} — {label}")
    sw = s["swap"]
    note(f"swap     {swap_url}  "
         f"{'работает' if sw.get('up') else 'не запущен'}"
         + (f", в VRAM: {len(s['swap_models'])}" if sw.get("up") else ""))
    for m in s["swap_models"]:
        note(f"           {m}")
    if s["vram_used"] is not None and s["vram_total"]:
        free = s["vram_total"] - s["vram_used"]
        note(f"VRAM     {s['vram_used']} / {s['vram_total']} MiB "
             f"(свободно {free})")
    else:
        note("VRAM     неизвестна (nvidia-smi не найден)")


# ── шаги настройки ──
def _pick_build(reg: dict) -> str | None:
    names = list(reg)
    opts = []
    rec = recommend_build(reg)
    for name in names:
        st = reg[name].status()
        kind = "роутер" if st["router"] else "single"
        state = "собран" if st["bin_exists"] else "НЕ собран"
        ver = str(st.get("version") or "")
        opts.append((name, f"{kind}, {state}" + (f", {ver}" if ver else "")))
    default = (names.index(rec) + 1) if rec in names else 1
    label = ask_pick(f"Вопрос {qnum()} · какая сборка llama.cpp",
                     opts, default,
                     sub="faks — самая прогрессивная: роутер, n-cpu-moe, "
                         "патчи Fable. Остальные — по необходимости")
    return label


def _pick_build_registry() -> dict:
    reg = builds.all_builds()
    if reg:
        return reg
    warn("реестр сборок пуст — ищем форки автоматически")
    if ask_yn(f"Вопрос {qnum()} · найти сборки (builds detect --apply)?", True,
              sub="обычные каталоги: ~/git/llama-*, ~/llama-*"):
        _run_cli("builds", "detect", "--apply")
        reg = builds.all_builds()
    if not reg:
        warn("сборок нет. Собери llama-server и вернись: "
             "cmake -B build && cmake --build build -j")
    return reg


def measurement_of(pairs: dict, records: dict) -> dict | None:
    """Замер для пресета — по подписи конфигурации, а не по имени.

    Ключ в measurements.json — хеш значимых ключей, и имя секции в нём не
    участвует. Раньше мастер искал замер по имени и поэтому ни один не
    находил: метка «замер есть» была случайной (вешалось на первый пункт).
    """
    from . import measure as _m
    rec = records.get(_m.signature(pairs))
    if not rec:
        return None
    vram = rec.get("vram_mib") or rec.get("used_mib")
    return {"vram_gb": round(vram / 1024.0, 2) if vram else None,
            "tps": rec.get("deep_tps") or rec.get("short_tps"),
            "runs": rec.get("runs", 0)}


def _preset_options(ini) -> tuple[list[tuple[str, str]], list[str], str | None]:
    sections = [n for n in ini.names() if n != "*"]
    try:
        records = measure.load_store().get("records", {})
    except Exception:  # noqa: BLE001 — замеры опциональны
        records = {}
    opts, measured = [], []
    for name in sections:
        pairs = ini.section(name).pairs()
        low = {k.lower(): v for k, v in pairs.items()}
        m = measurement_of(pairs, records)
        bits = [f"c={low.get('c', '?')}",
                f"moe={low.get('n-cpu-moe', low.get('cpu-moe', '-'))}",
                "зрение" if "mmproj" in low else "текст"]
        if m and m["vram_gb"]:
            bits.append(f"VRAM {m['vram_gb']} GiB замер")
        elif m and m["tps"]:
            bits.append(f"{m['tps']} t/s замер")
        opts.append((name, " | ".join(bits)))
        if m:
            measured.append(name)
    default_name = measured[0] if measured else (sections[0] if sections else None)
    default = (sections.index(default_name) + 1) if default_name else 1
    return opts, sections, default_name


def _report_files(ini) -> None:
    ok, missing = 0, []
    for name in ini.names():
        if name == "*":
            continue
        for key in ("model", "mmproj"):
            p = ini.section(name).pairs().get(key)
            if not p:
                continue
            if Path(p).exists():
                ok += 1
            else:
                missing.append(f"{name} ← {key}")
    head("проверка файлов", "пути из models.ini против диска")
    note(f"на месте: {ok};  недостаёт: {len(missing)}")
    for m in missing[:6]:
        warn(f"нет: {m}")
    if len(missing) > 6:
        warn(f"…ещё {len(missing) - 6}")
    note(f"GGUF от IsValorum (abliterated, твои Qwen3.8/Tiel уже оттуда):")
    note(f"  {ISVALORUM}")


def _do_router(action: str, build: str | None = None) -> int:
    if action == "start":
        return _run_cli("runtime", "start", "--build", build)
    if action == "restart":
        return _run_cli("runtime", "restart", "--build", build)
    if action == "stop":
        return _run_cli("runtime", "stop")
    return 1


def _do_preset(ini, build: str, swap_url: str) -> int:
    """Загрузить/сменить пресет. Под swap — через API, иначе через CLI."""
    opts, sections, default_name = _preset_options(ini)
    if not opts:
        warn(f"в {ini.path} нет ни одного пресета")
        return 1
    name = ask_pick(f"Вопрос {qnum()} · какой пресет грузим", opts,
                    sections.index(default_name) + 1 if default_name else 1,
                    sub="★ = есть живой замер VRAM или скорости — "
                        "ему верь, а не оценке")
    _run_cli("validate", name, "--build", build)
    head("что дальше с этим пресетом")
    options = [("Загрузить сейчас", "займёт VRAM и время"),
               ("Только проверить бюджет VRAM", "ничего не грузить"),
               ("Загрузить и снять факт VRAM", "load + measure")]
    choice = ask_pick(f"Вопрос {qnum()} · глубина проверки", options, 2)
    if choice == "Загрузить сейчас":
        return _run_cli("load", name, "--build", build)
    if choice == "Загрузить и снять факт VRAM":
        rc = _run_cli("load", name, "--build", build)
        _run_cli("measure", "--swap-url", swap_url, name)
        return rc
    return _run_cli("budget", name, "--explain")


def _do_swap(action: str, swap_url: str) -> int:
    if action == "stop":
        res = swap.down()
        note(res["message"])
        return 0 if res["ok"] else 1
    if action == "start":
        build = _ask_build()
        if build == BACK:
            return "menu"
        _run_cli("swap", "export", "--build", build, "-o",
                 str(swap.default_output()))
        # порт не передаём: swap сам берёт свободный, если дефолтный занят
        res = swap.up(auto_port=True)
        note(res["message"])
        if res["ok"]:
            url = res.get("url") or swap_url
            note("")
            note("дальше:")
            for s in swap.next_steps(url):
                note(s)
            head("быстрая проверка", "первый запрос грузит модель — это минуты")
            note(f"curl -s -m 900 -X POST {url}/v1/chat/completions \\")
            note("  -H 'Content-Type: application/json' \\")
            note("  -d '{\"model\":\"qwen3.8-35B-A3B-miniplus-128ctx-ngram-mmproj\",")
            note("       \"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}],\"max_tokens\":32}'")
        return 0 if res["ok"] else 1
    return 1


def run() -> int:
    if not sys.stdin.isatty():
        print("мастер интерактивный — запусти в терминале: llamastery wizard")
        return 2
    print("╔" + "═" * 62 + "╗")
    print("║  llamastery · мастер".ljust(63) + "║")
    print("╚" + "═" * 62 + "╝")
    print("Проверка пресетов, честный прогноз VRAM, замер факта.")
    print("Замер важнее расчёта. Ничего не гружу и не правлю без твоего «да».")

    swap_url = swap.load_state().get("url") or f"http://{swap.SWAP_LISTEN}"
    # состояние печатает _main_menu: раньше run() рисовал его сам, а меню
    # рисовало ещё раз — на старте экран удваивался
    while True:
        reset_questions()      # новый проход по меню — нумерация с нуля
        rc, action = _main_menu(swap_url)
        if rc != 0 or action is None:
            return rc
        if action == "Только показать состояние":
            print()
            note("ничего не менял")
            return 0

        rc = _dispatch(action, swap_url)
        if rc == "menu":             # 0 дошёл до начала — снова меню
            continue
        return rc


def _main_menu(swap_url: str) -> tuple[int, str | None]:
    """Главное меню. Возвращает (код, действие); действие None — 0, выход."""
    s = _state_lines(swap_url)
    print_state(s, swap_url)
    router_up = bool(s["router"]["up"])
    loaded = [m["id"] for m in s["router"].get("models", [])
              if m["status"] in ("loaded", "sleeping")]
    swap_up = bool(s["swap"].get("up"))
    options = menu_for(router_up, loaded, swap_up)
    labels = [o[0] for o in options]
    idx = ask(f"Вопрос {qnum()} · что делаем?", options,
              menu_default(options, router_up),
              sub="меню построено по состоянию выше; Enter — рекомендованный пункт, "
                  "0 — выход")
    if idx == BACK:
        print()
        note("вышел из мастера")
        return 0, None
    action = labels[idx]
    print(f"    {ARROW} ответ: {action}")
    return 0, action


def _ask_build() -> str:
    """Вопрос о сборке. BACK, если человек ушёл назад."""
    reg = _pick_build_registry()
    if not reg:
        raise SystemExit(1)
    return _pick_build(reg)


def _dispatch(action: str, swap_url: str):
    """Что делать после выбора в меню. 'menu' = вернуться к меню.

    Вопрос о сборке задаётся только там, где он реально нужен, и внутри
    потока — ровно один раз. Раньше он спрашивался и в меню, и первым шагом
    потока, то есть человек отвечал на него дважды подряд.
    """
    if action == "Настроить заново":
        return _setup_flow(swap_url)
    if action == "Остановить роутер":
        return _do_router("stop")
    if action == "Загрузить или сменить пресет":
        return _preset_flow(swap_url)
    if action.startswith("Запустить llama-swap"):
        return _do_swap("start", swap_url)
    if action == "Остановить llama-swap":
        return _do_swap("stop", swap_url)
    if action == "Перезапустить роутер":
        build = _ask_build()
        if build == BACK:
            return "menu"
        return _do_router("restart", build)
    if action.startswith("Запустить роутер"):
        build = _ask_build()
        if build == BACK:
            return "menu"
        _do_router("start", build)
        if action == "Запустить роутер и загрузить пресет":
            return _preset_after_router(build, swap_url)
        return 0
    return 0


def _preset_after_router(build: str, swap_url: str):
    """Роутер уже поднят выбранной сборкой — вопрос о ней не повторяем."""
    return _preset_flow(swap_url, preset_build=build)


def _preset_flow(swap_url: str, preset_build: str | None = None):
    """Пресет: выбрать сборку → показать файлы → выбрать пресет → глубину.

    0 = на шаг назад; на первом экране 0 возвращает в главное меню.
    """
    from .inifile import IniFile
    try:
        ini = IniFile.load(str(paths.default_ini()))
    except OSError as exc:
        warn(f"не читается {paths.default_ini()}: {exc}")
        return 1
    _report_files(ini)
    nav = Nav()

    def pick_build(a: dict):
        if preset_build:
            note(f"сборка: {preset_build}")
            return preset_build
        return _ask_build()

    def pick_preset(a: dict):
        opts, sections, default_name = _preset_options(ini)
        if not opts:
            warn(f"в {ini.path} нет ни одного пресета")
            return None
        return ask_pick(f"Вопрос {qnum()} · какой пресет грузим", opts,
                        sections.index(default_name) + 1 if default_name else 1,
                        sub="★ = есть живой замер VRAM или скорости — "
                            "ему верь, а не оценке")

    def pick_depth(a: dict):
        return ask_pick(f"Вопрос {qnum()} · глубина проверки",
                        [("Загрузить сейчас", "займёт VRAM и время"),
                         ("Только проверить бюджет VRAM", "ничего не грузить"),
                         ("Загрузить и снять факт VRAM", "load + measure")], 2)

    res = nav.walk([
        ("build", pick_build),
        ("preset", pick_preset),
        ("depth", pick_depth),
    ])
    if res is None:
        return "menu"
    if res.get("build") == BACK:
        return "menu"
    name = res.get("preset")
    if not name:
        return 1
    _run_cli("validate", name, "--build", res["build"])
    depth = res.get("depth") or "Только проверить бюджет VRAM"
    if depth == "Загрузить сейчас":
        return _run_cli("load", name, "--build", res["build"])
    if depth == "Загрузить и снять факт VRAM":
        rc = _run_cli("load", name, "--build", res["build"])
        _run_cli("measure", "--swap-url", swap_url, name)
        return rc
    return _run_cli("budget", name, "--explain")


def _setup_flow(swap_url: str):
    """Полный маршрут: сборка → swap → пресет → тюн → контекст → зрение →
    ускорители → план. На каждом экране 0 = на шаг назад."""
    from .inifile import IniFile
    try:
        ini = IniFile.load(str(paths.default_ini()))
    except OSError as exc:
        warn(f"не читается {paths.default_ini()}: {exc}")
        return 1

    nav = Nav()

    def s_build(a: dict):
        return _ask_build()

    def s_swap(a: dict):
        if swap.find_binary():
            note("llama-swap уже установлен")
            return "установлен"
        if ask_yn(f"Вопрос {qnum()} · установить llama-swap?", True,
                  sub="прокси для хот-свапа: один порт, модели меняются полем model"):
            _run_cli("swap", "install")
            return "установлен"
        return "пропущен"

    def s_files(a: dict):
        _report_files(ini)
        return "проверено"

    def s_preset(a: dict):
        opts, sections, default_name = _preset_options(ini)
        if not opts:
            return None
        return ask_pick(f"Вопрос {qnum()} · какой пресет берём за основу", opts,
                        sections.index(default_name) + 1 if default_name else 1)

    def s_tune(a: dict):
        preset = a.get("preset")
        if not preset:
            return None
        pairs = ini.section(preset).pairs()
        if ask_yn(f"Вопрос {qnum()} · прогнать автотюн этого пресета?", False,
                  sub="долго (10–30 мин) и грузит GPU на 100% — "
                      "запускай в свободное время"):
            tune = (f"tune {paths.default_ini()} {preset} "
                    f"--build {a.get('build')} "
                    f"--extra c={pairs.get('c', 32768)} "
                    f"n-cpu-moe={pairs.get('n-cpu-moe', 0)} "
                    f"ubatch=1024 reserve=1024 min-tps=25")
            print()
            print(f"  {ARROW} команда: llamastery {tune}")
            return "показан"
        return "не просил"

    def s_ctx(a: dict):
        preset = a.get("preset")
        cur = ini.section(preset).pairs().get("c", "?") if preset else "?"
        return ask_text(f"Вопрос {qnum()} · контекст", str(cur),
                        sub="больше контекст = больше KV = больше VRAM")

    def s_vision(a: dict):
        preset = a.get("preset")
        if not preset:
            return None
        low = {k.lower() for k in ini.section(preset).pairs()}
        sections = [n for n in ini.names() if n != "*"]
        sibling = find_mmproj_sibling(sections, preset)
        if "mmproj" in low:
            note("зрение: в пресете уже есть mmproj")
            return "включено"
        if sibling:
            if ask_yn(f"Вопрос {qnum()} · взять близнец с mmproj ({sibling})?",
                      False, sub="в нём включено зрение; текстовый быстрее"):
                return sibling
            return "текстовый"
        ask_yn(f"Вопрос {qnum()} · нужен анализ картинок?", False,
               sub="понадобится mmproj-файл модели")
        return "текстовый"

    def s_ngram(a: dict):
        preset = a.get("preset") or ""
        pairs = ini.section(preset).pairs() if preset else {}
        return ask_yn(f"Вопрос {qnum()} · тестить с ngram (спекулятивный декодер)?",
                      ngram_default(pairs),
                      sub="×3–4 на повторяющемся тексте, но медленнее на уникальном")

    def s_mtp(a: dict):
        preset = a.get("preset") or ""
        pairs = ini.section(preset).pairs() if preset else {}
        return ask_yn(f"Вопрос {qnum()} · тестить с MTP (черновая голова)?",
                      mtp_default(preset, pairs))

    res = nav.walk([
        ("build", s_build),
        ("swap", s_swap),
        ("files", s_files),
        ("preset", s_preset),
        ("tune", s_tune),
        ("ctx", s_ctx),
        ("vision", s_vision),
        ("ngram", s_ngram),
        ("mtp", s_mtp),
    ])
    if res is None or res.get("build") == BACK:
        return "menu"

    preset = res.get("preset")
    if not preset:
        warn("без пресета план строить не на чем")
        return 1
    # близнец с mmproj мог поменять пресет
    if res.get("vision") and res["vision"] not in ("включено", "текстовый"):
        preset = res["vision"]
    ctx = res.get("ctx")
    pairs = ini.section(preset).pairs()

    head("план", "выполняется по порядку, каждый шаг можно пропустить")
    steps = [f"validate {preset} --build {res['build']}",
             f"budget {preset} --explain",
             f"load {preset} --build {res['build']}",
             f"measure --swap-url {swap_url} {preset}",
             f"probe --tokens 110000 --from-file <реальный-код> --record "
             f"--preset {preset}",
             f"swap export --build {res['build']} -o {swap.default_output()}"]
    for i, st in enumerate(steps, 1):
        print(f"    {i}. llamastery {st}")
    if res.get("ngram") or res.get("mtp"):
        note(f"ускорители в тесте: ngram="
             f"{'да' if res.get('ngram') else 'нет'}, "
             f"mtp={'да' if res.get('mtp') else 'нет'} "
             f"(флаги — в пресет перед load)")
    if ctx and ctx != str(pairs.get("c", "")):
        warn(f"контекст {pairs.get('c')} → {ctx}: сначала правка models.ini")
    print()
    if ask_yn("выполнить первые два шага (validate + budget, безопасно)?", True):
        _run_cli("validate", preset, "--build", res["build"])
        _run_cli("budget", preset, "--explain")
    print()
    note(f"готово. загрузка вручную: "
         f"llamastery load {preset} --build {res['build']}")
    return 0
