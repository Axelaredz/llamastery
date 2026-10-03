"""Интерактивный мастер: голый запуск llamastery ведёт за руку.

Зачем: флагов десятки, а эффективный маршрут один — мастер показывает его
сразу и везде предлагает дефолт ★ (Enter = самый эффективный вариант).

Маршрут: описание → сборка (дефолт faks) → swap → модели (+IsValorum) →
пресет → контекст → картинки → ngram/mtp → план с согласия.

Правила мастера:
  * чтение — молча и само (реестр, файлы моделей, замеры);
  * запись/загрузка/сеть — только после явного «да»;
  * сам мастер ничего не грузит в VRAM и не правит models.ini;
  * запуск шагов — через собственный CLI подпроцессом, чтобы не дублировать
    логику команд.
"""

import subprocess
import sys
from pathlib import Path

from . import builds, paths, swap

STAR = "★"
ISVALORUM = "https://huggingface.co/collections/IsValorum/abliterated"


def ask(prompt: str, default: str = "") -> str:
    """Вопрос с дефолтом. Пустой ввод = дефолт."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return raw or default


def ask_yn(prompt: str, default_yes: bool) -> bool:
    """Да/нет. Дефолт помечен ★ и выбирается Enter."""
    hint = "Y/n" if default_yes else "y/N"
    try:
        raw = input(f"{prompt} ({hint}) {STAR}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    if not raw:
        return default_yes
    return raw in ("y", "yes", "д", "да")


def clamp_choice(raw: str, n: int, default: int) -> int:
    """Номер пункта 1..n, мусор = дефолт."""
    try:
        v = int(raw)
        if 1 <= v <= n:
            return v
    except (ValueError, TypeError):
        pass
    return default


# ── чистые дефолты (покрыты тестами) ──
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


def _bin() -> str:
    return str(Path(__file__).resolve().parent.parent / "bin" / "llamastery")


def _run_cli(*args: str) -> int:
    print(f"  → выполняю: llamastery {' '.join(args)}")
    try:
        return subprocess.call([_bin(), *args])
    except OSError as exc:
        print(f"  ! не запустилось: {exc}")
        return 1


def run() -> int:
    if not sys.stdin.isatty():
        print("мастер интерактивный — запусти в терминале: llamastery wizard")
        return 2
    print("=== llamastery: мастер ===")
    print("Проверка пресетов, честный прогноз VRAM, замер факта.")
    print("Замер важнее расчёта. Ничего не гружу и не правлю без твоего «да».\n")

    # ── 1. сборка ──
    print("── 1/7 сборка ──")
    reg = builds.all_builds()
    if not reg:
        print("реестр пуст.")
        if ask_yn("Найти сборки автоматически (builds detect --apply)?", True):
            _run_cli("builds", "detect", "--apply")
            reg = builds.all_builds()
        if not reg:
            print("сборок нет — дальше некуда. Собери llama-server и повтори.")
            return 1
    names = list(reg)
    for i, name in enumerate(names, 1):
        st = reg[name].status()
        mark = f" {STAR}" if name == recommend_build(reg) else ""
        print(f"  {i}. {name:<10} "
              f"{'роутер' if st['router'] else 'single':<7} "
              f"{'собран' if st['bin_exists'] else 'НЕ собран'}"
              f"{(' ' + str(st.get('version') or '')) if st.get('version') else ''}"
              f"{mark}")
    print("Рекомендация: faks — самый прогрессивный (роутер, n-cpu-moe, патчи).")
    rec = recommend_build(reg)
    default_n = (names.index(rec) + 1) if rec in names else 1
    build = names[clamp_choice(ask("Сборка номером", str(default_n)),
                               len(names), default_n) - 1]
    print(f"выбрана: {build}\n")

    # ── 2. swap ──
    print("── 2/7 llama-swap ──")
    print("Прокси для хот-свапа моделей на одном порту: поднял один :8087,")
    print("а модели переключаются сами по полю \"model\" в запросе.")
    if swap.find_binary():
        print("swap уже стоит.")
    elif ask_yn("Установить бинарь swap v261 в ~/.local/bin?", True):
        _run_cli("swap", "install")
    print()

    # ── 3. модели ──
    print("── 3/7 модели ──")
    from .inifile import IniFile
    try:
        ini = IniFile.load(str(paths.default_ini()))
    except OSError as exc:
        print(f"не читается {paths.default_ini()}: {exc}")
        return 1
    sections = [n for n in ini.names() if n != "*"]
    ok, missing = [], []
    for name in sections:
        pairs = ini.section(name).pairs()
        for key in ("model", "mmproj"):
            p = pairs.get(key) or pairs.get("m")
            if not p:
                continue
            (ok if Path(p).exists() else missing).append(
                f"{name} ← {key}: {p}")
    print(f"файлов на месте: {len(ok)}; недостаёт: {len(missing)}")
    for m in missing[:6]:
        print(f"  ! нет: {m}")
    if len(missing) > 6:
        print(f"  …ещё {len(missing) - 6}")
    print(f"Рекомендованный автор GGUF (abliterated, твои Qwen3.8/Tiel уже оттуда):\n  {ISVALORUM}\n")

    # ── 4. пресет ──
    print("── 4/7 пресет ──")
    try:
        from . import measure as _m
        records = _m.load_store().get("records", {})
    except Exception:  # noqa: BLE001 — замеры опциональны
        records = {}
    measured = {n for n in sections
                if any(n.lower() in str(k).lower() for k in records)}
    for i, name in enumerate(sections, 1):
        pairs = ini.section(name).pairs()
        low = {k.lower(): v for k, v in pairs.items()}
        model = (pairs.get("model") or "?").rsplit("/", 1)[-1][:44]
        mark = f" {STAR}" if name in measured else ""
        print(f"  {i}. {name} | c={low.get('c', '?')} "
              f"| moe={low.get('n-cpu-moe', low.get('cpu-moe', '-'))} "
              f"| {'mmproj' if 'mmproj' in low else 'текст'}{mark}")
    if measured:
        print("★ = есть живой замер (точнее любой оценки).")
    default_p = (sections.index(sorted(measured)[0]) + 1) if measured else 1
    preset = sections[clamp_choice(ask("Пресет номером", str(default_p)),
                                   len(sections), default_p) - 1]
    pairs = ini.section(preset).pairs()
    print(f"выбран: {preset}")
    if ask_yn("Провести автотюн этого пресета (долго, грузит GPU)?", False):
        tune = (f"tune {paths.default_ini()} {preset} --build {build} "
                f"--extra c={pairs.get('c', 32768)} "
                f"n-cpu-moe={pairs.get('n-cpu-moe', 0)} "
                f"ubatch=1024 reserve=1024 min-tps=25")
        print(f"  запуск: llamastery {tune}")
        print("  (запусти сам, когда будет 10–30 свободных минут)")
    print()

    # ── 5. контекст ──
    print("── 5/7 контекст ──")
    cur_ctx = pairs.get("c", "?")
    print(f"сейчас: c={cur_ctx}. Больше контекст = больше KV = больше VRAM.")
    ctx = ask("Контекст (Enter — оставить, или число)", str(cur_ctx))
    print(f"контекст: {ctx}\n")

    # ── 6. картинки ──
    print("── 6/7 картинки ──")
    low = {k.lower(): v for k, v in pairs.items()}
    sibling = find_mmproj_sibling(sections, preset)
    if "mmproj" in low:
        print("в пресете уже есть mmproj — зрение включено.")
    elif sibling:
        print(f"есть близнец с mmproj: {sibling}")
        if ask_yn("Взять его вместо текстового?", False):
            preset = sibling
            pairs = ini.section(preset).pairs()
    else:
        print("mmproj-секции для этой модели нет.")
        ask_yn("Нужен анализ картинок (понадобится mmproj-файл)?", False)
    print()

    # ── 7. ускорители ──
    print("── 7/7 ускорители ──")
    use_ngram = ask_yn("Тестить с ngram (спекулятивный декодер)?",
                       ngram_default(pairs))
    if use_ngram and "mmproj" in {k.lower() for k in pairs}:
        print("  ! ngram-mod несовместим с mmproj — при тесте будет отключён.")
    use_mtp = ask_yn("Тестить с MTP (черновая голова)?",
                     mtp_default(preset, pairs))
    print()

    # ── план ──
    print("── план ──")
    steps = [f"validate {preset} --build {build}",
             f"budget {preset} --explain",
             f"load {preset} --build {build}",
             "measure",
             f"probe --tokens 110000 --from-file <реальный-код> --record --preset " + preset]
    for i, s in enumerate(steps, 1):
        print(f"  {i}. llamastery {s}")
    print(f"  6. swap export --build {build} -o ~/.config/llama-swap/config.yaml")
    if use_ngram or use_mtp:
        print(f"  ускорители в тесте: ngram={'да' if use_ngram else 'нет'}, "
              f"mtp={'да' if use_mtp else 'нет'} (флаги — в пресет перед load)")
    if ctx != str(pairs.get("c", "")):
        print(f"  ! контекст {pairs.get('c')} → {ctx}: сначала правка models.ini")
    print()
    if ask_yn("Выполнить шаги 1–2 (validate+budget, безопасно)?", True):
        _run_cli("validate", preset, "--build", build)
        _run_cli("budget", preset, "--explain")
    print("\nГотово. Загрузка — только вручную: "
          f"llamastery load {preset} --build {build}")
    return 0
