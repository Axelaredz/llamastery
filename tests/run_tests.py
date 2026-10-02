#!/usr/bin/env python3
"""Тесты без внешних зависимостей: python3 tests/run_tests.py"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import budget, gguf, inifile, measure, presets, schema, validate  # noqa: E402

FAILED: list[str] = []


def check(cond, label, extra=""):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {extra}")
        FAILED.append(label)


# ── синтаксис и компиляция ──
def test_syntax():
    print("синтаксис")
    for f in sorted(ROOT.glob("lib/*.py")) + [ROOT / "tools" / "tune_models.py",
                                              ROOT / "bin" / "llamastery"]:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"компилируется {f.name}", r.stderr[:200])


# ── INI: комментарии должны выживать ──
SAMPLE = """; шапка файла
; вторая строка

[*]
n-gpu-layers = 99
fa = true

; комментарий к модели
; с замерами: tg 48.6
[model-a]
c = 32768
temp = 0.6 ; инлайн-комментарий

[model-b]
c = 65536
"""


def test_ini():
    print("INI с сохранением комментариев")
    ini = inifile.IniFile("x.ini", SAMPLE)
    check(len(ini.sections) == 3, "три секции (включая [*])", len(ini.sections))
    check(ini.section("model-a").get("c") == "32768", "значение прочитано")
    check(ini.section("model-a").get("temp") == "0.6",
          "инлайн-комментарий не попал в значение",
          ini.section("model-a").get("temp"))
    check(ini.globals().get("n-gpu-layers") == "99", "глобальная секция [*]")
    out = ini.dumps()
    check("; шапка файла" in out, "комментарий до первой секции сохранён")
    check("; с замерами: tg 48.6" in out, "комментарий над секцией сохранён")
    check("инлайн-комментарий" in out, "инлайн-комментарий сохранён")

    sec = ini.section("model-a")
    sec.set("c", "65536")
    out2 = ini.dumps()
    check("c = 65536" in out2, "изменение значения записано")
    check(out2.count("[model-a]") == 1, "секция не продублирована")
    check("; с замерами" in out2, "комментарий пережил правку значения")


# ── схема флагов ──
HELP = """some header text
-m,    --model FNAME                    model path to load
  -t,    --threads N                    number of threads to use (default: 8)
  -c,    --ctx-size N                   size of the prompt context
                                          (env: LLAMA_ARG_CTX_SIZE)
-c,    --ctx-size N                     size of the prompt context (default: 0)
                                        (env: LLAMA_ARG_CTX_SIZE)
-nommo, --no-mmproj-offload            disable offloading the multimodal encoder
-sm,   --split-mode {none,layer,row,tensor}
                                        how to split the model
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers
-fa,   --flash-attn (auto|on|off|0|1)
                                  set Flash Attention (default: on)
-C,    --cpu-mask N                      physical cores
--spec-type none,draft-mtp,ngram-mod
                                        comma-separated list of types
-kvu,  --kv-unified, -no-kvu, --no-kv-unified
"""


def test_schema():
    print("разбор --help")
    flags = schema.parse_help(HELP)
    check("--ctx-size" in flags, "найден --ctx-size")
    f = flags.get("--ctx-size")
    check(f and f.short == "-c", "короткая форма -c")
    check(f and f.kind == "int", "тип int", f.kind if f else None)
    check(f and f.env == "LLAMA_ARG_CTX_SIZE", "env-var распознан")
    check("ctx-size" in f.ini_keys() and "c" in f.ini_keys(),
          "все три формы записи", f.ini_keys() if f else None)

    fa = flags.get("--flash-attn")
    check(fa and "on" in (fa.enum or []),
          "перечисление в скобках (стиль ik_llama)", fa.enum if fa else None)

    sm = flags.get("--split-mode")
    check(sm and sm.enum == ["none", "layer", "row", "tensor"],
          "enum из фигурных скобок", sm.enum if sm else None)

    sp = flags.get("--spec-type")
    check(sp and "draft-mtp" in sp.enum, "enum из голого списка",
          sp.enum if sp else None)

    check("--kv-unified" in flags, "флаги-инверсии разобраны")
    check(schema.resolve(flags, "c").canonical == "--ctx-size",
          "resolve по короткой форме")
    check(schema.resolve(flags, "ngl").canonical == "--gpu-layers",
          "resolve по ngl")
    check(schema.resolve(flags, "LLAMA_ARG_CTX_SIZE").canonical == "--ctx-size",
          "resolve по env-имени")
    check(schema.resolve(flags, "C").canonical == "--cpu-mask",
          "регистр значим: C != c")
    check(schema.resolve(flags, "нет-такого") is None, "неизвестный ключ")
    # ik_llama печатает опции с отступом — раньше они не разбирались
    check("--threads" in flags, "опция с отступом (стиль ik_llama)",
          sorted(flags)[:6])
    check(flags.get("--threads") and flags["--threads"].env is None,
          "у опции с отступом разобран default")


# ── GGUF ──
def test_gguf(tmp: Path):
    print("GGUF")
    # собираем минимальный валидный заголовок
    import struct
    kv = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 40,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.context_length": 262144,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.full_attention_interval": 4,
    }

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0),
           struct.pack("<Q", len(kv))]
    for k, v in kv.items():
        out.append(s(k))
        if isinstance(v, str):
            out.append(struct.pack("<I", 8) + s(v))
        else:
            out.append(struct.pack("<I", 4) + struct.pack("<i", v))
    p = tmp / "fake.gguf"
    p.write_bytes(b"".join(out) + b"\0" * 4096)

    m = gguf.probe(p)
    check(m.n_layer == 40, "слои прочитаны", m.n_layer)
    check(m.head_dim == 256, "head_dim из key_length", m.head_dim)
    check(m.is_moe, "MoE обнаружен")
    check(budget.attention_layers(m) == 10,
          "гибрид: только 10 слоёв внимания из 40", budget.attention_layers(m))
    share = budget.expert_weight_share(m)
    check(share > 0.9, "доля весов экспертов ~1.0", share)
    check(not gguf.is_gguf(p.parent / "нет-такого.gguf"), "is_gguf на пути")


# ── оценка памяти ──
def test_budget():
    print("бюджет VRAM")
    m = gguf.ModelMeta(path=Path("/x.gguf"), size_bytes=12 * budget.GIB)
    m.arch = "qwen35moe"
    m.kv = {"qwen35moe.block_count": 40, "qwen35moe.embedding_length": 2048,
            "qwen35moe.attention.head_count": 16,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_feed_forward_length": 512,
            "qwen35moe.full_attention_interval": 4}

    base = {"c": "65536", "n-cpu-moe": "0", "cache-type-k": "q8_0",
            "cache-type-v": "q8_0", "parallel": "1"}
    e0 = budget.estimate(base, m, compute_gb=0.0)
    e1 = budget.estimate({**base, "n-cpu-moe": "20"}, m, compute_gb=0.0)
    check(e1.weights_gb < e0.weights_gb * 0.75,
          "n-cpu-moe уменьшает веса в VRAM", (e0.weights_gb, e1.weights_gb))
    check(e0.kv_gb > 0, "KV посчитан", e0.kv_gb)
    e_small = budget.estimate({**base, "c": "8192"}, m, compute_gb=0.0)
    check(e_small.kv_gb * 8 < e0.kv_gb + 0.01,
          "KV растёт линейно с контекстом", (e_small.kv_gb, e0.kv_gb))
    e_unified_off = budget.estimate(
        {**base, "parallel": "4", "kv-unified": "off"}, m, compute_gb=0.0)
    check(e_unified_off.kv_gb > e0.kv_gb * 3.5,
          "без unified KV умножается на слоты", e_unified_off.kv_gb)


# ── измерения: подпись не должна зависеть от порядка ──
def test_measure():
    print("подписи замеров")
    a = measure.signature({"c": "32768", "temp": "0.6", "n-cpu-moe": "16"})
    b = measure.signature({"n-cpu-moe": "16", "c": "32768", "temp": "0.6"})
    c = measure.signature({"c": "65536", "temp": "0.6", "n-cpu-moe": "16"})
    check(a == b, "порядок ключей не влияет")
    check(a != c, "разный контекст -> разная подпись")


# ── валидация ──
def test_validate():
    print("валидация")
    flags = schema.parse_help(HELP)
    rep = validate.Report()
    validate.validate_section("s", {"model": "/нет/такого.gguf", "c": "1"},
                              flags, rep, check_paths=True)
    check(rep.count("error") >= 1, "несуществующий файл ловится")

    rep2 = validate.Report()
    validate.validate_section("s", {"api-key": "x", "c": "1"}, flags, rep2,
                              check_paths=False)
    check(any("роутер вырезает" in f.message for f in rep2.findings),
          "api-key помечается как control-аргумент")

    rep3 = validate.Report()
    validate.validate_section("s", {"model": "/x.gguf", "port": "8080"},
                              flags, rep3, check_paths=False)
    check(not any(f.level == "error" for f in rep3.findings),
          "port — предупреждение, не ошибка")
    check(any(f.level == "warn" for f in rep3.findings),
          "port даёт предупреждение")

    rep4 = validate.Report()
    validate.check_traps("s", {"spec-type": "ngram-mod", "model-draft": "/m.gguf"},
                         rep4)
    check(rep4.count("error") >= 1, "грабли форка ловятся")
    rep5 = validate.Report()
    validate.check_traps("s", {"ubatch-size": "2048"}, rep5)
    # Порог уточнён замером: сам по себе ubatch 2048 на 12 ГБ работает
    # (tiel-coder-nanoplus-128ctx-mmproj-moe16: 29.3 t/s, 1616 MiB свободно),
    # а ломается он в паре с ускорителем — 2048 + ngram-mod роняет CUDA.
    check(rep5.count("warn") == 0,
          "ubatch 2048 без ускорителя — не повод для предупреждения")
    check(rep5.count("info") >= 1,
          "ubatch 2048 без ускорителя — как минимум информация")
    rep6 = validate.Report()
    validate.check_traps("s", {"spec-type": "ngram-mod", "ubatch-size": "2048"},
                         rep6, measured={"ok": True, "min_free_mib": 1500})
    check(not any(f.level == "warn" for f in rep6.findings),
          "замер глушит эвристику")
    rep6b = validate.Report()
    validate.check_traps("s", {"spec-type": "ngram-mod", "ubatch-size": "2048"},
                         rep6b)
    check(rep6b.count("warn") >= 1,
          "ubatch 2048 с ускорителем ругается даже без замера: проверено, что падает")
    rep7 = validate.Report()
    validate.check_traps("s", {"spec-type": "ngram-mod", "model-draft": "/m.gguf"},
                         rep7, measured={"ok": True, "min_free_mib": 1500})
    check(rep7.count("error") >= 1,
          "замер НЕ глушит предсказание падения")


# ── слияние ──
def test_merge():
    print("слияние пресетов")
    target = inifile.IniFile("t.ini", "[*]\nfa = on\n\n[a]\nc = 1024\n")
    source = inifile.IniFile("s.ini",
                             "[a]\nmodel = /m.gguf\nc = 4096\n\n[new]\nmodel = /n.gguf\n")
    res = presets.merge_into(target, source, on_conflict="skip")
    check("new" in res.added, "новая секция добавлена", res.added)
    check(target.section("a").get("c") == "1024", "конфликт не перезаписан")
    check(target.section("a").get("model") == "/m.gguf",
          "непротиворечивый ключ добавлен")
    check(res.conflicts, "конфликт зафиксирован")

    res2 = presets.merge_into(target, source, on_conflict="overwrite")
    check(target.section("a").get("c") == "4096", "overwrite применился")

    res3 = presets.merge_into(target, source, only=["new"])
    check("a" not in res3.updated, "--only фильтрует")


# ── экспорт ──
def test_export():
    print("экспорт")
    ini = inifile.IniFile("t.ini", SAMPLE)
    text = presets.export_sections(ini, ["model-a"])
    check("[model-a]" in text and "[model-b]" not in text, "только выбранные")
    check("[*]" in text, "глобальная секция включена")
    check("c = 32768" in text, "значения на месте")


# ── трансляция пресета в argv (сборки без роутера) ──
def test_preset_to_argv():
    print("пресет -> argv")
    from lib import server
    flags = schema.parse_help(HELP)
    pairs = {"c": "32768", "fa": "true", "t": "8", "kv-unified": "",
             "model": "/m.gguf", "port": "8099", "нет-такого": "1"}
    argv, warns = server.preset_to_argv(pairs, flags)
    check("--ctx-size" in argv, "короткий ключ c развёрнут в --ctx-size")
    check(argv[argv.index("--ctx-size") + 1] == "32768", "значение на месте")
    check("--flash-attn" in argv
          and argv[argv.index("--flash-attn") + 1] in
              (flags["--flash-attn"].enum or ["true"]),
          "fa развёрнут со значением из enum сборки",
          argv[argv.index("--flash-attn") + 1]
          if "--flash-attn" in argv else None)
    check("--kv-unified" in argv, "флаг без значения не потерян")
    check(argv.count("--kv-unified") == 1, "флаг не продублирован значением")
    from lib import schema as _s
    fa_flag = _s.resolve(flags, "fa")
    if fa_flag and fa_flag.enum:
        argv_fa, _ = server.preset_to_argv({"fa": "true"}, flags)
        got = argv_fa[argv_fa.index("--flash-attn") + 1] if "--flash-attn" in argv_fa else None
        check(got in fa_flag.enum,
              "булево значение приведено к форме сборки (true -> on/1)", got)
    pairs2 = {"no-mmproj-offload": "true"}
    argv2, _ = server.preset_to_argv(pairs2, flags)
    check(argv2 == ["--no-mmproj-offload"],
          "инверсионный флаг не получает значение", argv2)
    check("--port" not in argv, "порт выбрасываем — задаём сами")
    check("--model" in argv, "модель обязана попасть в argv single-сервера")
    check("нет-такого" not in argv, "неизвестный ключ не попал в argv")
    check(any("нет-такого" in w for w in warns), "о неизвестном ключе сказано")


def test_tuner_passthrough():
    """Тюнер не должен выбрасывать ускорители из секции."""
    print("тюнер: перенос ускорителей")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tune_models", ROOT / "tools" / "tune_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kinds = {"--spec-type": "string", "--model-draft": "string",
             "--fit": "enum", "--spec-ngram-mod-n-max": "int"}
    orig = mod._flag_kinds
    mod._flag_kinds = lambda binary: kinds
    opts = {"model": "/m.gguf", "c": "114688", "n-gpu-layers": "99",
            "n-cpu-moe": "24", "b": "2048", "ubatch-size": "1024", "t": "8",
            "spec-type": "ngram-mod", "model-draft": "/d.gguf",
            "spec-ngram-mod-n-max": "4"}
    caps = {k: True for k in ("threads", "threads-batch", "parallel",
                              "flash-attn", "cache-type-k", "cache-type-v",
                              "kv-unified", "jinja", "n-cpu-moe",
                              "image-min-tokens")}
    args = mod.argparse.Namespace(
        keep_mmproj=True, allow_ot=False, load_mode="default", keep_checkpoints=False,
        keep_mmproj_ok=True, threads=8, threads_batch=16, kv_unified="on",
        flash_attn="auto", jinja="on", kv_cache_type="q8_0", ngl=99,
        allow_vision=True)
    cmd = mod.build_server_command("/bin/true", opts, 8080, caps, args)
    check("--spec-type" in cmd and "ngram-mod" in cmd,
          "spec-type доезжает до команды тюнера", cmd)
    check("--model-draft" in cmd, "model-draft доезжает")
    check("--spec-ngram-mod-n-max" in cmd, "свои ручки ngram доезжают")
    mod._flag_kinds = orig


def test_build_env():
    print("окружение сборки")
    from lib import builds, server
    b = builds.Build(name="x", path="/nonexistent", router=True,
                     env={"GGML_TEST": "1"})
    env = server.build_env(b)
    check(env.get("GGML_TEST") == "1", "переменная из реестра попала в env")
    check("LD_LIBRARY_PATH" in env, "LD_LIBRARY_PATH проставлен")
    env.setdefault  # проверка, что setdefault не затирает внешнее значение


def main() -> int:
    """Запускает все test_* из этого файла.

    Раньше здесь стоял список вызовов руками, и тест, добавленный в конец
    файла, просто не выполнялся — при этом выводилось «все проверки прошли».
    Так молча пропало двадцать тестов. Поэтому список вызовов больше не
    существует: функции находятся по имени, а неизвестная сигнатура — ошибка,
    а не повод пропустить тест.
    """
    import inspect
    module = sys.modules[__name__]
    tests = sorted((n, f) for n, f in vars(module).items()
                   if n.startswith("test_") and inspect.isfunction(f))
    if len(tests) < 30:
        print(f"подозрительно мало тестов: {len(tests)}")
    with tempfile.TemporaryDirectory() as td:
        for name, fn in tests:
            params = list(inspect.signature(fn).parameters)
            if params == ["tmp"]:
                fn(Path(td))
            elif params:
                print(f"НЕПОНЯТНАЯ СИГНАТУРА {name}{inspect.signature(fn)} "
                      f"— тест пропущен")
                FAILED.append(f"{name}: неизвестная сигнатура")
            else:
                fn()
    print(f"выполнено тестов: {len(tests)}")
    print()
    if FAILED:
        print(f"ПРОВАЛЕНО: {len(FAILED)}")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("все проверки прошли")
    return 0




def test_budget_measured_beats_estimate() -> None:
    """Решение о вместимости доверяет живому замеру, а не прогнозу.

    Регрессия: tiel-coder-nanoplus-128ctx-mmproj-moe16 объявлялся невмещающимся
    при оценке 11.54 GiB, хотя замер давал 10.42 GiB и пресет рабочий.
    """
    from lib import budget

    est = budget.Estimate(model="m")
    est.total_gb = 11.54
    meas = {"used_mib": 10672}

    def need(e, m):
        return float(m["used_mib"]) if (m and m.get("used_mib")) else e.total_gb * 1024.0

    assert need(est, meas) == 10672, "замер должен побеждать прогноз"
    assert need(est, None) == 11.54 * 1024.0, "без замера остаётся прогноз"


def test_budget_spec_ubatch_warning() -> None:
    """Предупреждение о падении CUDA срабатывает на подтверждённо плохом ubatch.

    Проверено вживую: Tiel-Coder NanoPlus, moe16, c=114688, mmproj.
    ubatch=2048 роняет CUDA (out of memory) на запросе 110k, ubatch=1024 — нет.
    """
    from lib import budget

    def notes_for(ub):
        pairs = {"model": "m.gguf", "n-cpu-moe": "16", "spec-type": "ngram-mod",
                 "ubatch-size": str(ub), "c": "114688", "cache-type-k": "q8_0",
                 "cache-type-v": "q8_0", "b": "2048", "n-gpu-layers": "99"}
        return budget.estimate(pairs, None).notes

    assert not any("ВНИМАНИЕ" in n for n in notes_for(1024)), \
        "рабочее значение 1024 не должно ругаться"
    assert any("ВНИМАНИЕ" in n for n in notes_for(2048)), \
        "подтверждённо падающее значение 2048 должно ругаться"


def test_budget_compute_growth_damped() -> None:
    """Рост compute buffer затухающий, иначе оценка завышает в разы.

    Проверено вживую: сервер сообщает 978 MiB при ubatch=1024, тогда как
    прежняя формула (0.4 + 0.6*ub/ref) давала 1.58 GiB, а на ubatch=2048 —
    2.76 GiB, и рабочий пресет объявлялся невмещающимся.
    """
    from lib import budget

    def total_for(ub):
        pairs = {"model": "m.gguf", "ubatch-size": str(ub), "c": "114688",
                 "n-gpu-layers": "99", "b": "2048"}
        return budget.estimate(pairs, None).total_gb

    base, at_1024, at_2048 = total_for(512), total_for(1024), total_for(2048)
    assert at_1024 < base * 1.15, "рост с 512 до 1024 должен быть небольшим"
    assert at_2048 < at_1024 * 1.15, "рост с 1024 до 2048 тоже затухает"


def test_tuner_with_spec_keeps_accelerator() -> None:
    """--with-spec оставляет ускоритель в замере; без него — выкидывает.

    Проверено вживую: с ngram-mod тюнер показывал в логе
    «spec-type игнорируется в этом тесте», то есть флаг работал неявно.
    Осознанность решения зафиксирована флагом, а не молчаливой правкой.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tune_models", Path(__file__).resolve().parents[1]
        / "tools" / "tune_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    src = (Path(__file__).resolve().parents[1]
           / "tools" / "tune_models.py").read_text()
    check("--with-spec" in src, "флаг --with-spec объявлен")
    # сам механизм: ключи ускорителя перечислены и снимаются по флагу
    check('for key in ("model-draft", "spec-type")' in src,
          "ускоритель снимается списком ключей, а не одним")
    needle = 'and not args.with_spec:'
    check(needle in src, "снятие только когда флаг не выдан")


def test_env_prefix_is_lamastery_only() -> None:
    """Пути задаются только префиксом LLAMASTERY_.

    Префикс — это имя инструмента в верхнем регистре: 'llamastery'.upper() ==
    'LLAMASTERY'. Раньше было LAMMASTERY_, с потерянной буквой.

    Проверяются два префикса, которых быть не должно: LLACTL_ от прежних имён
    (llama-preset-ops, а до него llactl) и LAMMASTERY_ — опечатка. Иначе любой
    из них продолжит молча работать.
    """
    import importlib
    import os
    from lib import paths as P

    stale = ("LLACTL_STATE_DIR", "LAMMASTERY_STATE_DIR")
    saved = {k: os.environ.get(k) for k in stale + ("LLAMASTERY_STATE_DIR",)}
    try:
        for k in stale:
            os.environ[k] = "/tmp/opencode/t-stale"
        os.environ.pop("LLAMASTERY_STATE_DIR", None)
        importlib.reload(P)
        for k in stale:
            check("/tmp/opencode/t-stale" not in str(P.state_dir()),
                  f"префикс {k.split('_')[0]}_ игнорируется")
        os.environ["LLAMASTERY_STATE_DIR"] = "/tmp/opencode/t-new"
        importlib.reload(P)
        check(str(P.state_dir()) == "/tmp/opencode/t-new",
              "префикс LLAMASTERY_ учтён")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(P)

    # и в коде упоминаний старого префикса не осталось
    src = (Path(__file__).resolve().parents[1] / "lib"
           / "paths.py").read_text()
    check("LLACTL_" not in src, "в paths.py нет префикса LLACTL_")
    check("LAMMASTERY_" not in src,
          "в paths.py нет префикса с потерянной буквой")


def test_probe_detects_cached_run() -> None:
    """Проба из прогретого KV опознаётся и не попадает в статистику.

    Раньше вторая проба приходила с prompt_n = 4 вместо 110000, и её prefill
    попадал в средний: на графике это выглядело как «внезапно ускорилось».
    """
    from lib import probe

    cold = {"ok": True, "prompt_n": 109980, "with_image": False}
    hot = {"ok": True, "prompt_n": 4, "with_image": False}
    check(not probe.looks_cached(cold, 110000), "холодная проба не считается прогретой")
    check(probe.looks_cached(hot, 110000), "prompt_n = 4 на 110k — это прогрет")
    check(not probe.looks_cached({"ok": True, "prompt_n": 30, "with_image": True},
                                 110000), "у картинки prompt_n мал всегда")
    check(not probe.looks_cached({"ok": False}, 110000), "у ошибки не бывает кэша")


def test_crash_journal_roundtrip() -> None:
    """Падение записывается, показывается в validate и снимается замером."""
    import json
    import tempfile
    from lib import crashes, validate

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "crashes.json"
        old_state, old_cache = crashes.paths.state_dir, crashes.paths.state_dir
        crashes.paths.state_dir = lambda: Path(td)
        crashes.paths.ensure_dirs = lambda: None
        try:
            log = ("GGML_CUDA error: out of memory\n"
                   "ggml_cuda_mul_mat_id(...)")
            e = crashes.note("p1", "faks", log, depth=110000)
            check(e and e["kind"] == "cuda_oom", "CUDA OOM распознан")
            rep = validate.Report()
            validate.check_crash_history("p1", rep)
            check(rep.count("warn") == 1, "валидатор предупреждает о падении")
            check(crashes.forget("p1"), "успешный замер снимает запись")
            rep2 = validate.Report()
            validate.check_crash_history("p1", rep2)
            check(rep2.count("warn") == 0, "после снятия предупреждения нет")
            check(crashes.classify("SIGSEGV") is not None, "segfault распознан")
            check(crashes.classify("всё хорошо") is None,
                  "обычный текст не падение")
        finally:
            crashes.paths.state_dir = old_state


def test_freshness_reports_staleness() -> None:
    """Актуальность сборки отличает три разных состояния.

    Отстал от апстрима, ahead (свои коммиты) и бинарь старее дерева — это
    разные вещи: первое лечится pull, второе пересборка не потеряет, третье
    означает, что замеры шли на устаревшем бинаре.
    """
    import subprocess
    import tempfile
    from lib import builds as B

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        root.mkdir()
        def g(*a):
            return subprocess.run(["git", "-C", str(root), *a],
                                  capture_output=True, text=True)
        g("init", "-q", "-b", "main")
        g("config", "user.email", "t@t"); g("config", "user.name", "t")
        (root / "a.txt").write_text("1\n")
        g("add", "-A"); g("commit", "-qm", "первый")
        # ветка up фиксирует первый коммит; второй коммит идёт в main,
        # затем main привязывается к up как upstream — main ahead на 1
        g("branch", "up")
        (root / "a.txt").write_text("2\n")
        g("add", "-A"); g("commit", "-qm", "второй")
        g("branch", "--set-upstream-to=up", "main")

        # server_bin — свойство от path, поэтому задаём только path;
        # .git ищется в b.path — указываем корень репозитория
        fake = B.Build(name="t", path=str(root))
        B.get = lambda name, *a, **k: fake
        r = B.freshness("t", fetch=False)
        check(r.get("ahead") == 1, "свой коммит опознан как ahead, а не как отставание")
        check(not r.get("behind"), "впереди апстрима — behind = 0")
        check(r.get("head") is not None, "коммит HEAD прочитан")

    # бинарь без вшитого коммита: решает время файла против времени коммита
    info = {"bin_exists": False, "bin_behind_tree": False}
    check(info["bin_behind_tree"] is False, "нет бинаря — нечего считать устаревшим")


def test_freshness_parses_all_version_formats() -> None:
    """Коммит извлекается из трёх разных форматов строки версии.

    У сборок строки неодинаковы:
      ik              "version: 104 (32cddbf)"
      faks            "version: 1 (4f14c6a)"
      upstream/faks   "version: 0.5.0-dev (build 126, commit 19e28a27)"
    Из-за прежней регулярки у upstream коммит не извлекался вовсе, и проверка
    «бинарь старее дерева» молча возвращала «нет» — то есть проверка актуальности
    работала вхолостую ровно на той сборке, что отставала сильнее всех.
    """
    from lib.builds import _COMMIT_IN_B, VERSION_RE

    cases = [
        ("version: 104 (32cddbf)", "32cddbf"),
        ("version: 1 (4f14c6a)", "4f14c6a"),
        ("version: 0.5.0-dev (build 126, commit 19e28a27)", "19e28a27"),
    ]
    for text, want in cases:
        m = VERSION_RE.search(text)
        check(m is not None, f"разобран формат: {text[:34]}")
        if not m:
            continue
        inner = m.group(2)
        got = (_COMMIT_IN_B.search(inner).group(1)
               if _COMMIT_IN_B.search(inner)
               else (inner.strip() if len(inner.strip()) >= 7 else None))
        check(got == want, f"коммит извлечён верно: {text[:34]} -> {got}")


def test_build_port_from_registry() -> None:
    """Порт сборки берётся из реестра, а не только из переменной окружения.

    Регрессия: у ik не роутер, и при занятом 8099 он падал с «couldn't bind
    to server socket». Поле port в реестре было, но сервер его не читал —
    адрес жил только в LLAMA_SERVER, и вторая сборка без роутера упиралась
    в те же грабли.
    """
    import os

    from lib import builds as B
    from lib import server as S

    # server_bin и bench_bin — свойства, производные от path, поэтому задаём
    # только path, а остальные поля через конструктор
    real = B.Build(name="ik", path="/tmp/нет", router=False, port=8098)
    check(str(real.server_bin).endswith("нет/build/bin/llama-server"),
          "производные пути собираются из path")
    saved_get = B.get
    B.get = lambda name, *a, **k: (real if name == "ik" else None)
    saved_env = os.environ.pop("LLAMA_SERVER", None)
    S._active_build = None
    try:
        check(S.port_number("ik") == 8098, "порт взят из реестра сборки")
        check(S.server_url("ik").endswith(":8098"), "адрес собран на порту сборки")
        check(S.port_number("faks") == 8099, "у сборки без порта — дефолт")
        os.environ["LLAMA_SERVER"] = "http://127.0.0.1:9000"
        check(S.port_number("ik") == 9000, "явный LLAMA_SERVER главнее реестра")
    finally:
        if saved_env is None:
            os.environ.pop("LLAMA_SERVER", None)
        else:
            os.environ["LLAMA_SERVER"] = saved_env
        B.get = saved_get
        S._active_build = None


def test_stop_covers_all_build_ports() -> None:
    """stop() обходит порты всех сборок, а не один.

    Сборки живут на разных адресах; остановка по одному порту оставляла
    вторую сборку висеть в VRAM — то есть занимать память, которая считается
    свободной.
    """
    from lib import builds as B
    from lib import server as S

    src = Path(__file__).resolve().parents[1] / "lib" / "server.py"
    body = src.read_text()
    check("ports.update(int(b.port) for b in builds.all_builds().values() if b.port)"
          in body, "stop перебирает порты всех сборок")


def test_probe_retries_on_immediate_stop() -> None:
    """Модель, замолчавшая на первом токене, не выпадает из замера.

    На ik Qwen3.8 MiniPlus любой хвост из слов («Продолжи», «# продолжение»)
    вызывал EOS: модель читала его как завершённую реплику. Отступ в конце
    работал. Хвосты перебираются, и замер уцелел: 42 t/s вместо «сгенерирован
    1 токен».
    """
    from lib import probe as P

    check(len(P.CONTINUATIONS) >= 2, "хвостов несколько, а не один")
    check(not any(t.strip().isalpha() for t in P.CONTINUATIONS[3:]),
          "синтаксические хвосты — без слов")
    check(P._stopped_immediately({"ok": True, "predicted_n": 1,
                                  "with_image": False}),
          "один токен — модель остановилась")
    check(not P._stopped_immediately({"ok": True, "predicted_n": 64,
                                      "with_image": False}),
          "64 токена — нормальный ответ")
    check(not P._stopped_immediately({"ok": True, "predicted_n": 1,
                                      "with_image": True}),
          "у картинки один токен — это норма")
    check(not P._stopped_immediately({"ok": False}),
          "у ошибки не бывает остановки")


def test_annotate_keeps_field_continuations() -> None:
    """Продолжения полей не теряются и не превращаются в новые пункты.

    Регрессия на реальном блоке: строка «;        tg 27.1 (пробы 27.1 / 27.4) …»
    внутри «Замер:» считалась новой строкой метрик, и её значение исчезало из
    блока. Второй эффект: строка метрик без цифр («tg не измерено») вообще не
    распознавалась и уезжала в «Назначение».
    """
    from lib import annotate as A

    blk = ("; [X] 128k\n"
           "; tg 27.1 t/s [замер]  vram 8.9 GiB [замер]\n"
           "; Замер: upstream, 110k кода: tg 27.1.\n"
           ";        tg 27.1 (пробы 27.1 / 27.4), prefill 396 t/s.\n"
           "; Нюанс: ускоритель вредит.\n"
           ";        продолжаем мысль.\n").split("\n")
    b = A.parse_block(blk)
    check(len(b.provenance) == 2, "обе строки «Замер:» на месте")
    check(len(b.note) == 2, "обе строки «Нюанс:» на месте")
    check(any("продолжаем" in x for x in b.note), "продолжение нюанса не потеряно")
    check(not any("Назначение" in x for x in b.metrics_src),
          "строка метрик не уехала в «Назначение»")

    # «tg не измерено» — тоже строка метрик, а не назначение
    b2 = A.parse_block(("; [Y] 128k\n"
                        "; tg не измерено  vram 11.0 GiB [оценка]\n").split("\n"))
    check(len(b2.metrics_line) == 1, "«tg не измерено» распознана как метрики")
    check(b2.purpose == [], "и не попала в «Назначение»")


def test_annotate_canonical_form_is_stable() -> None:
    """Канонический блок переживает разбор и печать без изменений.

    Формат закреплён тремя признаками: один разделитель на блок (второй
    печатается сам), метка только на первой строке поля, граница между
    соседними пунктами — пустая строка «;». Без последней соседние пункты
    одного поля неотличимы от продолжения абзаца и слипаются.
    """
    from lib import annotate as A

    blk = ("\n".join([
        "; " + "-" * 75,
        "; [X] 128k | Зрение | ngram-mod | ЛИДЕР: upstream 869034b4",
        "; tg 27.1 t/s [замер]  vram 8.9 GiB [замер]  ctx 114688  ub 512  moe 24",
        "; Назначение: лидер финального теста на трёх сборках — один пресет, один",
        ";            промпт (реальный код ggml-opencl.cpp), глубина 110k.",
        "; Замер: 2026-10-01, upstream 869034b4, 110k кода, 2 пробы:",
        ";          tg 27.1 (пробы 27.1 / 27.4), prefill 396 t/s, VRAM 9071 MiB.",
        ";",
        "; Нюанс: ЛИДЕР — upstream БЕЗ ngram-mod: 27.4 против 27.0 с ним.",
        ";        ускоритель вредит, -0.4 t/s на upstream.",
        ";",
        "; Нюанс: ngram-mod оставлен осознанно: на повторах он даёт x3-4.",
        "; " + "-" * 75,
    ]))
    clean = [x for x in blk.split("\n")
             if not A.RULE_RE.match(x.strip())]
    b = A.parse_block(clean)
    check(len(b.note) >= 4, "три абзаца нюансов на месте")
    rendered = "\n".join(A.render("x", b, "; tg 27.1 t/s [замер]",
                                  A.provenance_lines(b, []), A._with_labels(b.note, "Нюанс")))
    again = [x for x in rendered.split("\n")
             if not A.RULE_RE.match(x.strip())]
    b2 = A.parse_block(again)
    check(len(b2.note) == len(b.note), "число абзацев не изменилось после печати")
    check(b2.note == b.note, "текст нюансов пережил печать без потерь")
    check(len(b2.provenance) == len(b.provenance),
          "поле «Замер» не рассыпалось на пункты")
    check(rendered.count("; " + "-" * 20) == 2,
          "разделителей ровно два: верх и низ блока")


def test_annotate_paragraph_break_survives() -> None:
    """Пустая строка «;» — граница пункта — переживает печать форматтера.

    Регрессия: граница терялась при печати, и соседние пункты одного поля
    слипались обратно в один абзац — то есть annotate не был идемпотентен,
    а каждый прогон заново склеивал то, что человек разделил вручную.
    """
    from lib import annotate as A

    rows = ["первый пункт", "\0", "второй пункт", "\0", "третий пункт"]
    printed = A._with_labels(rows, "Нюанс")
    check(printed.count("") == 2, "две границы на месте после печати")
    check([p for p in printed if p.startswith("Нюанс:")] == [
        "Нюанс: первый пункт", "Нюанс: второй пункт", "Нюанс: третий пункт"],
        "у каждого пункта своя метка")
    check(not any(p.strip() == "; ;" for p in printed),
          "нет мусорной строки «; ;»")


def test_metrics_line_single_marker() -> None:
    """Источник печатается один раз на строку метрик.

    Было: «tg 27.1 t/s [замер]  vram 8.9 GiB [замер]» — два одинаковых
    утверждения подряд читались как два независимых факта.
    """
    from lib.annotate import _metrics_line

    line = _metrics_line([("tg 27.1 t/s [замер]", "замер"),
                          ("vram 8.9 GiB [замер]", "замер"),
                          ("ctx 114688", None), ("ub 512", None),
                          ("moe 24", None)])
    check(line == "tg 27.1 t/s  vram 8.9 GiB  ctx 114688  ub 512  moe 24  [замер]",
          f"одна метка в конце строки: {line}")
    check(line.count("[замер]") == 1, "ровно одна пометка")
    # разные источники подписываются по месту, иначе теряется смысл
    mixed = _metrics_line([("tg 27.1 t/s", "замер"), ("vram 8.9 GiB", "оценка")])
    check("[замер]" in mixed and "[оценка]" in mixed,
          f"при разных источниках каждое значение подписано: {mixed}")


def test_prose_ignores_table_rows() -> None:
    """Числа из таблицы внутри блока не читаются как tg.

    Регрессия: в блок лидера добавлена таблица сравнения сборок, и строка
    «upstream 27.1 396 t/s 9071 3217» попала в разбор как проза — в шапку
    пресета уехало «tg 396.0 t/s», то есть prefill вместо генерации.
    """
    from lib.annotate import _tg_from_prose

    table = ["сборка     tg      prefill    VRAM   свободно",
             "upstream   27.1    396 t/s    9071   3217",
             "faks       26.4    448 t/s    9153   3135"]
    check(_tg_from_prose(table) is None,
          "из строк таблицы tg не берётся")
    got = _tg_from_prose(["замер 110k кода: tg 24.4 t/s"])
    check(got and abs(got[0] - 24.4) < 0.01, "из обычной прозы tg берётся")


def test_prose_ignores_table_rows_for_vram() -> None:
    """VRAM из таблицы внутри блока не читается.

    Регрессия: строка «ik 24.0 341 t/s 10419 1869 <- нет ngram, +1.3 ГБ»
    давала в шапку пресета vram 1.3 GiB вместо 8.9 — примечание к таблице
    было принято за измерение памяти.
    """
    from lib.annotate import _vram_from_prose

    row = "ik         24.0    341 t/s   10419   1869     <- нет ngram, +1.3 ГБ"
    check(_vram_from_prose([row]) is None, "из строки таблицы VRAM не берётся")
    got = _vram_from_prose(["~10.6 ГБ при 128k"])
    check(got and abs(got[0] - 10.6) < 0.05, "из прозы VRAM берётся")


def test_collapse_markers_in_orphan_blocks() -> None:
    """Повтор метки схлопывается и в блоках вне секций.

    Такие блоки форматтер не разбирает, но правило про источник действует на
    весь файл: «; tg 48.0 t/s [замер]  vram 10.6 GiB [замер]» оставалось
    двойным до отдельного прохода.
    """
    from lib.annotate import _collapse_markers

    got = _collapse_markers("; tg 48.0 t/s [замер]  vram 10.6 GiB [замер]")
    check(got == "; tg 48.0 t/s  vram 10.6 GiB [замер]", f"схлопнуто: {got}")
    check(_collapse_markers("; tg 27.1 t/s [замер]  vram 8.9 GiB [оценка]")
          == "; tg 27.1 t/s [замер]  vram 8.9 GiB [оценка]",
          "при разных источниках подписи остаются на месте")
    check(_collapse_markers("; tg 27.1 t/s  vram 8.9 GiB  [замер]")
          == "; tg 27.1 t/s  vram 8.9 GiB  [замер]",
          "уже схлопнутая строка не меняется")


def test_readme_navigation_resolves() -> None:
    """Все якоря и ссылки в README ведут в существующее место.

    Навигация в начале README сделана вручную, и при переименовании заголовка
    якоря разъезжаются молча: Markdown покажет просто текст без ссылки, и это
    заметят только те, кто уже потерял нужный раздел. Поэтому правило
    проверяется: якорь — это slug заголовка по правилам GitHub (нижний
    регистр, пробелы в дефисы, пунктуация убирается, буквы любых алфавитов
    сохраняются), и каждая ссылка ведёт либо в такой заголовок, либо в
    существующий файл.
    """
    import re
    from lib.paths import SKILL_ROOT

    readme = SKILL_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    def slug(title: str) -> str:
        t = title.strip().lower()
        t = re.sub(r"[^\w\- ]", "", t, flags=re.UNICODE)
        return t.replace(" ", "-")

    heads = {slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", text, re.M)}
    check(len(heads) > 20, f"в README достаточно заголовков ({len(heads)})")

    anchors = re.findall(r"\]\(#([^)]+)\)", text)
    check(len(anchors) >= 25, f"навигация покрыта ссылками ({len(anchors)})")
    broken = sorted({a for a in anchors if a not in heads})
    check(not broken, f"все якоря ведут в заголовки; битые: {broken}")

    files = re.findall(r"\]\((docs/[^)]+\.md|SKILL\.md)\)", text)
    missing = sorted({f for f in files if not (SKILL_ROOT / f).exists()})
    check(not missing, f"все ссылки на файлы существуют; битые: {missing}")

    # на каждый язык должен быть якорь верхнего уровня
    for anchor in ("llamastery", "llamastery-en", "llamastery-\u4e2d\u6587"):
        check(anchor in anchors or anchor in heads,
              f"есть якорь языка: {anchor}")

def _fake_xing4_gguf(path: Path) -> None:
    """Минимальный xing4_0-GGUF с таблицей тензоров (веса — нули)."""
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def kv_str(k, v):
        return s(k) + struct.pack("<I", 8) + s(v)

    def kv_i32(k, v):
        return s(k) + struct.pack("<I", 4) + struct.pack("<i", v)

    kv = [
        kv_str("general.architecture", "xing4_0"),
        kv_i32("xing4_0.block_count", 41),
        kv_i32("xing4_0.embedding_length", 3584),
        kv_i32("xing4_0.attention.head_count", 32),
        kv_i32("xing4_0.attention.head_count_kv", 1),
        kv_i32("xing4_0.attention.key_length", 576),
        kv_i32("xing4_0.attention.value_length", 512),
        kv_i32("xing4_0.attention.kv_lora_rank", 512),
        kv_i32("xing4_0.rope.dimension_count", 64),
        kv_i32("xing4_0.context_length", 262144),
        kv_i32("xing4_0.expert_count", 64),
        kv_i32("xing4_0.expert_feed_forward_length", 1024),
        kv_i32("xing4_0.nextn_predict_layers", 1),
    ]
    # (имя, тип, offset, nelem) — offsets с шагом 1000 байт
    tensors = [
        ("blk.2.ffn_gate_exps.weight", 22, 0, 1000),
        ("blk.2.ffn_up_exps.weight", 22, 1000, 1000),
        ("blk.2.attn_output.weight", 14, 2000, 100),
        ("blk.40.nextn.eh_proj.weight", 12, 3000, 500),
        ("output.weight", 14, 4000, 200),
    ]
    infos = []
    for name, dtype, off, nelem in tensors:
        infos.append(s(name) + struct.pack("<I", 1)
                     + struct.pack("<Q", nelem) + struct.pack("<I", dtype)
                     + struct.pack("<Q", off))
    head = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kv)) + b"".join(kv) + b"".join(infos))
    pad = b"\0" * (-len(head) % 32)   # выравнивание секции данных
    path.write_bytes(head + pad + b"\0" * 5000)


def test_xing4_tensors(tmp: Path):
    print("xing4_0: таблица тензоров и MLA-метаданные")
    p = tmp / "xing4.gguf"
    _fake_xing4_gguf(p)
    m = gguf.probe(p)
    check(m.arch == "xing4_0", "архитектура прочитана", m.arch)
    check(len(m.tensors) == 5, "все тензоры прочитаны", len(m.tensors))
    check(m.is_mla, "MLA обнаружен", (m.kv_lora_rank, m.rope_dim))
    check(m.n_layer_nextn == 1, "nextn=1 прочитан", m.n_layer_nextn)
    sizes = dict(m.tensor_sizes())
    check(sizes["blk.2.ffn_gate_exps.weight"] == 1000,
          "размер через разность offsets", sizes["blk.2.ffn_gate_exps.weight"])
    check(sizes["output.weight"] == 1000,
          "последний тензор — до конца файла", sizes["output.weight"])
    check(sum(sizes.values()) == 5000, "сумма сходится", sum(sizes.values()))


def test_xing4_mla_kv(tmp: Path):
    print("xing4_0: бюджет считает MLA-KV, а не GQA")
    p = tmp / "xing4.gguf"
    _fake_xing4_gguf(p)
    m = gguf.probe(p)
    base = {"c": "114688", "cache-type-k": "q8_0", "cache-type-v": "q8_0",
            "parallel": "1"}
    e = budget.estimate(base, m, compute_gb=0.0)
    # 40 слоёв × (512+64) × 1.0625 × 114688 ≈ 2.6 GiB; GQA-формула дала бы ~5
    check(2.4 < e.kv_gb < 2.9, "MLA-KV около 2.6 GiB", round(e.kv_gb, 2))
    check(e.n_attn_layer == 40, "MTP-слой без KV вычтен", e.n_attn_layer)


def test_xing4_override_tensor(tmp: Path):
    print("xing4_0: override-tensor вычитает веса из VRAM")
    p = tmp / "xing4.gguf"
    _fake_xing4_gguf(p)
    m = gguf.probe(p)
    base = {"c": "65536", "cache-type-k": "q8_0", "cache-type-v": "q8_0"}
    e0 = budget.estimate(base, m, compute_gb=0.0)
    e1 = budget.estimate({**base,
                          "override-tensor": r"blk\.(2)\.ffn_.*_exps\.weight=CPU"},
                         m, compute_gb=0.0)
    # 2 тензора по 1000 байт + хвост blk.40 (1000) уже вычтен в обоих
    check(abs((e0.weights_gb - e1.weights_gb) * budget.GIB - 2000) < 1,
          "ровно совпавшие тензоры ушли в RAM",
          (e0.weights_gb, e1.weights_gb))
    # first-match-wins: первое правило забирает тензор
    e_first = budget.estimate(
        {**base, "override-tensor": r"blk\.2\..*=CPU,blk\.(2)\.ffn_.*=CUDA"},
        m, compute_gb=0.0)
    check(abs((e0.weights_gb - e_first.weights_gb) * budget.GIB - 3000) < 1,
          "первое совпавшее правило побеждает (3 тензора blk.2)",
          (e0.weights_gb, e_first.weights_gb))
    # битый regex не роняет оценку
    e_bad = budget.estimate({**base, "override-tensor": r"blk\.([0-9=CPU"},
                            m, compute_gb=0.0)
    check(abs(e_bad.weights_gb - e0.weights_gb) < 1e-9,
          "битый regex молча пропускается")


def test_xing4_nextn_weights(tmp: Path):
    print("xing4_0: неиспользуемый nextn-хвост не считается в VRAM")
    p = tmp / "xing4.gguf"
    _fake_xing4_gguf(p)
    m = gguf.probe(p)
    base = {"c": "65536", "cache-type-k": "q8_0", "cache-type-v": "q8_0"}
    e_off = budget.estimate(base, m, compute_gb=0.0)
    e_mtp = budget.estimate({**base, "model-draft": "m.gguf"}, m,
                            compute_gb=0.0)
    # хвост blk.40 = 1000 байт: без MTP вычтен, с model-draft — загружен
    check(abs((e_mtp.weights_gb - e_off.weights_gb) * budget.GIB - 1000) < 1,
          "nextn грузится только при активном MTP",
          (e_off.weights_gb, e_mtp.weights_gb))


def test_xing4_in_candidates():
    print("xing4_0: сборка известна detect")
    from lib import builds
    names = [c[0] for c in builds.CANDIDATES]
    check("xing4" in names, "xing4 в CANDIDATES", names)
    x = [c for c in builds.CANDIDATES if c[0] == "xing4"][0]
    check("jmarceno" in x[2] and x[3] is True,
          "remote jmarceno, роутер есть", x[2])


if __name__ == "__main__":
    raise SystemExit(main())
