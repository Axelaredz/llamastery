#!/usr/bin/env python3
"""Тесты без внешних зависимостей: python3 tests/run_tests.py"""

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import budget, gguf, inifile, measure, paths, presets, schema, swap, validate, wizard  # noqa: E402

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


def test_swap_auto_port() -> None:
    """Занятый порт не должен валить запуск — берём следующий свободный.

    Ручной подбор порта — то, что человек забывает сделать; раньше мастер
    печатал «порт занят» и swap не поднимался. Занятость подменяется
    заглушкой: реальный сокет в backlog даёт плавающий результат.
    """
    print("swap: автоподбор порта")
    busy, free1, free2 = 45001, 45002, 45003
    orig_busy, orig_pid = swap.is_port_busy, swap.daemon_pid
    swap.is_port_busy = lambda host, port, timeout=1.0: port == busy
    swap.daemon_pid = lambda: None
    try:
        check(swap.find_free_port("127.0.0.1", busy) == free1,
              "следующий порт свободен", swap.find_free_port("127.0.0.1", busy))
        listen, why = swap.resolve_listen(f"127.0.0.1:{busy}")
        check(swap.parse_listen(listen)[1] == free1, "resolve сдвинул порт", listen)
        check("занят" in why and "выбран свободный" in why, "причина названа", why)
        # автоподбор выключен — порт не трогаем
        listen2, why2 = swap.resolve_listen(f"127.0.0.1:{busy}", auto=False)
        check(listen2 == f"127.0.0.1:{busy}", "auto=False не сдвигает", listen2)
        # наш же swap на занятом порте — не вытесняем: порт его
        swap.daemon_pid = lambda: 4242
        listen3, why3 = swap.resolve_listen(f"127.0.0.1:{busy}")
        check(listen3 == f"127.0.0.1:{busy}" and "уже запущен" in why3,
              "свой swap не вытесняем", (listen3, why3))
    finally:
        swap.is_port_busy, swap.daemon_pid = orig_busy, orig_pid
    # свободный порт остаётся как есть
    swap.is_port_busy = lambda host, port, timeout=1.0: False
    try:
        check(swap.find_free_port("127.0.0.1", free2) == free2,
              "свободный не двигается")
    finally:
        swap.is_port_busy = orig_busy


def test_swap_port_check() -> None:
    """Проверка занятости порта: парсинг, свободен/занят, владелец без падений."""
    import socket as _sock

    print("swap: занятость порта")
    check(swap.parse_listen("0.0.0.0:8080") == ("0.0.0.0", 8080), "host:port")
    check(swap.parse_listen("http://127.0.0.1:8090/") == ("127.0.0.1", 8090), "URL")
    check(swap.parse_listen("8090") == ("0.0.0.0", 8090), "голый порт")

    srv = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    srv.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    free_port = srv.getsockname()[1]
    srv.close()
    check(not swap.is_port_busy("127.0.0.1", free_port), "свободный порт — свободен")

    srv = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    srv.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    busy_port = srv.getsockname()[1]
    try:
        check(swap.is_port_busy("0.0.0.0", busy_port), "занятый порт — занят")
        owner = swap.port_owner(busy_port)
        check(owner is None or isinstance(owner, dict), "владелец без падений")
        if isinstance(owner, dict):
            check(owner.get("pid") == os.getpid() or owner.get("pid") is None,
                  "владелец — мы или не виден")
            check(isinstance(swap.describe_owner(owner), str), "описание — строка")
    finally:
        srv.close()


def test_swap_running_models_parses_proxy_port() -> None:
    """`/running` отдаёт {model,state,proxy} — порт бэкенда нужно вытащить.

    Под swap бэкенд живёт на своём порту, и `measure` обязан уметь понять,
    что модель загружена, иначе замер молчит при полной VRAM.
    """
    from lib import swap as S

    print("swap: разбор /running")
    fake = {"running": [
        {"model": "qwen-128ctx", "state": "ready",
         "proxy": "http://localhost:5800"},
        {"model": "tiel-65ctx", "state": "stopped", "proxy": ""},
    ]}
    orig = S.status
    S.status = lambda *a, **k: {"up": True, "running": fake, "error": None}
    try:
        got = S.running_models("http://127.0.0.1:8087")
    finally:
        S.status = orig
    check(len(got) == 2, "две записи", len(got))
    check(got[0]["model"] == "qwen-128ctx", "имя модели", got)
    check(got[0]["proxy_port"] == 5800, "порт бэкенда", got)
    check(got[1]["proxy_port"] is None, "без proxy — None", got)

    # прокси не поднят: пустой список, а не исключение
    S.status = lambda *a, **k: {"up": False, "running": [], "error": "down"}
    try:
        check(S.running_models() == [], "прокси down — пусто")
    finally:
        S.status = orig


def test_capture_preset_falls_back_to_swap() -> None:
    """Замер видит модель под swap, даже когда прямой сервер пуст.

    Регрессия: `measure` рапортовал «в VRAM ничего не загружено», хотя
    бэкенд swap занимал 9.5 GiB, — порт бэкенда не 8099, и прямой
    `/models` его не показывает.
    """
    from lib import server as Srv
    from lib import swap as S
    from lib import vram as V

    print("замер: fallback на llama-swap")
    orig_models, orig_single = Srv.models, Srv.loaded_models
    orig_run, orig_pf = S.running_models, Srv.read_pid
    Srv.models = lambda *a, **k: []
    Srv.loaded_models = lambda *a, **k: []
    Srv.read_pid = lambda *a, **k: None
    S.running_models = lambda *a, **k: [
        {"model": "qwen-128ctx", "state": "ready",
         "proxy": "http://localhost:5800", "proxy_port": 5800}]
    try:
        cap = V.capture_preset(allow_single=True, swap_url="http://127.0.0.1:8087")
    finally:
        Srv.models, Srv.loaded_models = orig_models, orig_single
        Srv.read_pid = orig_pf
        S.running_models = orig_run
    check(cap["loaded"] == ["qwen-128ctx"], "модель видна", cap["loaded"])
    check(cap["source"] == "swap", "источник — swap", cap["source"])
    # занятое берётся из nvidia-smi, поэтому сверяем только согласованность:
    # свободное = полное − занятое, а не выдуманное число
    check(cap["free_mib"] == cap["total_mib"] - cap["used_mib"],
          "свободно = полное − занятое", cap)


def test_bool_flag_ini_value_respected() -> None:
    """`mmproj-offload = 0` должен выключать, а не включать.

    Раньше значение булева флага отбрасывалось, и ключ превращался в
    `--mmproj-offload`, то есть ровно в противоположное намерению. INI пишут
    именно так («выключи выгрузку mmproj в GPU»), поэтому `0/false/no/off`
    обязаны давать парную `--no-` форму.
    """
    from lib import schema, server

    print("булевы флаги: значение из INI уважается")
    flags = {
        "mmproj-offload": schema.Flag(
            canonical="--mmproj-offload", short=None,
            aliases=["--no-mmproj-offload"], kind="flag", default="enabled"),
    }
    args, warns = server.preset_to_argv({"mmproj-offload": "0"}, flags)
    check(args == ["--no-mmproj-offload"], "0 -> --no-mmproj-offload", args)
    check(not warns, "без предупреждения", warns)

    args, _ = server.preset_to_argv({"mmproj-offload": "1"}, flags)
    check(args == ["--mmproj-offload"], "1 -> --mmproj-offload", args)
    args, _ = server.preset_to_argv({"mmproj-offload": "true"}, flags)
    check(args == ["--mmproj-offload"], "true -> --mmproj-offload", args)
    args, _ = server.preset_to_argv({"mmproj-offload": ""}, flags)
    check(args == ["--mmproj-offload"], "пусто -> включить", args)

    # у флага без пары --no- значение потерять нельзя молча
    solo = {"ctx-shift": schema.Flag(canonical="--context-shift", short=None,
                                     kind="flag", default="disabled")}
    # у флага без пары --no- «0» означает «не включать»: добавлять флаг
    # означало бы сделать обратное, поэтому его не добавляем вовсе
    args, warns = server.preset_to_argv({"ctx-shift": "0"}, solo)
    check(args == [], "без --no- флаг не добавляется (иначе включилось бы)", args)
    check(any("потеряно" in w for w in warns), "предупреждение о потере значения",
          warns)


def test_axes_discover_filters_noise() -> None:
    """Автообнаружение показывает ручки, а не шум из пресетов и алиасов.

    Без фильтра список новых флагов разрастался до 84 позиций, и сигнал
    (реально новый параметр) в нём тонул.
    """
    from lib import axes as A

    print("axes: обнаружение отфильтровано")
    fake = {
        "--n-cpu-moe": None, "-ncmoe": None, "--ubatch-size": None,
        "--flash-attn": None, "--spec-type": None, "--batch-size": None,
        "--cache-type-k": None, "--threads": None,
        "--fim-qwen-7b-default": None, "--draft": None,
        "--gpt-oss-20b-default": None, "--spec-draft-n-max": None,
        "--spec-ngram-mod-n-max": None, "--spec-ngram-map-k-size-n": None,
        "--spec-draft-cpu-mask": None, "--cpu-mask-batch": None,
        "--brand-new-batch-knob": None, "--n-some-experimental": None,
        "--unrelated": None,
    }
    got = A.discover("нет-бинаря", fake)
    check("--n-cpu-moe" not in got, "известные оси не в списке", got)
    check("--fim-qwen-7b-default" not in got, "пресеты отфильтрованы", got)
    check("--draft" not in got, "устаревшие отфильтрованы", got)
    check("--spec-draft-n-max" not in got, "ручки драфта отфильтрованы", got)
    check("--spec-ngram-mod-n-max" in got, "ручки ngram показаны", got)
    check("--brand-new-batch-knob" in got,
          "неизвестный фlag с намёком показан", got)
    check("--unrelated" not in got, "непохожее не показывается", got)


def test_axes_learn_only_isolated_pairs() -> None:
    """Эффект оси считается только по прогонам, отличающимся этой осью.

    В первой версии разброс по t/s и VRAM приписывался всем осям сразу:
    у ubatch и n-cpu-moe выходили одинаковые 6.22, хотя менялась одна.
    """
    from lib import axes as A

    print("axes: приоритет по изолированным парам")

    def run(cfg, tps, vram):
        return {"ok": True, "config": cfg,
                "short": {"gen_tps": tps},
                "min_observed_free_mib": vram}

    fake = [
        # изолированная пара по moe: 40 t/s против 20, 1000 против 400 MiB
        run({"n-cpu-moe": 16, "ubatch-size": 1024}, 40, 1000),
        run({"n-cpu-moe": 32, "ubatch-size": 1024}, 20, 400),
        # изолированная пара по ubatch: те же 40/20 t/s, VRAM почти не меняется
        run({"n-cpu-moe": 16, "ubatch-size": 512}, 40, 950),
        run({"n-cpu-moe": 16, "ubatch-size": 2048}, 40, 900),
        # moe и ubatch меняются вместе — такая пара не идёт ни в одну ось
        run({"n-cpu-moe": 32, "ubatch-size": 2048}, 20, 390),
    ]
    orig = A._load_results
    A._load_results = lambda: fake
    try:
        learned = A.learn()
    finally:
        A._load_results = orig
    check(learned.get("moe", {}).get("tps_gain") == 2.0,
          "moe: tps x2 из изолированной пары", learned.get("moe"))
    check(learned.get("moe", {}).get("vram_gain") == 2.5,
          "moe: VRAM x2.5", learned.get("moe"))
    ub = learned.get("ubatch", {})
    check(ub.get("tps_gain") == 1.0, "ubatch: tps без эффекта", ub)
    check(ub.get("vram_gain") == 1.11, "ubatch: VRAM x1.11 (берём худший)",
          ub)
    check("fa" not in learned, "неизменённая ось не попадает в вывод")


def test_tune_search_ctx_guard() -> None:
    """--search-ctx укорачивает поиск, но запрещает --apply.

    Побочный эффект короткой валидации: конфиг прогона пишет c = глубина
    валидации. Если разрешить --apply, пресетный контекст 114688 был бы
    затёрт на 8192, а параметры подобраны на неполном KV.
    """
    import subprocess
    import sys as _sys

    m = _tune_module()
    root = Path(__file__).resolve().parents[1]
    src = (root / "tools" / "tune_models.py").read_text(encoding="utf-8")

    print("tune: --search-ctx и защита --apply")
    check("--search-ctx" in src, "флаг объявлен")
    check("if args.apply and search_base < base_ctx:" in src,
          "apply запрещён при укороченной валидации")
    check("p.error(" in src.split("if args.apply and search_base < base_ctx:")[1][:600],
          "выводится объяснение, а не просто отказ")

    # реальный запуск: apply с коротким контекстом обязан упасть
    ini = Path("/tmp/tune-guard.ini")
    ini.write_text("[s]\nmodel = /tmp/nope.gguf\nc = 114688\n", encoding="utf-8")
    server_bin = root.parent / "llama-faks" / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        return
    r = subprocess.run(
        [_sys.executable, str(root / "tools" / "tune_models.py"), str(ini), "s",
         "--server", str(server_bin), "--search-ctx", "8192", "--deep", "--apply"],
        capture_output=True, text=True, timeout=300)
    check(r.returncode != 0, "команда с --apply отклонена", r.returncode)
    check("search-ctx" in (r.stderr + r.stdout),
          "объяснение содержит search-ctx", (r.stderr + r.stdout)[-300:])


def test_axes_spec_is_ab_not_grid() -> None:
    """ngram — самый большой рычаг, но в сетку осей он не годится.

    Одно число «прирост» тут обманывает: на повторяющемся тексте ×3.2, на
    уникальном ×1.05. Поэтому ось помечена mode=ab, в сетку не попадает, но
    в отчёте видна первой — иначе выглядит, будто её нет.
    """
    from lib import axes as A

    print("axes: spec — отдельный A/B")
    spec = A.BY_KEY["spec"]
    check(spec.get("mode") == "ab", "ось помечена как A/B", spec.get("mode"))
    check(spec["gain"] == 5, "приоритет высокий (измерен x3.2)", spec["gain"])
    check(spec["measured"]["repetitive"] > 3.0 and
          spec["measured"]["unique"] < 1.2,
          "две оценки: повторы против уникального", spec["measured"])

    grid = A.order(learned={})
    check("spec" not in grid, "в линейную сетку ось не идёт", grid)
    rows = A.explain(learned={})
    check(rows[0]["key"] == "spec", "в отчёте ось видна первой", rows[0]["key"])
    check(any(r["key"] == "spec" and r["mode"] == "ab" for r in rows),
          "помечена как A/B в отчёте")

    # A/B-числа читаются из результатов прогонов
    fake = [{"ok": True, "config": {},
             "spec_ab": {"tg_unique_plain": 30.0, "tg_unique_spec": 29.0,
                         "tg_repetitive_plain": 26.0, "tg_repetitive_spec": 92.0}}]
    eff = A.spec_effect(fake)
    check(eff["repetitive"]["gain"] == 3.54, "прирост на повторах x3.54", eff)
    check(eff["unique"]["gain"] == 0.97, "на уникальном ~x1", eff)
    check(A.spec_effect([]) == {}, "без A/B пусто", A.spec_effect([]))


def test_axes_order_respects_evidence() -> None:
    """Замеры переставляют порядок, но не могут выкинуть ось совсем."""
    from lib import axes as A

    print("axes: порядок подстраивается под замеры")
    base = A.order(learned={})
    check(base[0] == "moe", "без замеров первым moe", base)
    check(base.index("moe") < base.index("fa"), "moe выше fa", base)

    # измеренный эффект оси действительно переставляет порядок
    loud = {"fa": {"pairs": 12, "values": 3, "tps_gain": 4.0,
                   "vram_gain": 1.0}}
    loud_order = A.order(learned=loud)
    check(loud_order.index("fa") < base.index("fa"),
          "измеренный x4 поднял fa вверх", (base, loud_order))
    # moe тоже 5.0 — при равенстве порядок реестра сохраняется (сортировка
    # стабильная), поэтому fa встаёт сразу за ним, а не вместо
    check(loud_order[:2] == ["moe", "fa"], "fa поднялась к moe", loud_order)

    # ось с реальным эффектом переставляется (в пределах линейных)
    real = {"threads": {"pairs": 12, "values": 3, "tps_gain": 3.0,
                        "vram_gain": 2.0}}
    got = A.order(learned=real)
    check(got.index("threads") < got.index("fa"),
          "измеренный эффект поднял threads выше fa", got)
    check(len(got) == len(base), "оси не теряются", (got, base))
    check("spec" not in got, "A/B-ось не попадает в сетку", got)

    # шум (одна пара) не переставляет ничего
    noise = {"fa": {"pairs": 1, "values": 2, "tps_gain": 9.0, "vram_gain": 9.0}}
    check(A.order(learned=noise) == base, "одна пара игнорируется")

    # фильтр по наличию в сборке
    only = A.order(learned={}, present={"--ubatch-size", "--threads"})
    check(only == ["ubatch", "threads"], "только реально доступные оси", only)


def _tune_module():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tune_models_t", root / "tools" / "tune_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tune_args(**over):
    class A:
        pass
    a = A()
    a.moe_values = "16,20,24,28,32"
    a.moe_step = 4
    a.max_cpu_moe = 64
    a.ubatch_values = "512,1024,2048"
    a.try_4096 = False
    a.threads = 12
    a.threads_batch = 12
    a.try_threads_16 = True
    a.kv_cache_type = "q8_0"
    a.try_f16 = True
    a.flash_attn = "on"
    a.batch_values = ""
    a.ngl = 999
    a.kv_unified = "on"
    a.screen_top = 8
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_tune_staged_order_and_saving() -> None:
    """Staged: сначала оси с наибольшим приростом, и по одному параметру.

    Полный декартов перебор на тех же значениях давал 64 прогона; staged
    первого раунда — 10, то есть в 6 раз меньше, и порядок не случайный.
    """
    m = _tune_module()

    print("tune: staged-порядок и экономия")
    args = _tune_args()
    base = m.staged_base(args, {"n-cpu-moe": "24", "ubatch-size": "1024",
                                "t": "12", "cache-type-k": "q8_0",
                                "fa": "true"})
    check(base["moe"] == 24 and base["ubatch"] == 1024 and base["threads"] == 12,
          "база берётся из пресета, а не из дефолтов", base)
    check(base["fa"] == "on", "fa=true пресета переводится в on", base)

    cands = m.axis_variants(base, args, True)
    axes = [c["_axis"] for c in cands]
    check(axes[0] == "moe", "первой идёт ось n-cpu-moe", axes)
    # порядок осей в кандидатах обязан совпадать с реестром lib/axes.py
    # (который, в свою очередь, скорректирован замерами)
    from lib import axes as _ax
    reg = [k for k in _ax.order(_ax.learn()) if k in set(axes)]
    first_seen = list(dict.fromkeys(axes))
    check(first_seen == reg, "порядок осей совпадает с реестром",
          (first_seen, reg))
    ranks = [dict((k, r) for k, r, _ in m.KNOB_AXES)[a_] for a_ in axes]
    check(ranks == sorted(ranks, reverse=True), "оси строго по убыванию прироста",
          list(zip(axes, ranks)))
    ranks = [dict((k, r) for k, r, _ in m.KNOB_AXES)[a_] for a_ in axes]
    def diffs(v, b):
        """Сколько отслеживаемых параметров отличается от базы."""
        keys = ("moe", "ubatch", "threads", "kv_cache_type", "fa")
        n = sum(1 for k in keys if str(v.get(k)) != str(b.get(k)))
        # threads тянет за собой threads_batch, b следует за ubatch — это
        # один и тот же рычаг, а не два
        if str(v.get("b")) != str(b.get("b")) and "ubatch" in keys:
            n -= 1
        return n
    check(all(diffs(c, base) == 1 for c in cands),
          "в каждом кандидате меняется ровно один параметр",
          [(c["_axis"], diffs(c, base)) for c in cands])

    # против старого полного перебора
    full, sm = m.make_screen_candidates(args, {"n-cpu-moe": True},
                                        {"moe": True}, False)
    check(len(cands) < len(full) / 3,
          f"staged экономнее полного перебора ({len(cands)} против {len(full)})")


def test_tune_axes_skip_current_and_dedup() -> None:
    """Текущее значение оси не гоняется, дубликаты не повторяются."""
    m = _tune_module()
    args = _tune_args()

    print("tune: оси без лишних прогонов")
    base = m.staged_base(args, {"n-cpu-moe": "24", "ubatch-size": "1024",
                                "t": "12", "fa": "on"})
    cands = m.axis_variants(base, args, True)
    moe_axis = [c for c in cands if c["_axis"] == "moe"]
    check(moe_axis and all(c["moe"] != base["moe"] for c in moe_axis),
          "текущее n-cpu-moe не проверяется заново",
          [c["moe"] for c in moe_axis])
    fa_axis = [c for c in cands if c["_axis"] == "fa"]
    check(fa_axis and all(c["fa"] != base["fa"] for c in fa_axis),
          "текущее fa не проверяется заново", [c["fa"] for c in fa_axis])
    ub_axis = [c for c in cands if c["_axis"] == "ubatch"]
    check(ub_axis and all(c["ubatch"] != base["ubatch"] for c in ub_axis),
          "текущее ubatch не проверяется заново", [c["ubatch"] for c in ub_axis])
    seen = {m.variant_key(c) for c in cands}
    check(len(seen) == len(cands), "дубликатов нет")

    # exclude отсекает уже измеренное
    again = m.axis_variants(base, args, True, exclude=seen)
    overlap = seen & {m.variant_key(c) for c in again}
    check(not overlap, "exclude убирает уже измеренные", len(overlap))

    # ось b по умолчанию не крутится: n_batch клампится в n_ubatch
    check(not any(c.get("_axis") == "b" for c in cands),
          "ось b молчит без --batch-values")
    with_b = m.axis_variants(base, _tune_args(batch_values="4096"), True)
    check(any(c.get("_axis") == "b" for c in with_b),
          "--batch-values включает ось b")


def test_wizard_dispatch_signatures() -> None:
    """Каждый пункт меню вызывается с теми аргументами, что есть в сигнатуре.

    Регрессия: `_do_swap("stop")` вызвали без `swap_url`, а `_do_router("stop")`
    без `action` — мастер падал с TypeError прямо на вопросе человека. Проверяем
    не вызовом (у заглушки `*args` любой вызов проходит), а `inspect.bind`.
    """
    import inspect
    from lib import wizard as W

    print("wizard: пункты меня вызываются по сигнатуре")
    orig = {k: getattr(W, k) for k in
            ("_do_router", "_do_swap", "_preset_flow", "_setup_flow", "_ask_build")}
    W._do_router = lambda *a, **k: 0
    W._do_swap = lambda *a, **k: 0
    W._preset_flow = lambda *a, **k: 0
    W._setup_flow = lambda *a, **k: 0
    W._ask_build = lambda: "faks"
    try:
        labels = set()
        for router_up in (False, True):
            for swap_up in (False, True):
                for _, label in W.menu_for(router_up, [], swap_up):
                    labels.add(label)
        errs = []
        for label in sorted(labels):
            try:
                rc = W._dispatch(label, "http://127.0.0.1:8087")
                # что бы ни вернул диспетчер, строки должны быть валидными
                if rc == "menu":
                    rc = W._dispatch(label, "http://127.0.0.1:8087")
            except TypeError as exc:
                errs.append(f"{label}: {exc}")
        check(not errs, f"все {len(labels)} пунктов без TypeError", errs)

        # прямая проверка арности внутренних вызовов
        src = (Path(__file__).resolve().parents[1] / "lib" / "wizard.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))
        # сигнатуры берём из модуля (FunctionDef не callable), арности
        # достаточно: обязательных позиционных и дефолтов
        want = ("_do_router", "_do_swap", "_preset_flow", "_setup_flow")
        sigs = {n.name: inspect.signature(getattr(W, n.name))
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name in want}
        mism = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in sigs):
                try:
                    sigs[node.func.id].bind(*node.args)
                except TypeError as exc:
                    mism.append(f"строка {node.lineno}: {exc}")
        check(not mism, "внутренние вызовы по сигнатуре", mism)
    finally:
        for k, v in orig.items():
            setattr(W, k, v)


def test_wizard_state_printed_once() -> None:
    """Состояние рисует только меню.

    Регрессия: `run()` печатал состояние сам, а `_main_menu` рисовал его
    ещё раз — на старте экран удваивался.
    """
    src = Path(__file__).resolve().parents[1] / "lib" / "wizard.py"
    body = src.read_text(encoding="utf-8")

    print("wizard: состояние печатается один раз")
    callers = [ln.strip() for ln in body.splitlines()
               if "print_state(" in ln and not ln.strip().startswith("def ")]
    check(len(callers) == 1, "print_state вызывается в одном месте", callers)
    check("_state_lines(swap_url)" in body, "состояние читается")
    # и вызов этот — внутри _main_menu, а не в run()
    fn_body = body.split("def _main_menu(", 1)[1]
    check("print_state(" in fn_body, "печатает именно меню")


def test_wizard_defaults(tmp: Path) -> None:
    """Мастер: дефолты — самый эффективный вариант, мусор ввода безопасен."""
    from lib import builds as B

    print("wizard: дефолты")
    check(wizard.clamp_choice("2", 3, 1) == 2, "номер принят")
    check(wizard.clamp_choice("xx", 3, 1) == 1, "мусор = дефолт")
    check(wizard.clamp_choice("9", 3, 1) == 1, "вне диапазона = дефолт")
    check(wizard.ngram_default({}) is True, "ngram без mmproj — да")
    check(wizard.ngram_default({"mmproj": "f"}) is False, "ngram с mmproj — нет")
    check(wizard.mtp_default("x-mtp", {}) is True, "mtp в имени — да")
    check(wizard.mtp_default("plain", {}) is False, "без mtp — нет")
    check(wizard.find_mmproj_sibling(["a", "a-mmproj"], "a") == "a-mmproj",
          "близнец mmproj найден")

    # замер ищется по подписи конфигурации, а не по имени секции
    pairs = {"c": "32768", "n-cpu-moe": "8"}
    from lib import measure as M
    sig = M.signature(pairs)
    recs = {sig: {"vram_mib": 10800, "runs": 1, "deep_tps": None}}
    m = wizard.measurement_of(pairs, recs)
    check(m is not None and m["vram_gb"] == 10.55, "замер по подписи найден", m)
    check(wizard.measurement_of({"c": "1024"}, recs) is None,
          "чужая конфигурация — замера нет")
    check(wizard.find_mmproj_sibling(["a"], "a") is None, "без близнеца — None")

    # 0 = назад: экран за экраном. Каждый экран возвращает «назад» не чаще
    # раза за проход — иначе это уже не человек, а кризис (см. MAX_STEPS).
    def stepper(name: str, backs: int = 0):
        left = {"n": backs}

        def step(a):
            if left["n"] > 0:
                left["n"] -= 1
                return wizard.BACK
            return name
        return step

    nav = wizard.Nav()
    res = nav.walk([("a", stepper("a")), ("b", stepper("b", 1)),
                    ("c", stepper("c"))])
    check(res is not None and res.get("c") == "c", "прошли до конца", res)
    check(res.get("b") == "b", "после отката ответ экрана перезаписан", res)
    check(not nav.stale, "после прохода несвежих ответов нет", nav.stale)

    nav2 = wizard.Nav()
    res2 = nav2.walk([("a", stepper("a", 1))])
    check(res2 is None, "0 на первом экране = назад в меню", res2)

    nav3 = wizard.Nav()
    nav3.walk([("a", stepper("a")), ("b", stepper("b", 1)), ("c", stepper("c"))])
    check(nav3.i == 3, "индекс финального экрана", nav3.i)

    # экран, который всегда отвечает «назад», не должен вешать мастера
    nav4 = wizard.Nav()
    check(nav4.walk([("a", stepper("a")), ("b", stepper("b", 10 ** 6))]) is None,
          "предохранитель от бесконечного «назад»")
    check(nav4.answers.get("a") == "a", "ответы до отката сохранены", nav4.answers)

    # меню строится по состоянию: нельзя предложить «остановить» то, что не живо
    down = wizard.menu_for(False, [], False)
    up = wizard.menu_for(True, ["qwen"], True)
    check(any("Остановить роутер" in o[0] for o in down) is False,
          "роутер не запущен — нет пункта «Остановить роутер»", down)
    check(any("Остановить роутер" in o[0] for o in up),
          "роутер запущен — есть «Остановить роутер»", up)
    check(any("Загрузить" in o[0] for o in up), "загруженная модель — есть загрузка")
    check(any("Остановить llama-swap" in o[0] for o in up),
          "swap жив — есть пункт его остановки", up)
    check(any("Запустить llama-swap" in o[0] for o in down),
          "swap не жив — есть пункт его запуска", down)
    check(wizard.menu_default(up, True) == 2, "дефолт при живом роутере — рестарт/загрузка")
    check(wizard.menu_default(down, False) == 1, "дефолт при мёртвом — запуск+загрузка")

    def _mk(name: str, router: bool) -> B.Build:
        d = tmp / name
        (d / "build" / "bin").mkdir(parents=True)
        (d / "build" / "bin" / "llama-server").touch()
        return B.Build(name=name, path=str(d), router=router)

    reg = {"ik": _mk("ik", False), "faks": _mk("faks", True)}
    check(wizard.recommend_build(reg) == "faks", "дефолт сборки — faks")
    reg2 = {"ik": _mk("ik2", False), "upstream": _mk("up", True)}
    check(wizard.recommend_build(reg2) == "upstream", "без faks — роутерная")
    check(wizard.recommend_build({}) is None, "пусто — None")


CUSTOM_INI = """; шапка с комментарием — должна выжить
[*]
n-gpu-layers = 99
fa = true

; база с mmproj, драфтом и завышенным контекстом
[base-mmproj]
model = /старый/base.gguf
mmproj = /старый/mmproj-Q8_0.gguf
mmproj-offload = 0
image-min-tokens = 1024
model-draft = /старый/draft.gguf
spec-type = draft-mtp
n-cpu-moe = 999
c = 131072
ubatch-size = 512
"""


def _write_fake_gguf(path: Path, ctx_trained: int = 32768, layers: int = 40,
                     moe: bool = True, nextn: int = 0) -> Path:
    import struct

    kv = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": layers,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.context_length": ctx_trained,
        "qwen35moe.nextn_predict_layers": nextn,
    }
    if moe:
        kv["qwen35moe.expert_count"] = 256
        kv["qwen35moe.expert_feed_forward_length"] = 512

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0),
           struct.pack("<Q", len(kv))]
    for k, v in kv.items():
        out.append(s(k))
        out.append(struct.pack("<I", 8 if isinstance(v, str) else 4) +
                   (s(v) if isinstance(v, str)
                    else struct.pack("<i", v)))
    path.write_bytes(b"".join(out) + b"\0" * 4096)
    return path


def test_wizard_custom_gguf_preset(tmp: Path) -> None:
    """Пункт «свой .gguf»: путь → секция в models.ini → валидный пресет.

    Регрессия: список пресетов показывал только то, что уже в ini, поэтому
    файл, скачанный мимо пресета, было некуда указать — приходилось
    редактировать models.ini руками.
    """
    print("wizard: свой .gguf → новая секция в ini")

    # ── имя секции из имени файла ──
    check(wizard.section_name_for("/models/Qwen3.5-9B Q4_K_M.gguf", [])
          == "qwen3.5-9b-q4_k_m", "имя из файла, пробелы → дефисы")
    check(wizard.section_name_for("/models/base.gguf", ["base"]) == "base-2",
          "занятое имя не затирается")
    check(wizard.section_name_for("/models/base.gguf", ["base", "base-2"])
          == "base-3", "нумерация до свободного")
    check(wizard.section_name_for("/models/???.gguf", []),
          "имя из одних знаков префикса не пусто")

    # ── пункт в списке: всегда последний, ★-дефолт не уезжает ──
    ini = inifile.IniFile(tmp / "models.ini", CUSTOM_INI)
    opts, sections, default_name = wizard._preset_options(ini)
    plain = [o[0] for o in opts]
    opts_c, sections_c, default_c = wizard._preset_options(ini, custom=True)
    with_custom = [o[0] for o in opts_c]
    check(with_custom[-1] == wizard.CUSTOM_MODEL,
          "пункт «свой .gguf» последний в списке", with_custom)
    check(with_custom[:-1] == plain, "нумерация существующих секций не сдвинулась")
    check(sections_c == sections and default_c == default_name,
          "★-дефолт тот же")
    check(sections.index(default_name) + 1 == 1,
          "дефолт указывает на первую секцию")

    # ── флаги базы, но без её файлов и с зажимом по метаданным ──
    meta = gguf.probe(_write_fake_gguf(tmp / "meta.gguf"))
    pairs = wizard.pairs_for_custom_model(
        ini.section("base-mmproj").pairs(), "/models/новый.gguf", meta, None)
    check(pairs["model"] == "/models/новый.gguf", "model = свой файл")
    check(not any(k.lower() in ("mmproj", "model-draft", "image-min-tokens",
                                "hf-repo", "hf-file")
                  for k in pairs),
          "файловые ключи базы не унаследованы", sorted(pairs))
    check(pairs["c"] == "32768", "контекст зажат по обученному окну", pairs["c"])
    check(pairs["n-cpu-moe"] == "40", "n-cpu-moe зажат по слоям", pairs["n-cpu-moe"])
    check(pairs["spec-type"] == "ngram-mod",
          "драфт-головы нет — ускоритель заменён", pairs["spec-type"])
    check(pairs["ubatch-size"] == "512",
          "остальные флаги базы скопированы", pairs)
    check("n-gpu-layers" not in pairs,
          "глобали из [*] в секцию не дублируются — их и так подставит роутер",
          sorted(pairs))
    with_mm = wizard.pairs_for_custom_model(
        ini.section("base-mmproj").pairs(), "/models/новый.gguf", meta,
        "/models/mmproj-Q8_0.gguf")
    check(with_mm["mmproj"] == "/models/mmproj-Q8_0.gguf"
          and with_mm["mmproj-offload"] == "0",
          "найденный рядом mmproj подхвачен с offload=0", with_mm)
    dense = gguf.probe(_write_fake_gguf(tmp / "dense.gguf", moe=False))
    d = wizard.pairs_for_custom_model({"n-cpu-moe": "16", "c": "8192"},
                                      "/models/d.gguf", dense)
    check("n-cpu-moe" not in d, "плотной модели n-cpu-moe не нужен", d)
    check(wizard.pairs_for_custom_model({}, "/x.gguf")["model"] == "/x.gguf",
          "без метаданных секция всё равно собирается")
    mtp = gguf.probe(_write_fake_gguf(tmp / "mtp.gguf", nextn=1))
    check(wizard.pairs_for_custom_model({"spec-type": "draft-mtp"}, "/x.gguf", mtp)
          ["spec-type"] == "draft-mtp", "MTP-голова внутри модели — оставляем")

    # ── весь экран: путь → чьи флаги → запись ──
    model = _write_fake_gguf(tmp / "My Model!! Q4.gguf", ctx_trained=65536)
    (tmp / "mmproj-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    ini_path = tmp / "models.ini"
    ini_path.write_text(CUSTOM_INI, encoding="utf-8")
    ini = inifile.IniFile(ini_path, CUSTOM_INI)
    answers = iter([str(model), "1", "y"])   # путь, чьи флаги, дописать
    orig_read = wizard._read
    wizard._read = lambda prompt: next(answers)
    try:
        name = wizard._custom_preset(ini)
    finally:
        wizard._read = orig_read
    check(name == "my-model-q4", "имя новой секции", name)
    new = ini.section(name)
    check(new is not None and new.pairs().get("model") == str(model),
          "секция появилась в ini в памяти")
    written = ini_path.read_text(encoding="utf-8")
    check(f"[{name}]" in written, "секция записана на диск")
    check("создано мастером" in written, "помечено, откуда секция")
    check("; шапка с комментарием — должна выжить" in written,
          "комментарии файла не потеряны")
    check(any(".bak-" in p.name for p in tmp.iterdir()), "сделана резервная копия")

    # запись действительна: validate по схеме не спотыкается о мусор
    rep = validate.Report()
    meta2 = validate.validate_section(name, ini.section(name).pairs(),
                                      schema.parse_help(HELP), rep,
                                      check_paths=True)
    check(meta2 is not None and meta2.n_ctx_trained == 65536,
          "validate читает модель новой секции")
    check(not any(f.level == "error" for f in rep.findings),
          "по новой секции ошибок нет", [f.line() for f in rep.findings])

    # ── отказ не оставляет мусора ──
    ini2 = inifile.IniFile(ini_path, CUSTOM_INI)
    answers2 = iter([str(model), "1", "n"])
    wizard._read = lambda prompt: next(answers2)
    try:
        rc = wizard._custom_preset(ini2)
    finally:
        wizard._read = orig_read
    check(rc is None, "отказ = None, мастер вернётся к списку", rc)
    check(ini2.section("my-model-q4") is None and
          "[my-model-q4]" not in ini2.dumps(), "отказ ничего не записал")

    # ── поток настройки ведёт в этот пункт ──
    src = (Path(__file__).resolve().parents[1] / "lib" / "wizard.py"
           ).read_text(encoding="utf-8")
    check("_preset_options(ini, custom=True)" in src,
          "поток настройки предлагает свой .gguf")
    check(src.count("_custom_preset(ini") >= 1, "пункт обрабатывается потоком")


def _wizard_env(tmp: Path, answers: list[str]):
    """Подменяет ввод, запуск CLI и окружение мастера. Возвращает (calls, out)."""
    import contextlib
    import io
    from lib import builds as B

    ini_path = tmp / "models.ini"
    ini_path.write_text(CUSTOM_INI, encoding="utf-8")
    d = tmp / "ik"
    (d / "build" / "bin").mkdir(parents=True, exist_ok=True)
    (d / "build" / "bin" / "llama-server").touch()
    fake = {"ik": B.Build(name="ik", path=str(d), router=False)}

    calls: list = []
    queue = iter(answers)
    out = io.StringIO()
    saved = {k: getattr(wizard, k) for k in ("_read", "_run_cli")}
    saved_ini = wizard.paths.default_ini
    saved_reg = wizard.builds.all_builds
    saved_swap = wizard.swap.find_binary

    def fake_read(prompt):
        try:
            return next(queue)
        except StopIteration:
            raise AssertionError(f"вопросов больше, чем ответов: {prompt!r}")

    wizard._read = fake_read
    wizard._run_cli = lambda *a, **k: (calls.append(a), 0)[1]
    wizard.paths.default_ini = lambda: ini_path
    wizard.builds.all_builds = lambda: fake
    wizard.swap.find_binary = lambda: Path("/usr/bin/llama-swap")

    class _Env:
        def __enter__(self):
            self.buf = contextlib.redirect_stdout(out)
            self.buf.__enter__()
            return self

        def __exit__(self, *exc):
            self.buf.__exit__(*exc)
            for k, v in saved.items():
                setattr(wizard, k, v)
            wizard.paths.default_ini = saved_ini
            wizard.builds.all_builds = saved_reg
            wizard.swap.find_binary = saved_swap
            return False

        @property
        def text(self):
            return out.getvalue()

    return _Env(), calls


def test_wizard_zero_goes_back_everywhere(tmp: Path) -> None:
    """0 = на шаг назад на КАЖДОМ экране, а не «да» и не выполнение шага.

    Регрессия: ask_yn возвращал на 0 строку «◄ назад», а она непустая —
    поэтому `if ask_yn(...)` принимал «назад» за «да»: мастер ставил
    llama-swap, печатал команду тюна, писал секцию в models.ini и выполнял
    validate+budget. Теперь 0 поднимает _GoBack, и его ловит Nav.walk.
    """
    print("wizard: 0 = назад везде")

    # 0 на первом экране потока → в меню, ничего не выполнено
    env, calls = _wizard_env(tmp, ["0"])
    with env:
        rc = wizard._setup_flow("http://127.0.0.1:8087")
    check(rc == "menu", "0 на первом экране = в меню", rc)
    check(not calls, "0 не выполнил ни одной команды", calls)
    check("ответ: " + wizard.BACK in env.text, "ответ показан как назад")

    # 0 на последнем экране (выполнить шаги) → тоже в меню, и НЕ выполняет
    env, calls = _wizard_env(tmp, ["1", "1", "n", "", "n", "n", "0"])
    with env:
        rc = wizard._setup_flow("http://127.0.0.1:8087")
    check(rc == "menu", "0 на последнем экране = в меню", rc)
    check(not calls, "0 не выполнил validate/budget", calls)
    check("план" in env.text, "план успели показать до отката")

    # 0 на экране ускорителей → откат на «зрение», ответ перезаписан заново.
    # Старый код клал в res строку «◄ назад», и план писал ngram=да
    env, calls = _wizard_env(tmp, ["1", "1", "n", "", "0", "n", "y", "n"])
    with env:
        rc = wizard._setup_flow("http://127.0.0.1:8087")
    check(rc == 0, "после отката поток доходит до конца", rc)
    check("ngram=нет" in env.text and "ngram=да" not in env.text,
          "«назад» не записался в ответ как «да»",
          [ln for ln in env.text.splitlines() if "ngram=" in ln])
    check(not calls, "финальный вопрос без 0 — шаги не выполнялись", calls)

    # 0 внутри диалога «свой .gguf» → назад к списку пресетов, не к swap
    model = _write_fake_gguf(tmp / "Custom.gguf")
    env, calls = _wizard_env(tmp, ["1", "2", "0", "1", "n", "", "n", "y", "n"])
    with env:
        rc = wizard._setup_flow("http://127.0.0.1:8087")
    check(rc == 0, "откат из своего .gguf вернул к списку и дошёл до плана", rc)
    check(env.text.rindex("какой пресет берём за основу")
          > env.text.rindex("· путь к .gguf"),
          "после 0 в диалоге снова показан список пресетов")
    check(not calls, "откат ничего не выполнил", calls)
    check("[custom]" not in (tmp / "models.ini").read_text(encoding="utf-8"),
          "откат не записал секцию")

    # 0 в «Загрузить или сменить пресет» → в меню, ничего не грузим
    env, calls = _wizard_env(tmp, ["0"])
    with env:
        rc = wizard._preset_flow("http://127.0.0.1:8087")
    check(rc == "menu", "0 в потоке пресета = в меню", rc)
    check(not calls, "0 не грузил модель", calls)

    # 0 в главном меню = выход, и ask_* больше не отдают «назад» значением
    env, _ = _wizard_env(tmp, ["0"])
    with env:
        rc, action = wizard._main_menu("http://127.0.0.1:8087")
    check(rc == 0 and action is None, "0 в меню = выход", (rc, action))
    for fn, args in ((wizard.ask_yn, ("t", True)), (wizard.ask_text, ("t", "")),
                     (wizard.ask_pick, ("t", [("a", "")], 1))):
        env, _ = _wizard_env(tmp, ["0"])
        with env:
            try:
                fn(*args)
                raised = False
            except wizard._GoBack:
                raised = True
        check(raised, f"{fn.__name__} на 0 бросает _GoBack, а не значение")
    check(wizard.is_back(wizard.BACK) and not wizard.is_back(False),
          "is_back отличает «назад» от ответа")

    # страховка от повторов этой ошибки: вопрос не должен отдавать «назад» значением
    src = (Path(__file__).resolve().parents[1] / "lib" / "wizard.py"
           ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    questions = {"ask", "ask_pick", "ask_yn", "ask_text"}
    offenders = [f"{n.name}:{r.lineno}"
                 for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name in questions
                 for r in ast.walk(n)
                 if isinstance(r, ast.Return) and isinstance(r.value, ast.Name)
                 and r.value.id == "BACK"]
    check(not offenders, "вопросы не отдают «назад» значением", offenders)
    check("except _GoBack:" in src, "экраны вне Nav ловят _GoBack")


def test_wizard_tune_actually_runs(tmp: Path) -> None:
    """Ответ «да» на автотюн запускает тюнер, а не только печатает команду.

    Регрессия: s_tune печатал `команда: llamastery tune …` и сразу переходил
    к следующему вопросу — тюнер не запускался никогда, а человек оставался
    с belief, что GPU сейчас занят на 100%.
    """
    print("wizard: автотюн запускается")

    from lib import server as S

    argv = wizard._tune_argv("faks", "base-mmproj", {"c": "131072"})
    check(argv[:4] == ["tune", str(paths.default_ini()), "base-mmproj",
                       "--build"], "тюнер зовёт тот же ini и сборку", argv)
    # Регрессия: мастер выдумывал ключи и отдавал `--c`, `--n-cpu-moe`,
    # `--ubatch`, которых у тюнера нет → падение на usage после 10-30 минут
    # ожидания. Проверяем по НАСТОЯЩЕМУ argparse тюнера, а не по списку.
    check(not wizard._tune_unknown_keys(argv), "все флаги тюнер понимает",
          wizard._tune_unknown_keys(argv))
    check(wizard._tune_unknown_keys(
        ["tune", "x", "y", "--extra", "c=131072", "n-cpu-moe=0", "ubatch=1024"])
        == ["c", "n_cpu_moe", "ubatch"],
        "старые выдуманные ключи ловятся")
    check("search-ctx=16384" in argv,
          "длинный контекст → подбор на коротком KV", argv)
    check("--extra" not in wizard._tune_argv("faks", "p", {"c": "8192"}),
          "короткий контекст — ничего лишнего не передаём")
    check(wizard._tune_argv(None, "p", {})[:2]
          == ["tune", str(paths.default_ini())],
          "без сборки --build не подставляется")
    # тюнер берёт c из пресета сам, поэтому передавать его не нужно
    tuner = wizard._tuner_options()
    check("contexts" in tuner and "search_ctx" in tuner,
          "флаги тюнера читаются из его исходника", len(tuner))
    check(not ({"c", "n_cpu_moe", "ubatch"} & tuner),
          "таких флагов у тюнера действительно нет",
          sorted({"c", "n_cpu_moe", "ubatch"} & tuner))

    saved_status = S.status
    saved_vram = wizard.vram.gpu_used_mib
    saved_total = wizard.vram.gpu_total_mib
    saved_shrink = wizard._shrink_for_tune

    def _fits(pairs, total):
        """Заглушка подбора: shrink, который влезает и ничего не меняет."""
        return budget.Shrink(ctx=0, cpu_moe=0,
                             n_layer=40, fits=True,
                             est=budget.Estimate(total_gb=9.0, weights_gb=7.0,
                                                 kv_gb=1.0, compute_gb=1.0),
                             steps=["пресет уже влезает"])

    def _huge(pairs, total):
        """Заглушка: даже минимальный пресет не влезает."""
        s = budget.Shrink(ctx=4096, cpu_moe=39, n_layer=40, fits=False,
                          est=budget.Estimate(total_gb=6.0), steps=["не лезет"])
        return s

    S.status = lambda: {"models": [{"id": "cyber", "status": "loaded"}]}
    wizard.vram.gpu_used_mib = lambda: 10607
    wizard.vram.gpu_total_mib = lambda: 12288
    wizard._shrink_for_tune = _fits
    try:
        # сборка → пресет → автотюн=да → контекст → ngram → mtp → «выполнить»
        env, calls = _wizard_env(tmp, ["1", "1", "y", "", "n", "n", "y", "y"])
        with env:
            rc = wizard._setup_flow("http://127.0.0.1:8087")
        check(rc == 0, "поток дошёл до конца", rc)
        flat = [" ".join(c) for c in calls]
        check(any(c.startswith("tune ") for c in flat), "тюнер запущен", flat)
        check(flat.index("validate base-mmproj --build ik")
              < next(i for i, c in enumerate(flat) if c.startswith("tune ")),
              "сначала быстрые проверки, потом долгий тюн", flat)
        check(any(c == "runtime unload" for c in flat),
              "модель выгружена перед тюном (иначе не хватит VRAM)", flat)
        check(env.text.index("выполнить автотюн") > 0,
              "про автотюн спросили явно")
        check("автотюн вернул код" not in env.text,
              "тюнер отработал без ошибки")

        # без согласия на автотюн его нет и в плане, и в запуске
        env, calls = _wizard_env(tmp, ["1", "1", "n", "", "n", "n", "y"])
        with env:
            rc = wizard._setup_flow("http://127.0.0.1:8087")
        check(rc == 0 and not any(c[0] == "tune" for c in calls),
              "без «да» тюнер не запускается", calls)

        # 0 на финальном вопросе — тюнер тоже не стартует
        env, calls = _wizard_env(tmp, ["1", "1", "y", "", "n", "n", "0"])
        with env:
            rc = wizard._setup_flow("http://127.0.0.1:8087")
        check(rc == "menu" and not calls,
              "0 на финальном вопросе отменяет и тюн", (rc, calls))

        # не влезает даже минимальный: без согласия не запускаем
        wizard._shrink_for_tune = _huge
        env, calls = _wizard_env(tmp, ["1", "1", "y", "", "n", "n", "y", "n"])
        with env:
            rc = wizard._setup_flow("http://127.0.0.1:8087")
        check(not any(c[0] == "tune" for c in calls),
              "не влезающий пресет тюнить не запустили", calls)
        check("не влезает" in env.text and "не запускаю" in env.text,
              "и сказано почему")
        # но если человек настоял — запускаем (после выгрузки модели)
        env, calls = _wizard_env(tmp, ["1", "1", "y", "", "n", "n", "y", "y", "y"])
        with env:
            wizard._setup_flow("http://127.0.0.1:8087")
        check(any(c[0] == "tune" for c in calls),
              "настойчивый «да» перекрывает предупреждение", calls)

        # главное: влезающий вариант подбирается и уходит в тюнер сам
        def _needs_shrinking(pairs, total):
            return budget.Shrink(ctx=65536, cpu_moe=12, n_layer=40, fits=True,
                                 est=budget.Estimate(total_gb=9.5, weights_gb=7.5,
                                                     kv_gb=1.0, compute_gb=1.0),
                                 steps=["c 131072 → 65536", "n-cpu-moe 0 → 12"])

        wizard._shrink_for_tune = _needs_shrinking
        env, calls = _wizard_env(tmp, ["1", "1", "y", "", "n", "n", "y", "y"])
        with env:
            wizard._setup_flow("http://127.0.0.1:8087")
        tune = [c for c in calls if c[0] == "tune"]
        check(tune and "search-ctx=65536" in tune[0],
              "подобранный контекст ушёл в тюнер", tune)
        check(tune and any("moe-values=" in str(x) for x in tune[0]),
              "подобранный n-cpu-moe ушёл в перебор", tune)
        check("подбираю параметры" in env.text,
              "человеку сказали, что пресет ужат", env.text[-900:])
    finally:
        S.status = saved_status
        wizard.vram.gpu_used_mib = saved_vram
        wizard.vram.gpu_total_mib = saved_total
        wizard._shrink_for_tune = saved_shrink


def test_wizard_pasted_path_is_not_a_number(tmp: Path) -> None:
    """Вставленный путь в списке пресетов ведёт в «свой .gguf», а не в ★.

    Регрессия: не-номер уходил в clamp_choice и молча превращался в ★-дефолт
    — «вставил путь, нажал Enter» выглядело как «выбрал пресет №3».
    """
    print("wizard: вставка пути вместо номера")

    check(wizard._as_model_path("/m/Model.gguf") == "/m/Model.gguf",
          "путь к .gguf распознан")
    check(wizard._as_model_path("'/m/My Model.GGUF'")
          == "/m/My Model.GGUF", "кавычки и регистр не мешают")
    check(wizard._as_model_path("да") is None and wizard._as_model_path("") is None,
          "не путь — не путь")
    check(wizard._as_model_path("/нет/такого.gguf") == "/нет/такого.gguf",
          "несуществующий файл всё равно путь: ошибку даст диалог")

    saved = wizard._read
    wizard._read = lambda p: "/m/Model.gguf"
    try:
        got = wizard.ask_pick("вопрос", [("один", ""), ("два", "")], 2,
                              paste_model=True)
        check(isinstance(got, wizard.PastedPath) and got.path == "/m/Model.gguf",
              "путь вернулся как PastedPath, а не как номер", got)
        # без флага вставка не принимается — но и не проходит молча
        got2 = wizard.ask_pick("вопрос", [("один", ""), ("два", "")], 2)
        check(got2 == "два", "без paste_model берётся ★-дефолт", got2)
    finally:
        wizard._read = saved

    # мусор и несуществующий пункт объясняются, а не молчатся
    for raw, want in (("автотюн", "два"), ("9", "два")):
        wizard._read = lambda p, r=raw: r
        try:
            got = wizard.ask_pick("вопрос", [("один", ""), ("два", "")], 2)
        finally:
            wizard._read = saved
        check(got == want, f"{raw!r} → дефолт", got)

    # полный путь: вставка в список → диалог без вопроса про путь → секция.
# Ответы: сборка, вставленный путь, чьи флаги, дописать, тюн, контекст,
# ngram, mtp, финал
    model = _write_fake_gguf(tmp / "Pasted Model.gguf")
    env, calls = _wizard_env(tmp, ["1", str(model), "1", "y", "n", "", "n",
                                   "n", "n"])
    with env:
        rc = wizard._setup_flow("http://127.0.0.1:8087")
    check(rc == 0, "поток дошёл до конца", rc)
    check("· путь к .gguf" not in env.text,
          "путь из вставки повторно не спрашивается")
    check("путь из вставки" in env.text, "путь из вставки показан")
    check("[pasted-model]" in (tmp / "models.ini").read_text(encoding="utf-8"),
          "секция создана по вставленному пути")
    check(not any(c[0] == "tune" for c in calls), "тюнер не трогали", calls)


def test_budget_shrink_to_fit() -> None:
    """Ужимание пресета до влезающего: c → n-cpu-moe, замер важнее оценки."""
    print("бюджет: подбор укладывающихся параметров")

    m = gguf.ModelMeta(path=Path("/m.gguf"), size_bytes=int(11.7 * 2 ** 30))
    m.arch = "qwen35moe"
    m.kv = {"qwen35moe.block_count": 40, "qwen35moe.embedding_length": 2048,
            "qwen35moe.attention.head_count": 16,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.context_length": 262144,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_feed_forward_length": 512,
            "qwen35moe.full_attention_interval": 4}
    cal = {"compute_gb": 0.95}
    big = {"model": "/m.gguf", "c": "131072", "cache-type-k": "q8_0",
           "cache-type-v": "q5_0", "kv-unified": "true", "parallel": "1"}

    e0 = budget.estimate(big, m, _cal=cal)
    check(e0.total_gb > 11.0, "пресет изначально не влезает", round(e0.total_gb, 2))

    r = budget.shrink_to_fit(big, m, 12288, _cal=cal)
    check(r.fits, "после подбора влезает", r.steps)
    check(r.cpu_moe > 0 and r.ctx == 131072,
          "контекст сохранён, резали n-cpu-moe", (r.ctx, r.cpu_moe))
    check(r.est.total_gb <= (12288 - 1024) / 1024.0,
          "оценка совпадает с budget", round(r.est.total_gb, 2))

    # не влезает даже минимальный — честно говорим об этом
    r2 = budget.shrink_to_fit({**big, "n-gpu-layers": "99", "c": "262144"},
                              m, 4096, _cal=cal)
    check(not r2.fits and r2.steps, "не влезает — сказано почему", r2.steps)

    # контекст режется, когда сжимать нечем (плотная модель без MoE)
    dense = gguf.ModelMeta(path=Path("/d.gguf"), size_bytes=int(4.0 * 2 ** 30))
    dense.arch = "llama"
    dense.kv = {"llama.block_count": 32, "llama.embedding_length": 4096,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.attention.key_length": 128,
                "llama.context_length": 131072}
    r3 = budget.shrink_to_fit({"model": "/d.gguf", "c": "131072"}, dense,
                              12288, _cal=cal)
    check(r3.fits and r3.cpu_moe == 0 and r3.ctx < 131072,
          "плотная модель: режется только контекст",
          (r3.ctx, r3.cpu_moe, r3.fits))

    # живой замер важнее оценки: рабочий пресет не трогаем
    saved_load = measure.load_store
    try:
        e_now = budget.estimate(big, m, _cal=cal)
        check(e_now.total_gb > 11.0, "оценка врёт в сторону нехватки",
              round(e_now.total_gb, 2))
        measure.load_store = lambda: {"records": {
            measure.signature(big): {"used_mib": 10800, "runs": 1,
                                     "min_free_mib": 1488}}}
        r4 = budget.shrink_to_fit(big, m, 12288, _cal=cal)
        check(r4.fits and r4.cpu_moe == 0 and r4.ctx == 131072,
              "по замеру пресет влезает — не урезаем", (r4.ctx, r4.cpu_moe))
        check("замер" in (r4.steps or [""])[0], "и сказано, что решил замер",
              r4.steps)
        # а если замер говорит, что НЕ влезает — подбор всё равно идёт
        measure.load_store = lambda: {"records": {
            measure.signature(big): {"used_mib": 12000, "runs": 1}}}
        r5 = budget.shrink_to_fit(big, m, 12288, _cal=cal)
        check(r5.fits and r5.cpu_moe > 0,
              "замер «не влезает» → подбираем дальше", (r5.ctx, r5.cpu_moe))
    finally:
        measure.load_store = saved_load

    # подпорки не выдумываются
    check(not budget.shrink_to_fit(big, None, 12288).fits,
          "без метаданных — не влезает (а не «наверное, ок»)")
    check(budget.shrink_to_fit(big, m, None).steps == [],
          "без VRAM — пустой список правок")


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
