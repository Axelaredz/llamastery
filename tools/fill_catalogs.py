#!/usr/bin/env python3
"""Заполнить каталоги переводов из таблиц ниже. Одноразовая начинка.

Держится отдельно от `lib/locale/*.json`, чтобы перевод читался как таблица
«слева исходник, справа перевод», а не как простыня JSON с запятыми. Ключи
сверяются с кодом скриптом `extract_i18n.py --check`.
"""

import json
import pathlib

LOCALE = pathlib.Path(__file__).resolve().parent.parent / "lib" / "locale"

EN = {
 "\nтребуют внимания: {0}": "\nneed attention: {0}",
 "обновление и пересборка — решение человека:": "updating and rebuilding is the human's call:",
 "  git -C <путь> pull && cmake --build build -j$(nproc)": "  git -C <path> pull && cmake --build build -j$(nproc)",
 "затем: llamastery schema <сборка> --refresh": "then: llamastery schema <build> --refresh",
 "источник: {0} ({1})": "source: {0} ({1})",
 "флагов:   {0}   mtime: {1}": "flags:   {0}   mtime: {1}",
 "пресет: {0}": "preset: {0}",
 "сборка: {0}   бинарь: {1}": "build: {0}   binary: {1}",
 "жду загрузки {0}...": "waiting for {0} to load...",
 "таймаут загрузки, статус {0}. Лог: {1}": "load timeout, status {0}. Log: {1}",
 "\nпадение записано в журнал ({0}): {1}": "\ncrash recorded in journal ({0}): {1}",
 "  теперь validate и presets annotate будут предупреждать о {0}": "  validate and presets annotate will now warn about {0}",
 "  прошёл замер — сбросить: llamastery crashes forget {0}": "  measured successfully — clear it: llamastery crashes forget {0}",
 "найдено в логе:": "found in the log:",
 "\nбыло: {0} GiB при ub={1}": "\nwas: {0} GiB at ub={1}",
 "стало: {0} GiB при ub={1}  [запись сохранена]": "now: {0} GiB at ub={1}  [record saved]",
 "проверь: llamastery budget --ini <файл> <секция>": "check: llamastery budget --ini <file> <section>",
 "падений записано: {0}\n": "crashes recorded: {0}\n",
 "\nжурнал: {0}": "\njournal: {0}",
 "сбросить после успешного замера: llamastery crashes forget <пресет>": "clear after a successful measurement: llamastery crashes forget <preset>",
 "неизвестный verb: {0}": "unknown verb: {0}",
 "учтено прогонов: {0}": "runs taken into account: {0}",
 "compute_gb = {0} при ubatch={1}  (разброс {2}..{3} GiB)": "compute_gb = {0} at ubatch={1}  (spread {2}..{3} GiB)",
 "записано: {0}": "written: {0}",
 "\nхранилище замеров: {0}": "\nmeasurement store: {0}",
 "замеров из tune-results: {0}": "measurements from tune-results: {0}",
 "всего записей: {0}": "total records: {0}",
 "из них со скоростью: {0}": "of those with a speed figure: {0}",
 "запуск:": "run:",
 "рабочий каталог: {0}": "working directory: {0}",
 "пути": "paths",
 "\nсборки": "\nbuilds",
 "\nсхема флагов: {0} (через {1})": "\nflag schema: {0} (via {1})",
 "\nменеджеры на PATH": "\nmanagers on PATH",
 "\nзамеры": "\nmeasurements",
 "  записей в measurements.json: {0}": "  records in measurements.json: {0}",
 "  каталогов tune-results: {0}": "  tune-results directories: {0}",
 "  калибровка compute_gb: {0}": "  compute_gb calibration: {0}",
 "  актуальность сборок: {0}": "  build freshness: {0}",
 "порядок проверки (по убыванию ожидаемого прироста):": "check order (by expected gain, descending):",
 "реестр пуст. Заполнить: llamastery builds detect --apply": "registry is empty. Populate it: llamastery builds detect --apply",
 "\nвсе сборки актуальны": "\nall builds are up to date",
 "добавлен: {0} -> {1}": "added: {0} -> {1}",
 "ошибка схемы: {0}": "schema error: {0}",
 "\nизменено секций: {0} из {1}": "\nsections changed: {0} of {1}",
 "источник: {0}": "source: {0}",
 "цель:     {0}": "target:     {0}",
 "добавлено: {0}  обновлено: {1}  пропущено: {2}": "added: {0}  updated: {1}  skipped: {2}",
 "пресет {0} не найден": "preset {0} not found",
 "подходит {0} секций, укажи точное имя:": "{0} sections match, give the exact name:",
 "\nотказ: пресет не проходит проверку. Исправить или --force.": "\nrefused: the preset fails validation. Fix it or use --force.",
 "внимание: {0} без роутера (--models-preset не работает), пресет транслируется в argv и сервер перезапускается": "note: {0} has no router (--models-preset is unsupported), the preset is translated into argv and the server restarts",
 "в пресете нет моделей ({0})": "the preset has no models ({0})",
 "секции {0} нет среди моделей роутера": "section {0} is not among the router's models",
 "уже загружена: {0}": "already loaded: {0}",
 "\n(dry-run) загрузится: {0}": "\n(dry-run) will load: {0}",
 "уже грузится: {0}, жду": "already loading: {0}, waiting",
 "готово: {0} — {1}": "done: {0} — {1}",
 "не загрузилась (OOM или ошибка). Проверь n-gpu-layers / n-cpu-moe в {0}": "failed to load (OOM or error). Check n-gpu-layers / n-cpu-moe in {0}",
 "в логе нет ни одной строки «CUDA0 compute buffer size»": "the log has not a single «CUDA0 compute buffer size» line",
 "  лог: {0}": "  log: {0}",
 "  такие строки сервер печатает при загрузке модели; в режиме": "  the server prints such lines when a model loads; in router",
 "  роутера отчёт инстанса иногда не попадает в общий лог — тогда": "  mode an instance report sometimes misses the shared log — then",
 "  запусти одиночный сервер (llamastery runtime start --preset ...) и повтори": "  start a single server (llamastery runtime start --preset ...) and retry",
 "  ubatch {0}: {1} MiB ({2} сообщ.)": "  ubatch {0}: {1} MiB ({2} messages)",
 "падений не записано": "no crashes recorded",
 "  журнал: {0}": "  journal: {0}",
 "    {0} раз, последний {1}, сборка {2}{3}": "    {0} times, last {1}, build {2}{3}",
 "роутер:    {0}   {1}": "router:    {0}   {1}",
 "порт:      {0}": "port:      {0}",
 "сборка:    {0}": "build:     {0}",
 "пресет:    {0}": "preset:    {0}",
 "лог:       {0}": "log:       {0}",
 "сервер жив на {0}": "server is up at {0}",
 "внимание: --build не указан, схема взята у сборки по умолчанию ({0}). Проверка флагов другой сборки будет ложной.": "note: --build is not given, the schema came from the default build ({0}). Checking another build's flags would be meaningless.",
 "проверено секций: {0}   схема: {1}": "sections checked: {0}   schema: {1}",
 "\nошибок: {0}  предупреждений: {1}": "\nerrors: {0}  warnings: {1}",
 "нет секций для расчёта": "no sections to compute",
 "\nслот одновременно (models-max = {0}), резерв {1} MiB → доступно {2} MiB": "\nslots at once (models-max = {0}), reserve {1} MiB → {2} MiB available",
 "\n(прогноз по слотам не построен: неизвестен объём VRAM)": "\n(slot forecast not built: VRAM size unknown)",
 "\nразбор «{0}»:" : "\nbreakdown of «{0}»:",
 "укажи секции явно (измерения по подписям не разворачиваются в секции)": "name the sections explicitly (measurements by label do not expand into sections)",
 "{0}: структура {1} GiB, факт {2} GiB, остаток {3} GiB": "{0}: structure {1} GiB, actual {2} GiB, remainder {3} GiB",
 "нечего калибровать": "nothing to calibrate",
 "нужен nvidia-smi": "nvidia-smi is required",
 "подходящих прогонов не найдено (нужны ok=True и min_observed_free_mib)": "no suitable runs found (need ok=True and min_observed_free_mib)",
 "сервер не запущен: llamastery runtime start": "server is not running: llamastery runtime start",
 "в VRAM ничего не загружено: llamastery load <пресет>": "nothing loaded into VRAM: llamastery load <preset>",
 "  проб из прогретого KV: {0} (в prefill не попали)": "  probes from warm KV: {0} (excluded from prefill)",
 "в VRAM ничего не загружено — снимать нечего": "nothing loaded into VRAM — nothing to measure",
 "  записано: vram {0} GiB [замер], свободно {1} MiB": "  recorded: vram {0} GiB [measured], free {1} MiB",
 "(только просмотр; --apply чтобы сохранить)": "(preview only; --apply to save)",
}

if __name__ == "__main__":
    # Ключ должен дословно совпадать со строкой в коде, включая пробелы и
    # перевод строки. Опечатка в ключе не падает: она молча даёт перевод,
    # который никогда не подставится, и строка останется на русском навсегда.
    # Поэтому сверяемся с кодом и отказываемся писать файл при расхождении.
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from extract_i18n import collect

    code_keys, _ = collect()
    unknown = [k for k in EN if k not in code_keys]
    if unknown:
        print(f"! {len(unknown)} ключей не найдено в коде — файл не записан:")
        for k in unknown:
            print(f"    {k!r}")
        raise SystemExit(1)

    p = LOCALE / "en.json"
    p.write_text(json.dumps(EN, ensure_ascii=False, indent=1) + "\n",
                 encoding="utf-8")
    print(f"{p}: {len(EN)} ключей, все совпали с кодом")