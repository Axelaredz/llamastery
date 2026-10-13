#!/usr/bin/env python3
"""Немецкий каталог переводов. Сверка ключей с кодом обязательна."""

import json
import pathlib
import sys

LOCALE = pathlib.Path(__file__).resolve().parent.parent / "lib" / "locale"

DE = {
 "\nтребуют внимания: {0}": "\nbrauchen Aufmerksamkeit: {0}",
 "обновление и пересборка — решение человека:": "Update und Neubau sind die Entscheidung des Menschen:",
 "  git -C <путь> pull && cmake --build build -j$(nproc)": "  git -C <pfad> pull && cmake --build build -j$(nproc)",
 "затем: llamastery schema <сборка> --refresh": "danach: llamastery schema <build> --refresh",
 "источник: {0} ({1})": "Quelle: {0} ({1})",
 "флагов:   {0}   mtime: {1}": "Flags:   {0}   mtime: {1}",
 "пресет: {0}": "Preset: {0}",
 "сборка: {0}   бинарь: {1}": "Build: {0}   Binär: {1}",
 "жду загрузки {0}...": "warte auf Ladevorgang {0}...",
 "таймаут загрузки, статус {0}. Лог: {1}": "Zeitüberschreitung beim Laden, Status {0}. Log: {1}",
 "\nпадение записано в журнал ({0}): {1}": "\nAbsturz im Journal vermerkt ({0}): {1}",
 "  теперь validate и presets annotate будут предупреждать о {0}": "  validate und presets annotate warnen jetzt vor {0}",
 "  прошёл замер — сбросить: llamastery crashes forget {0}": "  Messung bestanden — zurücksetzen: llamastery crashes forget {0}",
 "найдено в логе:": "im Log gefunden:",
 "\nбыло: {0} GiB при ub={1}": "\nvorher: {0} GiB bei ub={1}",
 "стало: {0} GiB при ub={1}  [запись сохранена]": "jetzt: {0} GiB bei ub={1}  [Eintrag gespeichert]",
 "проверь: llamastery budget --ini <файл> <секция>": "prüfen: llamastery budget --ini <datei> <abschnitt>",
 "падений записано: {0}\n": "Abstürze vermerkt: {0}\n",
 "\nжурнал: {0}": "\nJournal: {0}",
 "сбросить после успешного замера: llamastery crashes forget <пресет>": "nach erfolgreicher Messung zurücksetzen: llamastery crashes forget <preset>",
 "неизвестный verb: {0}": "unbekanntes Verb: {0}",
 "учтено прогонов: {0}": "berücksichtigte Läufe: {0}",
 "compute_gb = {0} при ubatch={1}  (разброс {2}..{3} GiB)": "compute_gb = {0} bei ubatch={1}  (Streuung {2}..{3} GiB)",
 "записано: {0}": "geschrieben: {0}",
 "\nхранилище замеров: {0}": "\nMessspeicher: {0}",
 "замеров из tune-results: {0}": "Messungen aus tune-results: {0}",
 "всего записей: {0}": "Einträge insgesamt: {0}",
 "из них со скоростью: {0}": "davon mit Geschwindigkeit: {0}",
 "запуск:": "Lauf:",
 "рабочий каталог: {0}": "Arbeitsverzeichnis: {0}",
 "пути": "Pfade",
 "\nсборки": "\nBuilds",
 "\nсхема флагов: {0} (через {1})": "\nFlag-Schema: {0} (über {1})",
 "\nменеджеры на PATH": "\nManager im PATH",
 "\nзамеры": "\nMessungen",
 "  записей в measurements.json: {0}": "  Einträge in measurements.json: {0}",
 "  каталогов tune-results: {0}": "  tune-results-Verzeichnisse: {0}",
 "  калибровка compute_gb: {0}": "  compute_gb-Kalibrierung: {0}",
 "  актуальность сборок: {0}": "  Aktualität der Builds: {0}",
 "порядок проверки (по убыванию ожидаемого прироста):": "Prüfreihenfolge (nach erwartetem Gewinn, absteigend):",
 "реестр пуст. Заполнить: llamastery builds detect --apply": "Registry ist leer. Füllen: llamastery builds detect --apply",
 "\nвсе сборки актуальны": "\nalle Builds sind aktuell",
 "добавлен: {0} -> {1}": "hinzugefügt: {0} -> {1}",
 "ошибка схемы: {0}": "Schemafehler: {0}",
 "\nизменено секций: {0} из {1}": "\ngeänderte Abschnitte: {0} von {1}",
 "источник: {0}": "Quelle: {0}",
 "цель:     {0}": "Ziel:     {0}",
 "добавлено: {0}  обновлено: {1}  пропущено: {2}": "hinzugefügt: {0}  aktualisiert: {1}  übersprungen: {2}",
 "пресет {0!r} не найден": "Preset {0!r} nicht gefunden",
 "подходит {0} секций, укажи точное имя:": "{0} Abschnitte passen, exakten Namen angeben:",
 "\nотказ: пресет не проходит проверку. Исправить или --force.": "\nAbgelehnt: das Preset besteht die Prüfung nicht. Korrigieren oder --force.",
 "внимание: {0} без роутера (--models-preset не работает), пресет транслируется в argv и сервер перезапускается": "Hinweis: {0} ohne Router (--models-preset wird nicht unterstützt), das Preset wird in argv übersetzt und der Server neu gestartet",
 "в пресете нет моделей ({0})": "im Preset sind keine Modelle ({0})",
 "секции {0!r} нет среди моделей роутера": "Abschnitt {0!r} ist nicht unter den Router-Modellen",
 "уже загружена: {0}": "bereits geladen: {0}",
 "\n(dry-run) загрузится: {0}": "\n(dry-run) wird geladen: {0}",
 "уже грузится: {0}, жду": "wird bereits geladen: {0}, warte",
 "готово: {0} — {1}": "fertig: {0} — {1}",
 "не загрузилась (OOM или ошибка). Проверь n-gpu-layers / n-cpu-moe в {0}": "Laden fehlgeschlagen (OOM oder Fehler). Prüfe n-gpu-layers / n-cpu-moe in {0}",
 "в логе нет ни одной строки «CUDA0 compute buffer size»": "das Log enthält keine einzige Zeile «CUDA0 compute buffer size»",
 "  лог: {0}": "  Log: {0}",
 "  такие строки сервер печатает при загрузке модели; в режиме": "  solche Zeilen druckt der Server beim Laden eines Modells; im",
 "  роутера отчёт инстанса иногда не попадает в общий лог — тогда": "  Router-Modus erreicht der Instanzbericht das gemeinsame Log",
 "  запусти одиночный сервер (llamastery runtime start --preset ...) и повтори": "  manchmal nicht — dann starte einen einzelnen Server (llamastery runtime start --preset ...) und wiederhole",
 "  ubatch {0}: {1} MiB ({2} сообщ.)": "  ubatch {0}: {1} MiB ({2} Nachr.)",
 "падений не записано": "keine Abstürze vermerkt",
 "  журнал: {0}": "  Journal: {0}",
 "    {0} раз, последний {1}, сборка {2}{3}": "    {0}-mal, zuletzt {1}, Build {2}{3}",
 "роутер:    {0}   {1}": "Router:    {0}   {1}",
 "порт:      {0}": "Port:      {0}",
 "сборка:    {0}": "Build:     {0}",
 "пресет:    {0}": "Preset:    {0}",
 "лог:       {0}": "Log:       {0}",
 "сервер жив на {0}": "Server läuft auf {0}",
 "внимание: --build не указан, схема взята у сборки по умолчанию ({0}). Проверка флагов другой сборки будет ложной.": "Hinweis: --build ist nicht angegeben, das Schema stammt vom Standard-Build ({0}). Die Flags eines anderen Builds zu prüfen wäre sinnlos.",
 "проверено секций: {0}   схема: {1}": "geprüfte Abschnitte: {0}   Schema: {1}",
 "\nошибок: {0}  предупреждений: {1}": "\nFehler: {0}  Warnungen: {1}",
 "нет секций для расчёта": "keine Abschnitte zum Berechnen",
 "\nслот одновременно (models-max = {0}), резерв {1} MiB → доступно {2} MiB": "\nSlots gleichzeitig (models-max = {0}), Reserve {1} MiB → {2} MiB verfügbar",
 "\n(прогноз по слотам не построен: неизвестен объём VRAM)": "\n(Slot-Vorhersage nicht erstellt: VRAM-Größe unbekannt)",
 "\nразбор «{0}»:": "\nAufschlüsselung «{0}»:",
 "укажи секции явно (измерения по подписям не разворачиваются в секции)": "Abschnitte explizit angeben (Messungen nach Beschriftung werden nicht zu Abschnitten ausgewertet)",
 "{0}: структура {1} GiB, факт {2} GiB, остаток {3} GiB": "{0}: Struktur {1} GiB, tatsächlich {2} GiB, Rest {3} GiB",
 "нечего калибровать": "nichts zu kalibrieren",
 "нужен nvidia-smi": "nvidia-smi wird benötigt",
 "подходящих прогонов не найдено (нужны ok=True и min_observed_free_mib)": "keine passenden Läufe gefunden (nötig sind ok=True und min_observed_free_mib)",
 "сервер не запущен: llamastery runtime start": "Server läuft nicht: llamastery runtime start",
 "в VRAM ничего не загружено: llamastery load <пресет>": "nichts in die VRAM geladen: llamastery load <preset>",
 "  проб из прогретого KV: {0} (в prefill не попали)": "  Proben aus warmem KV: {0} (nicht im Prefill gewertet)",
 "в VRAM ничего не загружено — снимать нечего": "nichts in die VRAM geladen — nichts zu messen",
 "  записано: vram {0} GiB [замер], свободно {1} MiB": "  gespeichert: vram {0} GiB [gemessen], frei {1} MiB",
 "(только просмотр; --apply чтобы сохранить)": "(nur Ansicht; --apply zum Speichern)",
 "язык: {0}": "Sprache: {0}",
 "доступно: {0}": "verfügbar: {0}",
 "сбросить: export LLAMASTERY_LANG=ru   (или LANG=ru_RU.UTF-8)": "zurücksetzen: export LLAMASTERY_LANG=ru   (oder LANG=ru_RU.UTF-8)",
 "конфиг": "Konfig", "состояние": "Zustand", "кэш": "Cache", "пресет": "Preset",
 "ok": "ok", "нет (создастся)": "fehlt (wird angelegt)", "НЕТ ФАЙЛА": "DATEI FEHLT",
 "VRAM на карте: {0} MiB": "VRAM auf der Karte: {0} MiB",
 "VRAM: неизвестно (нет nvidia-smi)": "VRAM: unbekannt (kein nvidia-smi)",
 "compute buffer (калибровка): {0} GiB": "Compute-Buffer (Kalibrierung): {0} GiB",
 "   ← НЕ КАЛИБРОВАНО, оценка занижена": "   ← NICHT KALIBRIERT, Schätzung zu niedrig",
 "веса": "Gewichte", "прочее": "Sonstige", "всего": "Gesamt", "tg": "tg",
 "  {0}. {1:<14} {2:<18} приоритет {3}{4}": "  {0}. {1:<14} {2:<18} Priorität {3}{4}",
 "  влезает: {0}  ({1:.2f} GiB, {2})": "  passt: {0}  ({1:.2f} GiB, {2})",
 "  не влезает:": "  passt nicht:",
}

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from extract_i18n import collect
    code, _ = collect()
    unknown = [k for k in DE if k not in code]
    if unknown:
        print(f"! {len(unknown)} ключей нет в коде:")
        for k in unknown:
            print("   ", repr(k))
        raise SystemExit(1)
    p = LOCALE / "de.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for k, v in DE.items():
        if not data.get(k):
            data[k] = v
            n += 1
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                 encoding="utf-8")
    done = sum(1 for v in data.values() if v)
    print(f"de.json: добавлено {n}, заполнено {done}/{len(data)}")