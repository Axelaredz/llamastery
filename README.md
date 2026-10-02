# llamastery

Уложи модель в VRAM с первого раза: проверка пресетов, честный прогноз памяти, замер факта.

`llamastery` — CLI и скилл для агента вокруг `llama-server`. Знает твои сборки и форки
(`faks`, `upstream`, `ik_llama`, `xing4`), проверяет пресет по схеме **именно той** сборки,
считает, влезет ли он в видеокарту, грузит его и меряет реальную скорость на длинном контексте.
Только стандартная библиотека Python 3.11+, зависимостей нет.

[English](#llamastery-en) · [中文](#llamastery-中文)

## Навигация

- [За 60 секунд](#за-60-секунд) · [Кому это нужно](#кому-это-нужно) · [Как это работает](#как-это-работает)
- [Быстрый старт](#быстрый-старт) · [Команды по задачам](#команды-по-задачам) · [Что внутри](#что-внутри)
- [Документация](#документация) · [Метки](#метки) · [Установка](#установка)
- [Переменные окружения](#переменные-окружения) · [Гарантии](#гарантии) · [Тесты](#тесты)
- [За 60 секунд](#за-60-секунд) · [Быстрый старт](#быстрый-старт) · [Гарантии](#гарантии)
- [In 60 seconds](#in-60-seconds) · [Quick start](#quick-start) · [Commands by task](#commands-by-task)
- [Whats inside](#whats-inside) · [Documentation](#documentation) · [Tests](#tests)
- [60秒速览](#60秒速览) · [快速开始](#快速开始) · [按任务查命令](#按任务查命令)
- [目录结构](#目录结构) · [文档](#文档) · [测试](#测试)
- [llamastery](#llamastery) · [llamastery (EN)](#llamastery-en) · [llamastery (中文)](#llamastery-中文)

## За 60 секунд

```bash
bin/llamastery doctor                  # что видно: сборки, пресет, замеры, калибровка
bin/llamastery validate                # пресет проходит схему твоей сборки?
bin/llamastery budget --models-max 2   # влезет ли, и что влезет одновременно
bin/llamastery load <пресет>           # загрузить в VRAM (с предпроверкой)
bin/llamastery measure                 # сколько реально занято на карте
bin/llamastery probe --tokens 110000   # честная скорость на глубине 110k
```

Типичный ответ `budget`:

```
пресет: qwen-32B-128ctx — веса 19.2G, KV 2.6G, всего 22.8G
  влезает: qwen-32B-128ctx (замер 21.9 GiB, замер)
  не влезает: big-70B (нужно 38.1 GiB, осталось 2.0 GiB)
```

Если вместо прогноза есть живой замер — решение принимается по замеру, оценка показывается рядом.

## Кому это нужно

- У тебя несколько сборок `llama.cpp` (апстрим + форки) и надо понять, **чем они отличаются** и что умеет каждая.
- У тебя один GPU на 8–24 ГБ и вопрос **«влезет ли модель с контекстом 100k+»** возникает раньше, чем вопрос скорости.
- Ты правишь `models.ini` руками и хочешь узнать об ошибке **до** загрузки, а не по падению CUDA в середине запроса.

Не нужно, если: одна стоковая сборка, одна модель, короткий контекст — хватит обычного `llama-server`.

## Как это работает

1. **Проверь.** `validate` сверяет каждую секцию `models.ini` со схемой флагов, разобранной из
   `llama-server --help` именно твоей сборки. Форковые флаги (`n-cpu-moe`, `spec-type`, `override-tensor`)
   видны автоматически, плюс ловятся известные грабли (например `ubatch 2048 + спекулятивный декодер = падение`).
2. **Спрогнозируй.** `budget` раскладывает VRAM на веса / KV / mmproj / compute buffer.
   Веса — по размеру GGUF с учётом `n-cpu-moe` и `-ot`, KV — по слоям внимания (у гибридов и MLA — своя формула),
   compute buffer — подогнанная константа. Пока калибровки нет, инструмент честно пишет «оценка занижена».
3. **Загрузи.** `load` сам поднимает сервер (роутер или single-режим для сборок без `--models-preset`),
   перед загрузкой показывает прогноз и отказывается грузить заведомо падающее (обходится `--force`).
4. **Замерь.** `measure` снимает факт с `nvidia-smi`, `probe` меряет tg/prefill на **реальном** длинном
   промпте (с выключенным prompt cache — прогретые пробы помечаются 🔥 и в статистику не идут).
   Замеры имеют приоритет над оценкой везде: в `budget`, `validate` и комментариях пресетов.

Падения на глубоком контексте запоминаются (`crashes`) — пресет, ронявший сервер, помечается 💥.

## Быстрый старт

Требования: Python 3.11+, собранный `llama-server` хотя бы одной сборки. Собирать ничего не надо —
`llamastery` только регистрирует готовые.

```bash
git clone <репозиторий> ~/git/llamastery
cd ~/git/llamastery
bin/llamastery doctor              # проверить окружение
bin/llamastery builds detect       # найти форки (только просмотр)
bin/llamastery builds detect --apply   # записать в реестр
bin/llamastery validate            # проверить все секции models.ini
bin/llamastery budget              # прогноз по всем пресетам
bin/llamastery runtime start --build faks  # поднять сервер
bin/llamastery load <пресет> --dry-run    # проверка без загрузки
bin/llamastery load <пресет>               # загрузить
bin/llamastery probe --tokens 110000 --from-file server.cpp  # скорость на реальном коде
bin/llamastery runtime stop
```

Порядок важен: `measure` и `probe` — только при загруженном пресете.
Импорт чужих пресетов всегда начинается с `--dry-run` (см. [Пресеты](docs/ru/presets.md)).

## Команды по задачам

| Хочу | Команда |
|---|---|
| Понять, что вообще видно | `llamastery doctor` |
| Зарегистрировать сборки | `llamastery builds list` / `detect` / `show <имя>` / `stale` |
| Узнать флаги своей сборки | `llamastery schema --build faks --grep moe` |
| Проверить пресет | `llamastery validate [секция]` |
| Посчитать память | `llamastery budget [секции] --models-max 2 --explain` |
| Откалибровать прогноз | `llamastery calibrate --free-mib 1147` / `--from-log` / `--from-tune` |
| Загрузить / выгрузить | `llamastery load <пресет>` / `llamastery runtime unload` |
| Померить скорость | `llamastery probe --tokens 110000 --from-file <код>` |
| Снять факт VRAM | `llamastery measure` / `measure --recalibrate` |
| Перенести чужой пресет | `llamastery presets import --source <файл|URL|git> --dry-run` |
| Оформить комментарии | `llamastery presets annotate --apply` (сначала без `--apply`) |
| Разобрать падение | `llamastery crashes` / `crashes --forget <пресет>` |
| Подобрать параметры | `llamastery tune <ini> <секция> --build faks --extra ...` (грузит GPU, только с согласия) |
| Выгрузить в llama-swap | `llamastery swap export --build faks -o <yaml> --dry-run` (дефолт связки — faks, см. docs/ru/swap.md) |

Полный разбор каждой команды — в [SKILL.md](SKILL.md) и `docs/`.

## Что внутри

Три слоя, снизу вверх:

- **Знание о железе и файлах:** `lib/gguf.py` (читает шапку GGUF: слои, головы, MLA/MTP, таблица тензоров),
  `lib/schema.py` (флаги из `--help` с кэшем по mtime), `lib/builds.py` (реестр сборок).
- **Решение «можно ли»:** `lib/budget.py` (прогноз VRAM), `lib/validate.py` (проверка + грабли форков),
  `lib/server.py` (жизненный цикл сервера), `lib/vram.py` + `lib/measure.py` (замеры важнее оценки).
- **Интерфейс:** `bin/llamastery` (единый CLI), `tools/tune_models.py` (автотюнер),
  `lib/annotate.py` (стандарт комментариев), `lib/probe.py`, `lib/crashes.py`, `lib/presets.py`, `lib/inifile.py`.

Детально по файлам — в [Документация](#документация).

## Документация

Читай на том языке, который быстрее идёт — содержание одинаковое.
Краткая памятка для агента — только на русском: [SKILL.md](SKILL.md).

| Тема | RU | EN | 中文 |
|---|---|---|---|
| Установка и первое действие | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| Сборки, форки, схема флагов | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| Пресеты: формат, правка, оформление | [docs/ru/presets.md](docs/ru/presets.md) · [comments](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [comments](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [comments](docs/zh/comments.md) |
| Бюджет VRAM и замеры | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| Автотюнинг: методика и ловушки | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| llama-swap: экспорт и связка | [docs/ru/swap.md](docs/ru/swap.md) | [docs/en/swap.md](docs/en/swap.md) | [docs/zh/swap.md](docs/zh/swap.md) |

Новичкам: `getting-started` → `builds` → `presets` → `measure`. Остальное — по мере вопросов.

## Метки

Комментарии над секциями пишет `llamastery presets annotate`, руками их править не надо.
Слова в файле русские, в EN/ZH-документации показаны символами.

| Символ | В файле | Смысл |
|---|---|---|
| ✅ | `[замер]` | измерено прогоном, а не посчитано |
| 🧮 | `[оценка]` | расчёт `budget` |
| 🔬 | `Замер:` | условия измерения |
| ⚠️ | `Нюанс:` | ловушки и условия применения |
| 💬 | `Заметка:` | старая проза дословно |
| 🔥 | `ПРОГРЕТО` | проба из тёплого KV — в prefill не считается |
| 💥 | `ПАДАЕТ` | пресет ронял сервер |

## Установка

Как скилл агента (симлинк, а не копия — правки видны всем сразу):

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

Как обычный CLI — никак: `bin/llamastery` работает из любого места, ставить ничего не надо.

## Переменные окружения

| Переменная | По умолчанию | Зачем менять |
|---|---|---|
| `LLAMA_MODELS_INI` | `~/.config/llama/models.ini` | пресет лежит в другом месте |
| `LLAMA_SERVER_BIN` | из реестра | обойти реестр одним бинарём |
| `LLAMA_SERVER` | `http://127.0.0.1:8099` | сервер на другом порту |
| `LLAMASTERY_CONFIG_DIR` | `~/.config/llamastery` | реестр сборок |
| `LLAMASTERY_STATE_DIR` | `~/.local/state/llamastery` | замеры, калибровка, журнал падений, лог |
| `LLAMASTERY_CACHE_DIR` | `~/.cache/llamastery` | кэш схемы флагов |

## Гарантии

1. **Ничего не пишет молча.** Импорт — с `--dry-run` по умолчанию, запись — после `.bak-<время>`.
2. **Замер важнее расчёта.** Есть факт — решение по факту, оценка рядом для сверки.
3. **Схема — от твоей сборки.** Не хардкод: форковые флаги подхватываются из `--help`.
4. **Честно про неточность.** Нет калибровки или большой разброс — так и пишет.
5. **Падения не забываются.** Упал на глубине — помечен, `validate` предупредит.

## Тесты

```bash
python3 tests/run_tests.py           # без зависимостей, ~37 проверок
```

---

# llamastery (EN)

Fit the model into VRAM on the first try: preset validation, honest memory forecast, measured fact.

`llamastery` is a CLI and agent skill around `llama-server`. It knows your builds and forks
(`faks`, `upstream`, `ik_llama`, `xing4`), validates each preset against **that** build's flag schema,
predicts whether it fits your GPU, loads it, and measures real speed at long context.
Python 3.11+ standard library only, no dependencies.

## Navigation

- [In 60 seconds](#in-60-seconds) · [Quick start](#quick-start) · [Commands by task](#commands-by-task)
- [Whats inside](#whats-inside) · [Documentation](#documentation) · [Tests](#tests)
- [In 60 seconds](#in-60-seconds) · [Quick start](#quick-start)

## In 60 seconds

```bash
bin/llamastery doctor                  # what is visible: builds, preset, measurements, calibration
bin/llamastery validate                # does the preset match your build's schema?
bin/llamastery budget --models-max 2   # does it fit, and what fits together
bin/llamastery load <preset>           # load into VRAM (with preflight check)
bin/llamastery measure                 # real usage from the card
bin/llamastery probe --tokens 110000   # honest speed at 110k depth
```

Where a live measurement exists, the "does it fit" decision uses it; the estimate is shown alongside.

## Quick start

Requirements: Python 3.11+, at least one built `llama-server`. `llamastery` only registers ready builds.

```bash
bin/llamastery doctor
bin/llamastery builds detect --apply   # register forks
bin/llamastery validate
bin/llamastery budget
bin/llamastery runtime start --build faks
bin/llamastery load <preset> --dry-run    # check without loading
bin/llamastery load <preset>
bin/llamastery probe --tokens 110000 --from-file server.cpp
```

`measure` and `probe` only make sense with a loaded preset. Import foreign presets with `--dry-run` first.

## Commands by task

| I want | Command |
|---|---|
| See what is visible | `llamastery doctor` |
| Register builds | `llamastery builds list` / `detect` / `show <name>` / `stale` |
| Learn my build's flags | `llamastery schema --build faks --grep moe` |
| Validate a preset | `llamastery validate [section]` |
| Estimate memory | `llamastery budget [sections] --models-max 2 --explain` |
| Load / unload | `llamastery load <preset>` / `llamastery runtime unload` |
| Measure speed | `llamastery probe --tokens 110000 --from-file <code>` |
| Capture real VRAM | `llamastery measure` |
| Import a foreign preset | `llamastery presets import --source <file\|URL\|git> --dry-run` |
| Export to llama-swap | `llamastery swap export --build faks -o <yaml> --dry-run` (default backend — faks, see docs/en/swap.md) |

Details: [SKILL.md](SKILL.md) (Russian) and [Documentation](#documentation).

## Whats inside

Three layers, bottom up:

- **Hardware and file knowledge:** `lib/gguf.py` (GGUF header: layers, heads, MLA/MTP, tensor table),
  `lib/schema.py` (flags from `--help`, cached by mtime), `lib/builds.py` (build registry).
- **The "may I" decision:** `lib/budget.py` (VRAM forecast), `lib/validate.py` (checks + fork traps),
  `lib/server.py` (server lifecycle), `lib/vram.py` + `lib/measure.py` (measurements beat estimates).
- **Interface:** `bin/llamastery` (single CLI), `tools/tune_models.py` (auto-tuner),
  `lib/annotate.py` (comment standard), `lib/probe.py`, `lib/crashes.py`, `lib/presets.py`, `lib/inifile.py`.

## Documentation

Read in whichever language is fastest — the content is the same.

| Topic | RU | EN | 中文 |
|---|---|---|---|
| Install and first steps | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| Builds, forks, flag schema | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| Presets: format, editing, comments | [docs/ru/presets.md](docs/ru/presets.md) · [comments](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [comments](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [comments](docs/zh/comments.md) |
| VRAM budget and measurements | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| Auto-tuning: method and traps | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| llama-swap: export and backend | [docs/ru/swap.md](docs/ru/swap.md) | [docs/en/swap.md](docs/en/swap.md) | [docs/zh/swap.md](docs/zh/swap.md) |

Beginners: `getting-started` → `builds` → `presets` → `measure`.

## Tests

```bash
python3 tests/run_tests.py           # no dependencies
```

---

# llamastery (中文)

一次就把模型装进显存：校验 preset、诚实的显存预测、实测为准。

`llamastery` 是围绕 `llama-server` 的 CLI 和 Agent 技能。它认得你的构建与分支
（`faks`、`upstream`、`ik_llama`、`xing4`），按**当前构建**的参数表校验 preset，
预测能否装进显卡，负责加载，并在长上下文下实测真实速度。
仅需 Python 3.11+ 标准库，零第三方依赖。

## 导航

- [60秒速览](#60秒速览) · [快速开始](#快速开始) · [按任务查命令](#按任务查命令)
- [目录结构](#目录结构) · [文档](#文档) · [测试](#测试)
- [60秒速览](#60秒速览) · [快速开始](#快速开始)

## 60秒速览

```bash
bin/llamastery doctor                  # 可见：构建、preset、实测、校准
bin/llamastery validate                # preset 是否符合当前构建的参数表？
bin/llamastery budget --models-max 2   # 装不装得下，哪些能同时装下
bin/llamastery load <preset>           # 加载进显存（含预检）
bin/llamastery measure                 # 显卡真实占用
bin/llamastery probe --tokens 110000   # 110k 深度下的真实速度
```

已有实测的地方，“装不装得下”以实测为准，估算值并列显示以便核对。

## 快速开始

要求：Python 3.11+，至少一个已编译好的 `llama-server`。`llamastery` 只做注册，不负责编译。

```bash
bin/llamastery doctor
bin/llamastery builds detect --apply   # 注册各分支
bin/llamastery validate
bin/llamastery budget
bin/llamastery runtime start --build faks
bin/llamastery load <preset> --dry-run    # 只检查不加载
bin/llamastery load <preset>
bin/llamastery probe --tokens 110000 --from-file server.cpp
```

`measure` 和 `probe` 只有在 preset 已加载时才有意义。导入外部 preset 先用 `--dry-run`。

## 按任务查命令

| 想做 | 命令 |
|---|---|
| 看清现状 | `llamastery doctor` |
| 注册构建 | `llamastery builds list` / `detect` / `show <name>` / `stale` |
| 查构建的参数 | `llamastery schema --build faks --grep moe` |
| 校验 preset | `llamastery validate [section]` |
| 估算显存 | `llamastery budget [sections] --models-max 2 --explain` |
| 加载 / 卸载 | `llamastery load <preset>` / `llamastery runtime unload` |
| 测速 | `llamastery probe --tokens 110000 --from-file <code>` |
| 实测显存 | `llamastery measure` |
| 导入外部 preset | `llamastery presets import --source <file\|URL\|git> --dry-run` |
| 导出到 llama-swap | `llamastery swap export --build faks -o <yaml> --dry-run`（默认后端 faks，见 docs/zh/swap.md）|

详情见 [SKILL.md](SKILL.md)（俄文）与[文档](#文档)。

## 目录结构

自下而上三层：

- **硬件与文件知识：** `lib/gguf.py`（GGUF 头：层数、注意力头、MLA/MTP、张量表）、
  `lib/schema.py`（从 `--help` 解析参数表，按 mtime 缓存）、`lib/builds.py`（构建注册表）。
- **“能不能”决策：** `lib/budget.py`（显存预测）、`lib/validate.py`（校验 + 各分支陷阱）、
  `lib/server.py`（服务器生命周期）、`lib/vram.py` + `lib/measure.py`（实测胜过推算）。
- **接口：** `bin/llamastery`（统一 CLI）、`tools/tune_models.py`（自动调优器）、
  `lib/annotate.py`（注释规范）、`lib/probe.py`、`lib/crashes.py`、`lib/presets.py`、`lib/inifile.py`。

## 文档

选读起来最快的那份即可，内容完全一致。

| 主题 | RU | EN | 中文 |
|---|---|---|---|
| 安装与上手 | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| 构建、分支、参数表 | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| preset：格式、编辑、注释规范 | [docs/ru/presets.md](docs/ru/presets.md) · [注释](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [注释](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [注释](docs/zh/comments.md) |
| 显存预算与实测 | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| 自动调优：方法与陷阱 | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| llama-swap：导出与后端 | [docs/ru/swap.md](docs/ru/swap.md) | [docs/en/swap.md](docs/en/swap.md) | [docs/zh/swap.md](docs/zh/swap.md) |

新手顺序：`getting-started` → `builds` → `presets` → `measure`。

## 测试

```bash
python3 tests/run_tests.py           # 无第三方依赖
```
