# llamastery

Управление сборками и форками llama.cpp, пресетами роутера и бюджетом VRAM.
Скилл для агентов (Skill-формат) плюс обычный CLI, который можно звать руками.

Только стандартная библиотека Python 3.11+. Внешних зависимостей нет.
Отдельных скриптов-менеджеров не требуется: `llamastery` сам поднимает сервер,
грузит и выгружает модели для любой зарегистрированной сборки, включая
сборки без роутер-режима.

## Навигация

- [Документация](#документация) · [Что внутри](#что-внутри) · [Быстрый старт](#быстрый-старт)
- [Установка как скилла для агента](#установка-как-скилла-для-агента) · [Переменные окружения](#переменные-окружения)
- [Метки в комментариях пресетов](#метки-в-комментариях-пресетов) · [Принципы](#принципы) · [Тесты](#тесты)
- [Быстрый старт](#быстрый-старт) · [Документация](#документация) · [Принципы](#принципы)
- [Documentation](#documentation) · [What is inside](#what-is-inside) · [Quick start](#quick-start)
- [Install as an agent skill](#install-as-an-agent-skill) · [Environment variables](#environment-variables)
- [Principles](#principles) · [Tests](#tests) · [Quick start](#quick-start)
- [文档](#文档) · [目录结构](#目录结构) · [快速开始](#快速开始)
- [作为 Agent 技能安装](#作为-agent-技能安装) · [环境变量](#环境变量) · [原则](#原则) · [测试](#测试)
- [llamastery](#llamastery) · [llamastery (EN)](#llamastery-en) · [llamastery (中文)](#llamastery-中文)

## Документация

Подробные файлы лежат отдельно на каждом языке. Начинать лучше с того,
который читается быстрее; содержание одинаковое.

| Тема | RU | EN | 中文 |
|---|---|---|---|
| Установка и первое действие | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| Сборки, форки, схема флагов | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| Пресеты: формат, правка, оформление | [docs/ru/presets.md](docs/ru/presets.md) · [comments](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [comments](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [comments](docs/zh/comments.md) |
| Бюджет VRAM и замеры | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| Автотюнинг: методика и ловушки | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| Краткая памятка для агента | [SKILL.md](SKILL.md) | — | — |

## Что внутри

```
bin/llamastery              единый CLI
lib/
  schema.py             схема флагов из `llama-server --help` (с кэшем по mtime)
  inifile.py            INI с сохранением комментариев (в отличие от configparser)
  presets.py            импорт/экспорт/слияние пресетов (файл, URL, git)
  gguf.py               ридер метаданных GGUF (для оценки памяти)
  budget.py             разложение VRAM на слагаемые + калибровка
  measure.py            сбор замеров из tune-results и комментариев models.ini
  validate.py           проверка пресетов + грабли форков
  builds.py             реестр сборок/форков + проверка актуальности
  crashes.py            журнал падений: какие пресеты роняют сервер
  runtime.py            пред-проверка пресета и выбор менеджера (необязателен)
  server.py             свой жизненный цикл сервера: start/stop/load/unload
  vram.py               живой замер VRAM с карты и перекалибровка
  annotate.py           единый стандарт комментариев над пресетами
  probe.py              замер tg/prefill на живом сервере
tools/tune_models.py    автотюнер (двухэтапный, stdlib only)
docs/                   подробная документация: ru / en / zh
```

## Быстрый старт

```bash
bin/llamastery doctor              # проверить окружение
bin/llamastery builds detect --apply   # зарегистрировать форки
bin/llamastery validate
bin/llamastery budget --models-max 2
bin/llamastery load <пресет> --dry-run     # проверка без загрузки
bin/llamastery runtime start --build faks  # поднять роутер
bin/llamastery load <пресет>               # загрузить в VRAM
bin/llamastery measure                     # снять факт памяти
bin/llamastery probe --tokens 110000        # скорость на реальной глубине
```

## Установка как скилла для агента

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

Один репозиторий — много агентов: правки видны всем сразу.

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `LLAMA_MODELS_INI` | `~/.config/llama/models.ini` | пресет роутера |
| `LLAMA_SERVER_BIN` | из реестра | путь к llama-server |
| `LLAMA_SERVER` | `http://127.0.0.1:8099` | адрес сервера |
| `LLAMASTERY_CONFIG_DIR` | `~/.config/llamastery` | реестр сборок |
| `LLAMASTERY_STATE_DIR` | `~/.local/state/llamastery` | замеры, калибровка, журнал падений |
| `LLAMASTERY_CACHE_DIR` | `~/.cache/llamastery` | кэш схемы флагов |

## Метки в комментариях пресетов

Формат задан инструментом, слова в нём русские — их пишет
`llamastery presets annotate`. В документации на других языках они показаны
символами (см. `docs/en/comments.md`, `docs/zh/comments.md`).

| Символ | Слово в файле | Смысл |
|---|---|---|
| ✅ | `[замер]` | фактически измеренный прогон |
| 🧮 | `[оценка]` | расчёт `budget`, не измерение |
| 🔬 | `Замер:` | условия измерения |
| ⚠️ | `Нюанс:` | ловушки и условия применения |
| 💬 | `Заметка:` | старая проза дословно |
| 🔥 | `ПРОГРЕТО` | проба из прогретого KV, в prefill не идёт |
| 💥 | `ПАДАЕТ` | пресет ронял сервер |

## Принципы

1. **Ничего не пишет молча.** Импорт по умолчанию `--dry-run`, перед реальной
   записью — `.bak-<время>`. Тюнинг и запись требуют явного согласия.
2. **Замер важнее расчёта.** `measurements.json` собирается из реальных
   прогонов; `validate` и `budget` используют его в приоритете. Если по
   пресету есть фактический `used_mib`, решение «влезает ли» принимается по
   замеру, а оценка показывается рядом для сверки.
3. **Схема — из help конкретной сборки.** Форки добавляют и переименовывают
   флаги, захардкоженный список устаревает.
4. **Честность оценок.** Если `calibrate` не запускался или разброс остатков
   большой, инструмент об этом говорит, а не делает вид, что знает.
5. **Падения запоминаются.** Пресет может пройти валидацию и упасть только на
   глубоком контексте. Такое попадает в журнал, и `validate` об этом
   предупреждает.

## Тесты

```bash
python3 tests/run_tests.py           # без зависимостей
```

---

# llamastery (EN)

Managing llama.cpp builds and forks, router presets, and the VRAM budget.
An agent skill (Skill format) plus a plain CLI you can run by hand.

Standard library Python 3.11+ only. No external dependencies. No separate
manager scripts needed: `llamastery` starts the server itself and loads and
unloads models for any registered build, including builds without router mode.

## Documentation

Detailed files are provided per language. Pick whichever you read fastest;
the content is the same.

| Topic | RU | EN | 中文 |
|---|---|---|---|
| Install and first steps | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| Builds, forks, flag schema | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| Presets: format, editing, comments | [docs/ru/presets.md](docs/ru/presets.md) · [comments](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [comments](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [comments](docs/zh/comments.md) |
| VRAM budget and measurements | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| Auto-tuning: method and traps | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| Short agent cheat sheet | [SKILL.md](SKILL.md) | — | — |

## What is inside

```
bin/llamastery              single CLI
lib/
  schema.py             flag schema parsed from `llama-server --help` (cached by mtime)
  inifile.py            INI that preserves comments (unlike configparser)
  presets.py            preset import/export/merge (file, URL, git)
  gguf.py               GGUF metadata reader (for memory estimation)
  budget.py             VRAM broken into components + calibration
  measure.py            gathers measurements from tune-results and models.ini comments
  validate.py           preset validation + known fork traps
  builds.py             build/fork registry + freshness check
  crashes.py            crash journal: which presets bring the server down
  runtime.py            preset preflight and manager selection (optional)
  server.py             own server lifecycle: start/stop/load/unload
  vram.py               live VRAM reading from the card and recalibration
  annotate.py           one comment standard for presets
  probe.py              tg/prefill measurement on a live server
tools/tune_models.py    auto-tuner (two-stage, stdlib only)
docs/                   detailed documentation: ru / en / zh
```

## Quick start

```bash
bin/llamastery doctor              # check the environment
bin/llamastery builds detect --apply   # register forks
bin/llamastery validate
bin/llamastery budget --models-max 2
bin/llamastery load <preset> --dry-run     # check without loading
bin/llamastery runtime start --build faks  # start the router
bin/llamastery load <preset>               # load into VRAM
bin/llamastery measure                     # capture real memory use
bin/llamastery probe --tokens 110000        # speed at real depth
```

## Install as an agent skill

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

One repository, many agents: every edit is visible to all of them at once.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_MODELS_INI` | `~/.config/llama/models.ini` | router preset |
| `LLAMA_SERVER_BIN` | from registry | path to llama-server |
| `LLAMA_SERVER` | `http://127.0.0.1:8099` | server address |
| `LLAMASTERY_CONFIG_DIR` | `~/.config/llamastery` | build registry |
| `LLAMASTERY_STATE_DIR` | `~/.local/state/llamastery` | measurements, calibration, crash journal |
| `LLAMASTERY_CACHE_DIR` | `~/.cache/llamastery` | flag schema cache |

## Principles

1. **Never writes silently.** Import defaults to `--dry-run`; a real write gets
   a `.bak-<timestamp>` first. Tuning and writing need explicit consent.
2. **Measurement beats calculation.** `measurements.json` is built from real
   runs and takes priority in `validate` and `budget`. If a preset has a real
   `used_mib`, the "does it fit" decision uses that number, and the estimate is
   shown alongside for cross-checking.
3. **The schema comes from that build's help.** Forks add and rename flags, so
   a hardcoded list goes stale.
4. **Honest estimates.** If `calibrate` has never run, or the spread of
   residuals is large, the tool says so instead of pretending to know.
5. **Crashes are remembered.** A preset can pass validation and still fall over
   only at deep context. That lands in the journal, and `validate` warns.

## Tests

```bash
python3 tests/run_tests.py           # no dependencies
```

---

# llamastery (中文)

管理 llama.cpp 的各个构建与分支、路由的 preset 配置，以及显存预算。
既是 Agent 技能（Skill 格式），也是可以直接手动调用的 CLI。

仅依赖 Python 3.11+ 标准库，没有任何第三方依赖。不需要额外的管理脚本：
`llamastery` 自己负责启动服务器，为任何已注册的构建加载和卸载模型，
包括不具备路由模式的构建。

## 文档

详细文档按语言分别提供。选择你读起来最快的那一份即可，内容完全一致。

| 主题 | RU | EN | 中文 |
|---|---|---|---|
| 安装与上手 | [docs/ru/getting-started.md](docs/ru/getting-started.md) | [docs/en/getting-started.md](docs/en/getting-started.md) | [docs/zh/getting-started.md](docs/zh/getting-started.md) |
| 构建、分支、参数表 | [docs/ru/builds.md](docs/ru/builds.md) | [docs/en/builds.md](docs/en/builds.md) | [docs/zh/builds.md](docs/zh/builds.md) |
| preset：格式、编辑、注释规范 | [docs/ru/presets.md](docs/ru/presets.md) · [注释](docs/ru/comments.md) | [docs/en/presets.md](docs/en/presets.md) · [注释](docs/en/comments.md) | [docs/zh/presets.md](docs/zh/presets.md) · [注释](docs/zh/comments.md) |
| 显存预算与实测 | [docs/ru/measure.md](docs/ru/measure.md) | [docs/en/measure.md](docs/en/measure.md) | [docs/zh/measure.md](docs/zh/measure.md) |
| 自动调优：方法与陷阱 | [docs/ru/tuning.md](docs/ru/tuning.md) | [docs/en/tuning.md](docs/en/tuning.md) | [docs/zh/tuning.md](docs/zh/tuning.md) |
| Agent 速查表 | [SKILL.md](SKILL.md) | — | — |

## 目录结构

```
bin/llamastery              统一的 CLI
lib/
  schema.py             从 `llama-server --help` 解析参数表（按 mtime 缓存）
  inifile.py            保留注释的 INI（不同于 configparser）
  presets.py            preset 导入/导出/合并（文件、URL、git）
  gguf.py               GGUF 元数据读取（用于估算显存）
  budget.py             将显存拆分为各项 + 校准
  measure.py            从 tune-results 与 models.ini 注释中汇总实测数据
  validate.py           preset 校验 + 各分支的已知陷阱
  builds.py             构建/分支注册表 + 版本新鲜度检查
  crashes.py            崩溃日志：哪些 preset 会让服务崩溃
  runtime.py            preset 预检查与管理器选择（可选）
  server.py             自带服务器生命周期：start/stop/load/unload
  vram.py               从显卡实时读取显存并重新校准
  annotate.py           preset 上方的统一注释规范
  probe.py              在运行中的服务器上实测 tg/prefill
tools/tune_models.py    自动调优器（两阶段，仅用标准库）
docs/                   详细文档：ru / en / zh
```

## 快速开始

```bash
bin/llamastery doctor              # 检查环境
bin/llamastery builds detect --apply   # 注册各分支
bin/llamastery validate
bin/llamastery budget --models-max 2
bin/llamastery load <preset> --dry-run     # 只检查不加载
bin/llamastery runtime start --build faks  # 启动路由
bin/llamastery load <preset>               # 加载进显存
bin/llamastery measure                     # 实测显存占用
bin/llamastery probe --tokens 110000        # 真实深度下的速度
```

## 作为 Agent 技能安装

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

一个仓库，多个 Agent 共用：任何修改所有 Agent 立刻可见。

## 环境变量

| 变量 | 默认值 | 含义 |
|---|---|---|
| `LLAMA_MODELS_INI` | `~/.config/llama/models.ini` | 路由 preset |
| `LLAMA_SERVER_BIN` | 取自注册表 | llama-server 路径 |
| `LLAMA_SERVER` | `http://127.0.0.1:8099` | 服务器地址 |
| `LLAMASTERY_CONFIG_DIR` | `~/.config/llamastery` | 构建注册表 |
| `LLAMASTERY_STATE_DIR` | `~/.local/state/llamastery` | 实测数据、校准、崩溃日志 |
| `LLAMASTERY_CACHE_DIR` | `~/.cache/llamastery` | 参数表缓存 |

## 原则

1. **绝不静默写入。** 导入默认带 `--dry-run`，真正写入前先做
   `.bak-<时间戳>` 备份。调优与写入都需要明确同意。
2. **实测胜过推算。** `measurements.json` 来自真实运行，并在 `validate` 与
   `budget` 中优先生效。若某个 preset 已有真实的 `used_mib`，
   「装不装得下」的判断以实测为准，估算值并列显示以便核对。
3. **参数表来自具体构建的 help。** 各分支会新增和重命名参数，
   硬编码的清单必然过时。
4. **诚实的估算。** 如果从没跑过 `calibrate`，或残差离散度很大，
   工具会明确说明，而不是假装知道。
5. **记住崩溃。** 某个 preset 可能通过校验，却只在深上下文下崩溃。
   这类情况会进入日志，`validate` 随后给出警告。

## 测试

```bash
python3 tests/run_tests.py           # 无第三方依赖
```