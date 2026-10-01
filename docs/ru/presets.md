# Формат пресетов llama.cpp

Три уровня конфигурации. Путаница между ними — источник почти всех
недоразумений, поэтому начни с этой таблицы.

| Уровень | Файл | Ключ | Что это |
|---|---|---|---|
| Per-model пресеты | `models.ini` / `presets.ini` | `--models-preset PATH` | секция = модель, `[*]` = общие дефолты |
| Именованные, шарятся | `preset.ini` в пустом HF-репо | `-hf user/repo` | пресет как «модель» с тегом |
| Общие для всех бинарей | `/etc/llama.cpp/config.ini`, `~/.config/llama.cpp/config.ini` | без флага | действует и на `llama-cli`; читаются только `[*]` и секция до первого заголовка |

Порядок применения (от слабого к сильному):
`config.ini` → переменные окружения → пресет модели → CLI самого роутера.
То есть аргумент, переданный в командной строке, **побеждает** значение из пресета.

Источник истины по «кто чем управляет» — `tools/server/server-models.cpp`,
функция `unset_reserved_args()`:

* вырезаются из любого пресета: `ssl-key-file`, `ssl-cert-file`, `api-key`,
  `models-dir`, `models-max`, `models-preset`, `models-autoload`;
* перезаписываются роутером при спавне дочернего процесса: `port`, `host`, `alias`;
* в per-model пресете `model` / `mmproj` / `hf-repo` задают **саму модель** и
  потому легальны (роутер режет их только из базового пресета).

## Три формы записи одного ключа

Эквивалентны, можно смешивать в одном файле:

```ini
ctx-size = 32768     # длинная
c = 32768            # короткая
LLAMA_ARG_CTX_SIZE = 32768   # имя переменной окружения
```

Логика: сначала дедупликация форм (короткая и длинная считаются одним
аргументом), затем слияние с базовыми аргументами роутера.

Булевы значения: `on` / `off`, `true` / `false`, `1` / `0`, `enabled` / `disabled`.
Флаг без значения (`kv-unified`, `jinja`) — просто оставь значение пустым.

## Пример

```ini
version = 1

; общие для всех моделей дефолты
[*]
n-gpu-layers = 99
fa = true
cache-type-k = q8_0
cache-type-v = q8_0

[my-model-65k]
m = /models/Qwen3-30B-A3B-Q4_K_M.gguf
c = 65536
n-cpu-moe = 18
temp = 0.6
top-p = 0.95
top-k = 20

[my-model-65k-vision]
m = /models/Qwen3-30B-A3B-Q4_K_M.gguf
mmproj = /models/mmproj-Q8_0.gguf
mmproj-offload = 0
image-min-tokens = 1024
c = 65536
n-cpu-moe = 18
```

## Откуда роутер берёт модели

1. `~/.cache/llama.cpp` (или `LLAMA_CACHE`) — закэшированные HF-модели;
2. `--models-dir PATH` — только прямые дети каталога, **без рекурсии**;
3. `--models-preset` — секции с явными путями (`m = ...`).

При совпадении имени приоритет: preset > models-dir > cache. Для
многошаговых моделей и `mmproj` файлы кладут в подкаталог, имя проектора
должно начинаться с `mmproj`.

Сопутствующие флаги роутера: `--models-max N` (по умолчанию 4, `0` без
лимита), `--models-autoload` / `--no-models-autoload`.

## Что важно понимать про память

Каждая загруженная модель — **отдельный процесс** `llama-server` на своём
свободном порту, запущенный на `127.0.0.1`. Роутер только проксирует
запросы. Отсюда:

* VRAM складывается по всем одновременно загруженным моделям, а не по
  «размеру пресета»;
* `LLAMA_SERVER_ROUTER_PORT` прокидывается в дочерние процессы;
* `parallel` внутри одного пресета — это отдельная ось: она кратно
  умножает KV-пул внутри одного процесса.

`llamastery budget` как раз отвечает на вопрос «влезет ли N пресетов
одновременно», которого нет ни в одном из существующих GUI-лаунчеров.

## Структура KV-кэша

```
байт_на_токен = n_внимания_слоёв × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
размер_кэша   = байт_на_токен × ctx × слоты
```

где `bytes(q8_0) = 1.0625`, `bytes(f16) = 2`, `bytes(q4_0) = 0.5625`.

Две ловушки:

* **GQA.** `n_kv_heads` обычно намного меньше `n_head` (у Qwen3-35B-A3B:
  16 и 2), поэтому KV в разы меньше, чем «наивные» расчёты.
* **Гибридные архитектуры.** У qwen35moe / Nemotron-H / Jamba часть слоёв —
  SSM (Mamba-подобные), их состояние не растёт с контекстом и KV они не
  хранят. Полное внимание только каждый `full_attention_interval`-й слой.
  `llamastery budget` это учитывает; «посчитать все слои» — занижение памяти
  в несколько раз.

`kv-unified` (включён у большинства современных пресетов) означает общий
пул KV на весь контекст: `parallel` перестаёт умножать кэш. Выключишь —
вернётся умножение на число слотов.

## Операции над пресетом

```bash
llamastery presets list                      # все секции
llamastery presets show <секция>             # ключи секции
llamastery presets globals                   # секция [*]
llamastery presets export -o - <секция>...   # выгрузка подмножества
llamastery presets annotate --dry-run        # что изменит форматтер
llamastery presets annotate --apply          # применить
```

### Импорт чужих пресетов

Источники: локальный файл, URL, `git-репо#ветка:путь/внутри`.

```bash
llamastery presets import --source ./foreign-presets.ini --dry-run
llamastery presets import --source 'https://github.com/u/repo#main:presets.ini' --dry-run
llamastery presets import --source git@github.com:u/repo.git --only my-model-128k
llamastery presets import --source ./p.ini --on-conflict new      # не перетирать
llamastery presets import --source ./p.ini --on-conflict overwrite --conflicts
```

Поведение: по умолчанию `--dry-run`; при конфликтах `skip`; перед записью
делается `.bak-<время>`. `--rename старая=новая` переименовывает секции. Если
целевого файла нет, он создаётся.

### Проверка

```bash
llamastery validate                          # все секции текущего models.ini
llamastery validate <секция> --json
llamastery validate --build ik               # проверка под конкретную сборку
llamastery validate --no-paths                # не ходить на диск (быстрее)
```

Ловит: неизвестные ключи, control-аргументы роутера (`api-key`, `models-max`
вырезаются; `port`/`host`/`alias` перезаписываются), несуществующие GGUF,
`c` больше обученного контекста, `n-cpu-moe` больше числа слоёв, известные
грабли форков, пресеты из журнала падений.

Схема флагов у каждой сборки своя, поэтому пресет, написанный под форк, на
ik_llama даст предупреждения: часть флагов там просто не существует. Подсказка
в ответе называет конкретный флаг.

## Загрузка и выгрузка

```bash
llamastery runtime start --build faks      # поднять роутер
llamastery runtime status                  # порт, pid, сборка, что в VRAM
llamastery load <пресет>                   # загрузить в VRAM
llamastery runtime unload [модель]         # выгрузить, сервер остаётся жив
llamastery runtime restart --build faks
llamastery runtime stop
llamastery runtime logs -n 100
```

`llamastery` сам управляет сервером — отдельных менеджеров (`llama`,
`llama-faks`, `llama-ik`) не требуется и не используется. Это осознанно:
скрипты под каждую сборку есть далеко не у всех, а инструмент должен работать
у того, кто их не ставил.

Перед загрузкой идёт проверка по схеме **этой** сборки и прогноз VRAM:

```
пресет: qwen3.8-35B-A3B-miniplus-21-128ctx-ngram-mmproj
сборка: faks   бинарь: /home/axel/git/llama-faks/build/bin/llama-server
  VRAM: 8.67 GiB из 12.00 GiB, запас +2.96 GiB
```

При ошибках загрузка не выполняется (`--force` обходит).

Перед загрузкой в VRAM выгружаются остальные: память на карте одна. Число
одновременно загруженных моделей задаётся `--models-max` роутера.

Оформление комментариев над секциями — в [comments.md](comments.md).
