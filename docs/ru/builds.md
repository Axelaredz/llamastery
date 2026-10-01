# Сборки, форки и схема флагов

## Реестр

Каталог исходников, remote, наличие роутера (`--models-preset`), свои
переменные окружения и свой порт. Роутер есть не у всех: `ik_llama`
работает в single-режиме, и пресет для него транслируется в argv вручную.

```bash
llamastery builds list                        # что зарегистрировано, собрано ли, версия
llamastery builds detect                      # найти форки в типичных каталогах (только просмотр)
llamastery builds detect --apply              # записать в реестр
llamastery builds show faks
llamastery builds add mine --path ~/git/mine-fork \
              --remote https://github.com/u/mine --no-router --port 8098
```

`--port` нужен сборке без роутера, если 8099 уже занят: без него вторая
сборка падает с `couldn't bind to server socket`. Переменные окружения
сборки задаются там же и подхватываются при старте:

```bash
llamastery builds add faks --path ~/git/llama-faks \
              --env GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1
```

## Актуальность

```bash
llamastery builds stale
llamastery builds stale --no-fetch            # без обращения к сети
```

Проверяет три разных состояния, которые легко спутать:

| Состояние | Что значит | Лечится |
|---|---|---|
| отстал от апстрима | есть коммиты, которых нет в upstream | `git pull` |
| ahead | свои коммиты, которых нет в upstream | ничего: пересборка их сохранит |
| бинарь старее дерева | HEAD двигался, а сборка не пересобрана | пересборка |

Отдельно показывает, менялись ли файлы с флагами (`arg.cpp`,
`server-context.cpp`): в форках новые коммиты меняют не только скорость, но и
набор флагов, и об этом лучше узнать до замеров.

Та же строка печатается в `doctor` одной строкой, чтобы проверка была частью
рутины. Ничего не пересобирает: решение остаётся за человеком.

Пересборка после обновления:

```bash
cd ~/git/llama-upstream && git pull --ff-only
cmake --build build -j$(nproc)
llamastery schema --build upstream --refresh   # обновить кэш параметров
```

## Схема флагов

Схема парсится из `--help` конкретного бинаря, поэтому форковые флаги
(`load-mode`, `image-min-tokens`, `ctx-checkpoints`, `spec-type`,
`kv-unified`) видны автоматически. Кэш по mtime бинаря, так что пересборка
сама инвалидирует старую схему.

```bash
llamastery schema --build faks                # сколько флагов у сборки
llamastery schema --build faks --grep moe     # что есть по слову
llamastery schema --build faks --json | jq '."--spec-type"'
llamastery schema --refresh                   # не брать кэш
```

Если нужно понять «что вообще умеет этот форк», начинать стоит отсюда.

Схема используется в трёх местах: перевод пресетных ключей в argv, проверка
`validate` и работа тюнера. Благодаря этому знание о форковых флагах не
дублируется в коде.

## Сборки без роутера

`ik_llama` не умеет `--models-preset`. `llamastery load --build ik`
транслирует секцию пресета в argv и перезапускает single-сервер:

```bash
llamastery load <пресет> --build ik --dry-run   # посмотреть argv, не запуская
llamastery load <пресет> --build ik
```

Трансляция учитывает схему этой сборки: флаги, которых в ik нет, отбрасываются
с предупреждением, а `gpu-layers` вместо `n-gpu-layers` распознаётся как
тот же ключ.

У такой сборки своя логика адресации: адрес сервера берётся из реестра и из
`LLAMA_SERVER`, а не из дефолтного 8099. `probe` и `measure` восстанавливают
сборку по pid-файлу — каждый вызов CLI это отдельный процесс.

## Идентификация процесса

Сборка опознаётся по `/proc/<pid>/exe` — сравнивается с путём `server_bin`
из реестра. Поэтому если пересобрать сборку по новому пути, а запись в
реестре не поправить, `status` перестанет узнавать владельца порта.