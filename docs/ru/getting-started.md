# Установка и первые шаги

## Что нужно

* Python 3.11 или новее. Больше ничего: внешних зависимостей нет, ничего
  ставить глобально не нужно.
* Собранный `llama-server` хотя бы для одной сборки или форка. Сборку
  инструмент не делает — только регистрирует готовую.

## Как получить

```bash
git clone <репозиторий> ~/git/llamastery
```

Репозиторий самодостаточен: `bin/llamastery` работает из любого места, путь
к нему в документации обозначен `$LM`.

```bash
LM=~/git/llamastery
$LM/bin/llamastery --help
```

## Как установить как скилл агента

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

Симлинк, а не копия: правки видны всем агентам сразу. `SKILL.md` в корне
репозитория — краткая памятка, остальное в `docs/`.

## Первый запуск

```bash
$LM/bin/llamastery doctor
```

`doctor` показывает, что удалось увидеть: зарегистрированные сборки, текущий
пресет, число замеров, калибровку compute buffer, общий объём VRAM и актуальность
сборок.

Если реестр пуст:

```bash
$LM/bin/llamastery builds detect --apply
```

`detect` ищет форки в типичных каталогах (`~/git/*`, `~/llama*`). Без
`--apply` только показывает найденное.

## Минимальный рабочий цикл

```bash
$LM/bin/llamastery validate                    # все секции проходят схему
$LM/bin/llamastery budget                     # сколько памяти нужно каждой
$LM/bin/llamastery runtime start --build faks  # поднять роутер
$LM/bin/llamastery load <пресет>               # загрузить модель в VRAM
$LM/bin/llamastery measure                     # снять факт памяти
$LM/bin/llamastery probe --tokens 110000        # скорость на реальной глубине
$LM/bin/llamastery runtime stop                # остановить
```

Порядок важен: `measure` имеет смысл только при загруженном пресете, `probe` —
тоже.

## Где что лежит

| Что | Путь |
|---|---|
| реестр сборок | `~/.config/llamastery/builds.json` |
| замеры и калибровка | `~/.local/state/llamastery/` |
| журнал падений | `~/.local/state/llamastery/crashes.json` |
| лог сервера | `~/.local/state/llamastery/router.log` |
| кэш схемы флагов | `~/.cache/llamastery/` |
| пресеты роутера | `~/.config/llama/models.ini` и рядом |

Переопределяется переменными окружения `LLAMASTERY_CONFIG_DIR`,
`LLAMASTERY_STATE_DIR`, `LLAMASTERY_CACHE_DIR` (см. README).
Префикс от прежнего имени инструмента не поддерживается.

## Проверка на своих данных

Перед тем как доверять цифрам, стоит убедиться, что замеры вообще есть:

```bash
$LM/bin/llamastery ingest            # что нашлось в tune-results и комментариях
$LM/bin/llamastery budget --explain  # разложение VRAM по слагаемым
```

Пока `calibrate` не запускался, оценка VRAM занижена — инструмент об этом
печатает предупреждение. Это не ошибка, а признак того, что калибровки ещё нет.