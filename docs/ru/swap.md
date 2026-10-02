# llama-swap: экспорт и рекомендации по связке

`llamastery` — источник правды (validate + budget + замеры идут напрямую
к `llama-server`), `llama-swap` — только рантайм-прокси: один порт,
хот-свап, `ttl`. Здесь — как экспортировать и с какой сборкой запускать.

## Команды

```bash
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml --dry-run  # сначала проверка
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml             # запись (бэкап .bak-<время> сам)
llamastery swap export --build faks --only qwen3.8-35B-A3B-miniplus-v2.1-128ctx     # подмножество в stdout
llamastery swap status                                # прокси :8080 + прямой сервер рядом
llamastery swap install                               # скачать бинарь v261 в ~/.local/bin (только Linux x64)
```

## Правила

1. Правится `models.ini`, в swap-YAML — только `export`. Руками YAML не править.
2. `validate / budget / measure / probe / tune` — только напрямую
   (`:8099`/`:8098`), не через `:8080`. Замер через прокси искажает timings.
3. Одна swap-конфигурация — одна сборка. Не смешивать `faks` и `ik`
   в одном файле: схемы флагов разные, часть флагов молча отвалится.
4. Роутерные сборки (`faks`, `upstream`) экспортируются в single-режим
   через `preset_to_argv`. Матрешка «swap → роутер → инстанс» запрещена.

## С какой сборкой использовать llama-swap (рекомендация)

Железо-ориентир: RTX 3060 12GB + Ryzen 5700X + 32GB RAM.

| Сборка | Когда | Статус для swap |
|---|---|---|
| `faks` | **Дефолт.** Пресеты `models.ini` писались под него: `n-cpu-moe`, `fa`, `ctx-checkpoints`, `kv-unified`, env `GGML_CUDA_REGISTER_HOST=1` + `GGML_SCHED_PREFETCH_EXPERTS=1`. Экспорт чистый, без варнингов. | ✅ основной |
| `upstream` | Запасной, если `faks` сломался или нужна самая свежая фича `ggml-org`. | ✅ fallback |
| `ik` | Максимум сырой MoE-скорости (IQK/Trellis, FlashMLA), но схема старая (синк август 2024): при экспорте отваливаются `n-gpu-layers`, `load-mode`, `kv-unified`, `n-predict`. Для рекордов грузи напрямую `llamastery load --build ik`, в swap — только отдельным конфигом и с проверкой варнингов. | ⚠️ отдельно |
| `xing4` | Только для моделей архитектуры `xing4`. | ❌ не смешивать |

Итог: **swap по умолчанию — `faks`** (`swap export --build faks`).
`ik` — для ручного рекорда вне swap.

## Запуск

```bash
~/.local/bin/llama-swap --config ~/.config/llama-swap/config.yaml --listen 0.0.0.0:8080
```

Способ установки — бинарь (выбран осознанно): хостовый CUDA уже работает,
docker без nvidia-runtime, а unified-образ потерял бы Faks-патчи и потребовал
бы перемаппивания путей HF-кэша.
