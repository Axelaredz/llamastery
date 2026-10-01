#!/usr/bin/env python3
# tune_models.py — двухэтапный тюнер MoE для llama.cpp.
#
# Stage 1: llama-bench / llama-sweep-bench для быстрого screening.
# Stage 2: llama-server для финальной валидации контекста, KV, RAM/VRAM, retrieval.
#
# Целевой профиль: Ryzen 7 5700X + RTX 3060 12GB + 32GB RAM.
# Только Python 3 stdlib.

import argparse
import configparser
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}

# Кэш калибровки токенайзера: (port) -> {"overhead": int, "per_line": float}.
# Токенайзер не меняется между пробами на одном сервере, поэтому калибруем
# один раз и переиспользуем — экономит ~10-15 /tokenize запросов на пробу.
_TOKEN_CALIB = {}

REQUEST_KEYS = {
    "temp",
    "top-p",
    "top-k",
    "min-p",
    "n-predict",
    "repeat-last-n",
    "repeat-penalty",
}

# Ключи, которыми роутер управляет сам: их в пресете быть не должно
RESERVED_KEYS = {
    "model", "m", "mmproj", "port", "host", "alias", "api-key",
    "models-dir", "models-max", "models-preset", "models-autoload",
}


def allowed_keys_for(server_bin) -> set[str]:
    """Допустимые ключи секции: базовый список + флаги конкретной сборки.

    Базовый список один на все форки и inevitably устаревает: форки добавляют
    свои флаги (load-mode, ctx-checkpoints, no-mmproj-offload), а ik_llama их
    не знает. Поэтому список расширяется тем, что реально объявлено в
    `--help` целевого бинаря.

    Скилл лежит рядом, поэтому схема берётся из него; если скилла нет —
    остаётся базовый список, и тюнер продолжает работать автономно.
    """
    allowed = set(ALLOWED_KEYS)
    try:
        import sys as _sys
        root = Path(__file__).resolve().parent.parent
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from lib import schema as _schema
        flags, meta = _schema.load(server_bin)
        if not meta.get("error"):
            seen = {id(v): v for v in flags.values()}
            for f in seen.values():
                for key in f.ini_keys():
                    if key.lower() not in RESERVED_KEYS:
                        allowed.add(key.lower())
    except Exception:
        pass
    return allowed


ALLOWED_KEYS = REQUEST_KEYS | {
    "model",
    "mmproj",
    "ot",
    "image-min-tokens",
    "ctx-checkpoints",
    "checkpoint-min-step",
    "load-mode",
    "mmproj-offload",
    "model-draft",
    "spec-type",
    "c",
    "n-gpu-layers",
    "n-cpu-moe",
    "b",
    "ubatch-size",
    "t",
    "threads-batch",
    "fa",
    "cache-type-k",
    "cache-type-v",
    "parallel",
    "kv-unified",
    "jinja",
    "cache-reuse",
}

WRITABLE_KEYS = {
    "c",
    "n-gpu-layers",
    "n-cpu-moe",
    "b",
    "ubatch-size",
    "t",
    "threads-batch",
    "parallel",
    "fa",
    "cache-type-k",
    "cache-type-v",
    "kv-unified",
    "jinja",
    "cache-reuse",
    "load-mode",
}

BENCH_FLAG_ALIASES = {
    "ctx": ("--ctx-size", "-c", "--context"),
    "ngl": ("--n-gpu-layers", "-ngl"),
    "moe": ("--n-cpu-moe", "-ncmoe"),
    "fa": ("--flash-attn", "-fa"),
    "ctk": ("--cache-type-k", "-ctk"),
    "ctv": ("--cache-type-v", "-ctv"),
    "b": ("--batch-size", "-b"),
    "ub": ("--ubatch-size", "-ub"),
    "t": ("--threads", "-t"),
    "tb": ("--threads-batch", "-tb"),
}


def is_true(value):
    return str(value).strip().lower() in TRUE


def is_false(value):
    return str(value).strip().lower() in FALSE


def safe_int(value, default=0):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def first_value(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def has_flag(text, *names):
    for name in names:
        pattern = r"(?<![\w-])" + re.escape(name) + r"(?![\w-])"
        if re.search(pattern, text):
            return True
    return False


def run_help(binary):
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return ""
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def detect_server_caps(binary):
    text = run_help(binary)
    return {
        "model": has_flag(text, "--model", "-m"),
        "mmproj": has_flag(text, "--mmproj"),
        "mmproj-offload": has_flag(text, "--mmproj-offload"),
        "no-mmproj-offload": has_flag(text, "--no-mmproj-offload"),
        "override-tensor": has_flag(text, "--override-tensor", "-ot"),
        "load-mode": has_flag(text, "--load-mode", "-lm"),
        "ctx-size": has_flag(text, "--ctx-size", "-c"),
        "n-gpu-layers": has_flag(text, "--n-gpu-layers", "-ngl"),
        "n-cpu-moe": has_flag(text, "--n-cpu-moe", "-ncmoe"),
        "batch-size": has_flag(text, "--batch-size", "-b"),
        "ubatch-size": has_flag(text, "--ubatch-size", "-ub"),
        "threads": has_flag(text, "--threads", "-t"),
        "threads-batch": has_flag(text, "--threads-batch", "-tb"),
        "parallel": has_flag(text, "--parallel", "-np"),
        "flash-attn": has_flag(text, "--flash-attn", "-fa"),
        "cache-type-k": has_flag(text, "--cache-type-k", "-ctk"),
        "cache-type-v": has_flag(text, "--cache-type-v", "-ctv"),
        "kv-unified": has_flag(text, "--kv-unified", "-kvu"),
        "no-kv-unified": has_flag(text, "--no-kv-unified", "-no-kvu"),
        "jinja": has_flag(text, "--jinja"),
        "cache-reuse": has_flag(text, "--cache-reuse"),
        "ctx-checkpoints": has_flag(text, "--ctx-checkpoints", "-ctxcp"),
        "checkpoint-min-step": has_flag(text, "--checkpoint-min-step", "-cms"),
        "image-min-tokens": has_flag(text, "--image-min-tokens"),
    }


def detect_bench_caps(binary):
    text = run_help(binary)
    caps = {}
    for key, aliases in BENCH_FLAG_ALIASES.items():
        for alias in aliases:
            if has_flag(text, alias):
                caps[key] = alias
                break
    return caps


def request(port, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:700]}") from exc


def gpu_mem(index):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i", str(index),
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip().splitlines()[0]
    return tuple(int(part.strip()) for part in output.split(","))


def ram_available_mib():
    text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^MemAvailable:\s+(\d+)", text, re.M)
    if not match:
        raise RuntimeError("Не удалось прочитать MemAvailable из /proc/meminfo")
    return int(match.group(1)) // 1024


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tail_file(path, chars=5000):
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - chars))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


# Ключи, которые тюнер не подбирает сам, но обязан передать дословно.
# Без этого тюнер меряет НЕ ту конфигурацию, которую потом разворачивают:
# пресет с spec-type = ngram-mod и mmproj тюнился как чисто текстовый без
# ускорителя, а результаты потом выдавались за «замеренные».
PASSTHROUGH_KEYS = (
    "spec-type", "model-draft",
    "spec-ngram-mod-n-max", "spec-ngram-mod-n-min", "spec-ngram-mod-n-match",
    "spec-draft-n-max", "spec-draft-n-min", "spec-draft-p-min",
    "fit",
)

_KIND_CACHE = {}


def _flag_kinds(binary):
    """{каноническое имя: нужен ли флагу значение}. Пусто, если схема недоступна."""
    key = str(binary)
    if key in _KIND_CACHE:
        return _KIND_CACHE[key]
    kinds = {}
    try:
        import sys as _sys
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parent.parent
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from lib import schema as _schema
        flags, meta = _schema.load(binary)
        if not meta.get("error"):
            uniq = {id(v): v for v in flags.values()}
            for f in uniq.values():
                kinds[f.canonical] = f.kind
    except Exception:
        kinds = {}
    _KIND_CACHE[key] = kinds
    return kinds


def build_server_command(binary, opts, port, caps, args):
    cmd = [str(binary), "--host", "127.0.0.1", "--port", str(port)]

    cmd += ["--model", str(opts["model"])]

    if args.keep_mmproj:
        if opts.get("mmproj"):
            cmd += ["--mmproj", str(opts["mmproj"])]
        if opts.get("image-min-tokens") and caps.get("image-min-tokens"):
            cmd += ["--image-min-tokens", str(opts["image-min-tokens"])]

        mm = str(opts.get("mmproj-offload", "")).strip().lower()
        if mm in TRUE and caps.get("mmproj-offload"):
            cmd.append("--mmproj-offload")
        elif mm in FALSE and caps.get("no-mmproj-offload"):
            cmd.append("--no-mmproj-offload")

    if args.allow_ot and opts.get("ot") and caps.get("override-tensor"):
        cmd += ["--override-tensor", str(opts["ot"])]

    # BUGFIX (тест 2026-09-30): раньше брался только args.load_mode (default mmap),
    # игнорируя load-mode из INI (у пресета none). Теперь INI уважаем, CLI приоритетнее.
    eff_load_mode = args.load_mode if args.load_mode != "default" else str(opts.get("load-mode", "default"))
    if caps.get("load-mode") and eff_load_mode != "default":
        cmd += ["--load-mode", eff_load_mode]

    cmd += ["--ctx-size", str(opts["c"])]
    cmd += ["--n-gpu-layers", str(opts.get("n-gpu-layers", args.ngl))]

    if caps.get("n-cpu-moe"):
        cmd += ["--n-cpu-moe", str(opts.get("n-cpu-moe", 0))]

    ubatch = safe_int(opts.get("ubatch-size", 2048), 2048)
    batch = max(safe_int(opts.get("b", 2048), 2048), ubatch)

    cmd += ["--batch-size", str(batch)]
    cmd += ["--ubatch-size", str(ubatch)]

    if caps.get("threads"):
        cmd += ["--threads", str(opts.get("t", args.threads))]

    if caps.get("threads-batch"):
        cmd += ["--threads-batch", str(opts.get("threads-batch", args.threads_batch))]

    if caps.get("parallel"):
        cmd += ["--parallel", str(opts.get("parallel", "1"))]

    if caps.get("flash-attn"):
        fa = str(opts.get("fa", args.flash_attn)).strip().lower()
        if fa not in {"on", "off", "auto"}:
            fa = args.flash_attn
        cmd += ["--flash-attn", fa]

    kv_k = str(opts.get("cache-type-k", args.kv_cache_type))
    kv_v = str(opts.get("cache-type-v", kv_k))
    if caps.get("cache-type-k"):
        cmd += ["--cache-type-k", kv_k]
    if caps.get("cache-type-v"):
        cmd += ["--cache-type-v", kv_v]

    if caps.get("kv-unified"):
        kvu = str(opts.get("kv-unified", args.kv_unified)).strip().lower()
        if kvu in TRUE:
            cmd.append("--kv-unified")
        elif kvu in FALSE and caps.get("no-kv-unified"):
            cmd.append("--no-kv-unified")
        # auto: флаг не передаём.

    if caps.get("jinja") and is_true(opts.get("jinja", args.jinja)):
        cmd.append("--jinja")

    if caps.get("cache-reuse"):
        reuse = safe_int(opts.get("cache-reuse", 0), 0)
        if reuse > 0:
            cmd += ["--cache-reuse", str(reuse)]

    # BUGFIX (тест 2026-09-30): раньше checkpoints всегда гасились в 0
    # "для чистоты бенча", из-за чего 114k ctx OOM там, где пресет с
    # ctx-checkpoints=16 грузится. Теперь уважаем INI.
    if caps.get("ctx-checkpoints"):
        if args.keep_checkpoints or opts.get("ctx-checkpoints") is not None:
            if opts.get("ctx-checkpoints") is not None:
                cmd += ["--ctx-checkpoints", str(opts["ctx-checkpoints"])]
        else:
            cmd += ["--ctx-checkpoints", "0"]
    if caps.get("checkpoint-min-step") and opts.get("checkpoint-min-step") is not None:
        cmd += ["--checkpoint-min-step", str(opts["checkpoint-min-step"])]

    # ускорители и прочее, что тюнер не варьирует, — переносим из секции
    kinds = _flag_kinds(binary)
    for key in PASSTHROUGH_KEYS:
        if key not in opts:
            continue
        val = str(opts[key]).strip()
        canon = "--" + key
        kind = kinds.get(canon)
        if kind == "flag" or val == "":
            cmd.append(canon)
        elif kind is None:
            # схема недоступна — предполагаем, что значение нужно
            cmd += [canon, val]
        else:
            cmd += [canon, val]

    # mmproj и режим его выгрузки: гейтим --keep-mmproj, иначе текстовый
    # замер не должен платить память за зрение
    if args.keep_mmproj and opts.get("mmproj"):
        pass  # уже добавлено выше
    elif opts.get("mmproj") and caps.get("mmproj"):
        if args.allow_vision:
            cmd += ["--mmproj", str(opts["mmproj"])]

    return cmd


def make_prompt(target, nonce, port, tokenize_timeout=240):
    """
    Строит промпт около target токенов.
    Секрет вставляется целостным блоком и не может быть обрезан.
    ОПТИМИЗАЦИЯ: калибровка токенайзера кэшируется на порт (1 раз на сервер),
    вместо ~15-25 /tokenize запросов на пробу делаем 3-5.
    """
    if target < 512:
        raise RuntimeError("target должен быть >= 512")

    secret = "K" + hashlib.sha256(nonce.encode()).hexdigest()[:18].upper()
    prefix = f"REQUEST_ID={nonce}\n"
    suffix = "\nReply with ONLY the value of VERIFICATION_SECRET:\n"
    secret_block = f"\nVERIFICATION_SECRET={secret}\n"
    line_fmt = (
        "Log entry {i}: module=router task=validate "
        "state={state} ref={ref}.\n"
    )

    def build(n):
        if n < 0:
            raise ValueError("Отрицательное число строк")
        entries = [
            line_fmt.format(
                i=i,
                state=(i * 17) % 997,
                ref=(i * 7919) % 100003,
            )
            for i in range(n)
        ]
        insert_at = min(n, n * 3 // 4)
        entries.insert(insert_at, secret_block)
        return prefix + "".join(entries) + suffix

    def token_count(text):
        response = request(
            port,
            "/tokenize",
            {"content": text},
            timeout=tokenize_timeout,
        )
        toks = response.get("tokens")
        if not isinstance(toks, list):
            raise RuntimeError(f"Неожиданный ответ /tokenize: {str(response)[:200]}")
        return len(toks)

    # per_line зависит только от токенайзера+шаблона строки, overhead —
    # от длины nonce. Порт меняется каждый evaluate, поэтому per_line
    # кэшируем глобально, а overhead меряем (1 запрос) каждый раз.
    overhead = token_count(build(0))
    if overhead >= target:
        raise RuntimeError(f"Служебный overhead слишком велик: {overhead}/{target}")
    cached_pl = _TOKEN_CALIB.get("per_line")
    if cached_pl is None:
        cal_n = 256
        cal_count = token_count(build(cal_n))
        per_line = max(1.0, (cal_count - overhead) / cal_n)
        _TOKEN_CALIB["per_line"] = per_line
    else:
        per_line = cached_pl

    est = max(0, int((target - overhead) / per_line))
    lines = est
    count = token_count(build(lines))

    # Быстрая сходимость: максимум 4 итерации вместо 8 + бинпоиск.
    for _ in range(4):
        if target * 0.95 <= count <= target:
            break
        if count > target:
            # целимся в 97% чтобы гарантированно попасть снизу
            want = target * 0.97
            delta = max(1, int((count - want) / per_line * 1.1))
            lines = max(0, lines - delta)
        else:
            if count >= target * 0.80:
                # близко — принимаем как есть, дальше правит статистика
                break
            want = target * 0.97
            delta = max(1, int((want - count) / per_line * 1.1))
            lines += delta
        # защита от убегания при дрейфе per_line на больших N
        if lines > 200000:
            raise RuntimeError(f"Оценка строк убежала: {lines} для target={target}")
        count = token_count(build(lines))
        # обновляем per_line онлайн для точности
        if lines > 0 and count > overhead:
            per_line = max(1.0, (count - overhead) / max(1, lines))
            _TOKEN_CALIB["per_line"] = per_line

    if count < target * 0.80 or count > target:
        raise RuntimeError(f"Не удалось подобрать размер промпта: {count}/{target}")

    prompt = build(lines)
    # финальный recount не нужен: count уже соответствует этому промпту
    # (nonce/секрет фиксированы). Проверяем только переполнение.
    if count > target:
        raise RuntimeError(f"Финальный промпт превысил цель: {count}/{target}")

    return prompt, secret, count


def completion(port, prompt, secret, estimated, timeout, n_predict, warm=False):
    start = time.monotonic()
    # BUGFIX (кампания qwen3.8 2026-09-30): сырой /completion обходит чат-шаблон
    # даже при --jinja, и instruct-модель в ответ на «Reply with ONLY the value»
    # выдаёт 1 токен и стоп — needle FAIL на 100% проб, кандидаты отсекались
    # как «Провалена проверка извлечения факта». Проба идёт через
    # /v1/chat/completions (шаблон применяется), cache_prompt=False — холодный
    # замер (warm-кэш занижает prompt_n). См. digest/procedure/
    # needle-harness-correctness.md. Диагностика: reasoning_content отдельно от
    # content, т.к. reasoning-модель может потратить весь бюджет на размышление.
    response = request(
        port,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0.0,
            "cache_prompt": False,
        },
        timeout=timeout,
    )
    elapsed = time.monotonic() - start

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""

    timings = response.get("timings") or {}
    prompt_n = safe_int(timings.get("prompt_n"), 0)
    predicted_n = safe_int(timings.get("predicted_n"), 0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)

    if predicted_n <= 0 or predicted_ms <= 0:
        raise RuntimeError(
            f"Нет пригодных timings генерации: {timings!r}; "
            f"finish_reason={choice.get('finish_reason')!r}, "
            f"content={content[:80]!r}, reasoning={reasoning[:80]!r}"
        )

    if not warm and (prompt_n < estimated * 0.75 or prompt_ms <= 0):
        raise RuntimeError(
            "Cold-проба неожиданно использовала prompt cache: "
            f"prompt_n={prompt_n}, ожидалось около {estimated}"
        )

    return {
        "estimated_tokens": estimated,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prefill_tps": (
            round(prompt_n * 1000 / prompt_ms, 2) if prompt_ms > 0 else None
        ),
        "gen_tps": round(predicted_n * 1000 / predicted_ms, 2),
        "wall_s": round(elapsed, 2),
        "needle_ok": secret in content,
        "answer": content[:160],
        "reasoning": reasoning[:120],
        "finish_reason": choice.get("finish_reason"),
        "low_decode_sample": predicted_n < 32,
    }


def probe_series(
    port,
    target,
    tag,
    timeout,
    n_predict,
    tokenize_timeout,
    repeats,
    warmup_tokens=0,
    warmup_n_predict=4,
):
    warmup_info = None

    if warmup_tokens > 0:
        try:
            wp, ws, we = make_prompt(
                warmup_tokens,
                f"{tag}-warm",
                port,
                tokenize_timeout,
            )
            wr = completion(
                port,
                wp,
                ws,
                we,
                timeout,
                warmup_n_predict,
                warm=True,
            )
            warmup_info = {
                "ok": True,
                "gen_tps": wr["gen_tps"],
                "tokens": we,
            }
        except Exception as exc:
            warmup_info = {
                "ok": False,
                "error": str(exc),
            }

    samples = []
    repeats = max(1, repeats)

    for i in range(repeats):
        nonce = f"{tag}-{i}-{uuid.uuid4().hex[:8]}"
        prompt, secret, estimated = make_prompt(
            target,
            nonce,
            port,
            tokenize_timeout,
        )
        res = completion(
            port,
            prompt,
            secret,
            estimated,
            timeout,
            n_predict,
            warm=False,
        )
        samples.append(res)

    gen_values = [float(s["gen_tps"]) for s in samples]
    prefill_values = [
        float(s["prefill_tps"])
        for s in samples
        if s.get("prefill_tps") is not None
    ]
    prompt_values = [float(s["prompt_n"]) for s in samples]
    estimated_values = [float(s["estimated_tokens"]) for s in samples]

    return {
        "target": target,
        "repeats": repeats,
        "gen_tps": round(statistics.median(gen_values), 2),
        "gen_tps_min": round(min(gen_values), 2),
        "gen_tps_max": round(max(gen_values), 2),
        "prefill_tps": (
            round(statistics.median(prefill_values), 2)
            if prefill_values else None
        ),
        "prompt_n_median": round(statistics.median(prompt_values), 1),
        "estimated_median": round(statistics.median(estimated_values), 1),
        "needle_ok": all(bool(s["needle_ok"]) for s in samples),
        "answer": samples[0]["answer"],
        "warmup": warmup_info,
        "samples": samples,
    }


def monitor_resources(gpu, stop, vram_samples, ram_samples):
    while not stop.is_set():
        try:
            vram_samples.append(gpu_mem(gpu)[2])
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            pass
        try:
            ram_samples.append(ram_available_mib())
        except (OSError, RuntimeError):
            pass
        stop.wait(1.0)


def extract_log_signals(path):
    warnings = []
    fa_signal = "unknown"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "flash_attention_signal": fa_signal,
            "log_warnings": warnings,
        }

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()

        if "flash" in low and "attention" in low:
            if any(
                x in low
                for x in (
                    "fallback",
                    "fall back",
                    "not supported",
                    "unsupported",
                    "cpu",
                )
            ):
                fa_signal = "fallback"
                warnings.append(line[:300])
            elif fa_signal == "unknown":
                fa_signal = "enabled_or_mentioned"

        if any(
            x in low
            for x in (
                "out of memory",
                "oom",
                "failed to allocate",
                "could not allocate",
            )
        ):
            warnings.append(line[:300])

        if "checkpoint" in low and any(
            x in low for x in ("ram", "invalid", "fail", "error")
        ):
            warnings.append(line[:300])

    return {
        "flash_attention_signal": fa_signal,
        "log_warnings": warnings[:10],
    }


def evaluate(binary, base_opts, context, variant, args, run_id, caps, short_only=False):
    port = free_port()

    cfg = dict(base_opts)
    ubatch = safe_int(variant.get("ubatch", 2048), 2048)
    batch = max(safe_int(variant.get("b", 2048), 2048), ubatch)

    cfg.update({
        "c": str(context),
        "n-gpu-layers": str(variant.get("ngl", args.ngl)),
        "n-cpu-moe": str(variant.get("moe", 0)),
        "b": str(batch),
        "ubatch-size": str(ubatch),
        "t": str(variant.get("threads", args.threads)),
        "threads-batch": str(variant.get("threads_batch", args.threads_batch)),
        "parallel": "1",
        "fa": str(variant.get("fa", args.flash_attn)),
        "cache-type-k": str(variant.get("kv_cache_type", args.kv_cache_type)),
        "cache-type-v": str(variant.get("kv_cache_type", args.kv_cache_type)),
        "kv-unified": str(variant.get("kv_unified", args.kv_unified)),
        "jinja": str(args.jinja),
        "cache-reuse": str(variant.get("reuse", args.cache_reuse)),
    })

    if not args.keep_mmproj:
        for key in ("mmproj", "image-min-tokens", "mmproj-offload"):
            cfg.pop(key, None)

    if not args.allow_ot:
        cfg.pop("ot", None)

    log_path = args.out / f"run-{run_id:03d}.log"

    selected_config_keys = (
        "c",
        "n-gpu-layers",
        "n-cpu-moe",
        "b",
        "ubatch-size",
        "t",
        "threads-batch",
        "parallel",
        "fa",
        "cache-type-k",
        "cache-type-v",
        "kv-unified",
        "jinja",
        "cache-reuse",
    )

    eff_lm = args.load_mode if args.load_mode != "default" else str(cfg.get("load-mode", "default"))
    result = {
        "config": {key: cfg[key] for key in selected_config_keys},
        "config_extra": {
            "load-mode": eff_lm if caps.get("load-mode") else "default",
        },
        "log": str(log_path),
    }

    vram_samples = []
    ram_samples = []
    stop = threading.Event()
    watcher = None
    process = None

    with log_path.open("w", encoding="utf-8") as log:
        try:
            cmd = build_server_command(binary, cfg, port, caps, args)
            result["command_preview"] = " ".join(cmd)

            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )

            watcher = threading.Thread(
                target=monitor_resources,
                args=(args.gpu, stop, vram_samples, ram_samples),
                daemon=True,
            )
            watcher.start()

            deadline = time.monotonic() + args.load_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server завершился: rc={process.returncode}"
                    )
                try:
                    health = request(port, "/health", timeout=3)
                    if isinstance(health, dict) and health.get("status") == "ok":
                        break
                    # некоторые сборки отдают {"status":"loading"} или список
                    if isinstance(health, dict) and health.get("status") in ("loading", "starting"):
                        pass
                except Exception:
                    # BUGFIX: раньше ловились только 4 типа, а обрывы соединения
                    # (ConnectionResetError, http.client.*) роняли весь прогон.
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Истёк таймаут загрузки модели")

            try:
                props = request(port, "/props", timeout=10)
                props_text = json.dumps(props, ensure_ascii=False).lower()
                result["moe_detected"] = "expert" in props_text
            except Exception:
                result["moe_detected"] = None

            signals = extract_log_signals(log_path)
            result.update(signals)

            if args.strict_flash_attn and result.get("flash_attention_signal") == "fallback":
                raise RuntimeError("Flash Attention упал в fallback/CPU")

            _, _, free = gpu_mem(args.gpu)
            ram_now = ram_available_mib()
            vram_samples.append(free)
            ram_samples.append(ram_now)

            if free < args.reserve:
                raise RuntimeError(
                    f"Мало VRAM после загрузки: {free} < {args.reserve} MiB"
                )
            if ram_now < args.ram_reserve:
                raise RuntimeError(
                    f"Мало RAM после загрузки: {ram_now} < {args.ram_reserve} MiB"
                )

            short_target = max(512, min(args.short_probe_tokens, context - 512))
            deep_target = min(context - 512, args.max_probe_tokens)
            nonce = uuid.uuid4().hex

            result["short"] = probe_series(
                port,
                short_target,
                f"{nonce}-short",
                adaptive_timeout(args.request_timeout, short_target, args.n_predict),
                args.n_predict,
                args.tokenize_timeout,
                args.probe_repeats,
                warmup_tokens=args.warmup_tokens,
            )

            if not short_only and deep_target > short_target + 1024:
                result["deep"] = probe_series(
                    port,
                    deep_target,
                    f"{nonce}-deep",
                    adaptive_timeout(args.request_timeout, deep_target, args.n_predict),
                    args.n_predict,
                    args.tokenize_timeout,
                    args.probe_repeats,
                    warmup_tokens=0,
                )
            else:
                result["deep"] = result["short"]

            signals = extract_log_signals(log_path)
            result.update(signals)

            if args.strict_flash_attn and result.get("flash_attention_signal") == "fallback":
                raise RuntimeError("Flash Attention упал в fallback/CPU после проб")

            _, _, free_after = gpu_mem(args.gpu)
            ram_after = ram_available_mib()
            vram_samples.append(free_after)
            ram_samples.append(ram_after)

            result["min_observed_free_mib"] = (
                min(vram_samples) if vram_samples else free_after
            )
            result["min_ram_available_mib"] = (
                min(ram_samples) if ram_samples else ram_after
            )

            if result["min_observed_free_mib"] < args.reserve:
                raise RuntimeError(
                    "Наблюдаемый запас VRAM ниже резерва: "
                    f"{result['min_observed_free_mib']} MiB"
                )
            if result["min_ram_available_mib"] < args.ram_reserve:
                raise RuntimeError(
                    "Наблюдаемый запас RAM ниже резерва: "
                    f"{result['min_ram_available_mib']} MiB"
                )
            if not result["short"]["needle_ok"] or not result["deep"]["needle_ok"]:
                raise RuntimeError(
                    "Провалена проверка извлечения факта: "
                    f"short={result['short']['answer']!r}, "
                    f"deep={result['deep']['answer']!r}"
                )
            if float(result["deep"]["gen_tps"]) < args.min_tps:
                raise RuntimeError(
                    f"Генерация медленнее {args.min_tps} токенов/с"
                )

            result["ok"] = True

        except Exception as exc:
            result["ok"] = False
            result["error"] = str(exc)
            result["log_tail"] = tail_file(log_path)

        finally:
            stop.set()
            if process is not None:
                stop_process(process)
            if watcher is not None:
                watcher.join(timeout=2)
            if vram_samples:
                result["min_observed_free_mib"] = min(vram_samples)
            if ram_samples:
                result["min_ram_available_mib"] = min(ram_samples)

    return result


def find_tool(server, name, explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        return None

    candidates = [
        Path(server).parent / name,
    ]

    for p in candidates:
        try:
            p = p.expanduser()
        except RuntimeError:
            pass
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    # BUGFIX: раньше проверялся Path(name) относительно cwd, а не $PATH.
    found = shutil.which(name)
    if found:
        p = Path(found)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    return None


def add_bench_flag(cmd, caps, key, value):
    if value is None:
        return
    flag = caps.get(key)
    if flag:
        cmd.extend((flag, str(value)))


def build_bench_command(tool, caps, model, ctx, variant, args):
    cmd = [
        str(tool),
        "-m", str(model),
        "-o", "json",
        "-r", str(args.screen_repeats),
        "-p", str(args.screen_pp),
        "-n", str(args.screen_tg),
    ]

    add_bench_flag(cmd, caps, "ctx", ctx)
    add_bench_flag(cmd, caps, "ngl", variant.get("ngl", args.ngl))
    add_bench_flag(cmd, caps, "moe", variant.get("moe", 0))
    add_bench_flag(cmd, caps, "fa", variant.get("fa", args.flash_attn))
    add_bench_flag(cmd, caps, "ctk", variant.get("kv_cache_type", args.kv_cache_type))
    add_bench_flag(cmd, caps, "ctv", variant.get("kv_cache_type", args.kv_cache_type))

    ubatch = safe_int(variant.get("ubatch", 2048), 2048)
    batch = max(safe_int(variant.get("b", 2048), 2048), ubatch)

    add_bench_flag(cmd, caps, "b", batch)
    add_bench_flag(cmd, caps, "ub", ubatch)
    add_bench_flag(cmd, caps, "t", variant.get("threads", args.threads))
    add_bench_flag(cmd, caps, "tb", variant.get("threads_batch", args.threads_batch))

    return cmd


def parse_bench_json(text):
    text = text.strip()
    start_array = text.find("[")
    start_obj = text.find("{")

    start = -1
    if start_array != -1 and start_obj != -1:
        start = min(start_array, start_obj)
    elif start_array != -1:
        start = start_array
    elif start_obj != -1:
        start = start_obj

    if start == -1:
        raise RuntimeError("В выводе llama-bench не найден JSON")

    # BUGFIX: раньше json.loads падал при trailing-логax после JSON.
    # Используем raw_decode чтобы пережить мусор в хвосте.
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"В выводе llama-bench битый JSON: {exc}") from exc

    if isinstance(data, dict):
        data = (
            data.get("results")
            or data.get("rows")
            or data.get("data")
            or []
        )

    if not isinstance(data, list):
        raise RuntimeError(f"Неожиданная структура llama-bench JSON: {type(data)}")

    rows = []

    for item in data:
        if not isinstance(item, dict):
            continue

        ts = first_value(item, ("avg_ts", "t/s", "ts", "speed"))
        if ts is None:
            continue

        sd = first_value(item, ("stddev_ts", "stddev", "stdev")) or 0.0

        n_prompt = first_value(item, ("n_prompt", "pp", "prompt_tokens"))
        n_gen = first_value(item, ("n_gen", "tg", "generation_tokens"))

        test = str(item.get("test", ""))

        if n_prompt is None:
            m = re.search(r"pp\s+(\d+)", test)
            n_prompt = int(m.group(1)) if m else 0

        if n_gen is None:
            m = re.search(r"tg\s+(\d+)", test)
            n_gen = int(m.group(1)) if m else 0

        try:
            n_prompt = int(n_prompt)
            n_gen = int(n_gen)
            ts = float(ts)
            sd = float(sd)
        except (TypeError, ValueError):
            continue

        if n_prompt > 0 and n_gen == 0:
            kind = "pp"
        elif n_gen > 0 and n_prompt == 0:
            kind = "tg"
        elif n_prompt > 0 and n_gen > 0:
            kind = "pg"
        else:
            kind = "other"

        rows.append({
            "kind": kind,
            "ts": ts,
            "sd": sd,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "test": test,
            "raw": item,
        })

    return rows


def bench_score(rows):
    """Взвешенный скор screening. BUGFIX: раньше 'pg' строки игнорировались,
    и если llama-bench вернул только pp+tg в одном тесте — все скор=0."""
    def best(kind):
        values = []
        for r in rows:
            if r["kind"] == kind and r["ts"] > 0:
                values.append(max(0.0, r["ts"] - 0.25 * r["sd"]))
        return max(values, default=0.0)

    pp = best("pp")
    tg = best("tg")
    pg_rows = [r for r in rows if r["kind"] == "pg" and r["ts"] > 0]
    if pg_rows:
        # pg-тест меряет смешанную пропускную способность; используем как
        # прокси и для pp, и для tg если чистых замеров нет.
        pg_best = max(max(0.0, r["ts"] - 0.25 * r["sd"]) for r in pg_rows)
        # Эвристика: n_prompt>>n_gen => ближе к pp, иначе к tg.
        for r in pg_rows:
            val = max(0.0, r["ts"] - 0.25 * r["sd"])
            if r["n_prompt"] >= r["n_gen"] * 8 and pp <= 0:
                pp = max(pp, val)
            if r["n_gen"] >= 16 and tg <= 0:
                tg = max(tg, val)
        if pp <= 0 or tg <= 0:
            # fallback: считаем pg скором напрямую
            return round(pg_best, 4), pp or pg_best, tg or pg_best

    if tg <= 0:
        return 0.0, pp, tg

    score = (max(tg, 0.1) ** 0.65) * (max(pp, 0.1) ** 0.35)
    return score, pp, tg


def rank_score(result, weights=(0.55, 0.30, 0.15)):
    """Финальный скор для выбора победителя.
    weights = (gen, prefill, стабильность/память). Возвращает float.
    Штрафы: большой разброс gen (min/max), низкий запас VRAM, большой moe
    без выигрыша в скорости учитывается на этапе фильтрации top-10%."""
    w_gen, w_pp, w_stab = weights
    deep = result.get("deep") or {}
    gen = float(deep.get("gen_tps") or 0.0)
    pp_raw = deep.get("prefill_tps")
    pp = float(pp_raw) if pp_raw is not None else 0.0
    gmin = float(deep.get("gen_tps_min") or gen)
    gmax = float(deep.get("gen_tps_max") or gen)
    stability = (gmin / gmax) if gmax > 0 else 0.0
    free = float(result.get("min_observed_free_mib") or 0.0)
    # headroom: 0..1, насыщение после 2 ГБ запаса
    headroom = min(1.0, max(0.0, free / 2048.0))
    # логарифмическая шкала чтобы prefill 500 vs 1000 не забивал gen
    import math as _m
    s = (w_gen * _m.log1p(gen) + w_pp * _m.log1p(pp)
         + w_stab * (0.7 * stability + 0.3 * headroom))
    return round(s, 5)


def run_llama_bench(tool, caps, model, ctx, variant, args):
    cmd = build_bench_command(tool, caps, model, ctx, variant, args)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.bench_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Таймаут llama-bench: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-bench rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[-700:]}"
        )

    rows = parse_bench_json(proc.stdout)
    if not rows:
        raise RuntimeError("llama-bench вернул пустой/непарсимый JSON")

    return rows, proc.stdout


def build_sweep_command(tool, caps, model, ctx, variant, args):
    cmd = [str(tool), "-m", str(model)]

    add_bench_flag(cmd, caps, "ctx", ctx)
    add_bench_flag(cmd, caps, "ngl", variant.get("ngl", args.ngl))
    add_bench_flag(cmd, caps, "moe", variant.get("moe", 0))
    add_bench_flag(cmd, caps, "fa", variant.get("fa", args.flash_attn))
    add_bench_flag(cmd, caps, "ctk", variant.get("kv_cache_type", args.kv_cache_type))
    add_bench_flag(cmd, caps, "ctv", variant.get("kv_cache_type", args.kv_cache_type))

    ubatch = safe_int(variant.get("ubatch", 2048), 2048)
    batch = max(safe_int(variant.get("b", 2048), 2048), ubatch)

    add_bench_flag(cmd, caps, "b", batch)
    add_bench_flag(cmd, caps, "ub", ubatch)
    add_bench_flag(cmd, caps, "t", variant.get("threads", args.threads))
    add_bench_flag(cmd, caps, "tb", variant.get("threads_batch", args.threads_batch))

    return cmd


def run_llama_sweep_bench(tool, caps, model, ctx, variant, args, log_path):
    cmd = build_sweep_command(tool, caps, model, ctx, variant, args)

    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.bench_timeout,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1

    text = log_path.read_text(encoding="utf-8", errors="replace")

    values = [
        float(x)
        for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*t/s", text)
    ]

    return {
        "command": cmd,
        "log": str(log_path),
        "returncode": rc,
        "median_tps": round(statistics.median(values), 2) if values else None,
        "min_tps": round(min(values), 2) if values else None,
        "max_tps": round(max(values), 2) if values else None,
        "samples": len(values),
    }


def variant_key(variant):
    return json.dumps(variant, sort_keys=True, ensure_ascii=False)


def config_to_variant(cfg):
    """Собрать variant-словарь из config прошлого результата (для --reuse-results)."""
    return {
        "moe": safe_int(cfg.get("n-cpu-moe", 0), 0),
        "ubatch": safe_int(cfg.get("ubatch-size", 2048), 2048),
        "threads": safe_int(cfg.get("t", 8), 8),
        "threads_batch": safe_int(cfg.get("threads-batch", 16), 16),
        "kv_cache_type": str(cfg.get("cache-type-k", "q8_0")),
        "fa": str(cfg.get("fa", "on")),
        "reuse": safe_int(cfg.get("cache-reuse", 0), 0),
        "ngl": safe_int(cfg.get("n-gpu-layers", 999), 999),
        "kv_unified": str(cfg.get("kv-unified", "on")),
    }


def log_progress(path, message):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def default_variant(args):
    return {
        "moe": 0,
        "ubatch": 2048,
        "b": 2048,
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "kv_cache_type": args.kv_cache_type,
        "fa": args.flash_attn,
        "reuse": 0,
        "ngl": args.ngl,
        "kv_unified": args.kv_unified,
    }


def adaptive_timeout(base_timeout, target_tokens, n_predict):
    """Таймаут под размер пробы: ~50 tok/s prefill минимум + запас.
    Защита от зависания на малых пробах и достаточный лимит на больших."""
    need = target_tokens / 25.0 + n_predict * 1.5 + 180
    return int(max(180, min(base_timeout, need)))


def moe_value_list(args, enabled):
    """
    Список значений n-cpu-moe для перебора.

    --moe-values (явный список) имеет приоритет над диапазоном
    0..--max-cpu-moe с шагом --moe-step: для пресетов, где 0 заведомо OOM
    (MoE 35B на 12 ГБ), диапазон от нуля тратит время на гарантированный
    отсев. Пустой список недопустим — иначе кандидатов не будет.
    """
    if not enabled:
        return [0]
    raw = str(getattr(args, "moe_values", "") or "").strip()
    if raw:
        try:
            values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--moe-values: ожидались целые числа через запятую")
        if not values or any(v < 0 for v in values):
            raise SystemExit(
                "--moe-values: нужен непустой список неотрицательных целых чисел"
            )
        return sorted(set(values))
    return list(range(0, args.max_cpu_moe + 1, args.moe_step))


def ubatch_value_list(args):
    """
    Список значений ubatch-size. --ubatch-values (явный список) имеет приоритет
    над дефолтной градацией; b всегда не меньше 2048 и не меньше ubatch.
    """
    raw = str(getattr(args, "ubatch_values", "") or "").strip()
    if raw:
        try:
            values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--ubatch-values: ожидались целые числа через запятую")
        if not values or any(v < 128 for v in values):
            raise SystemExit(
                "--ubatch-values: нужен непустой список значений >= 128"
            )
        return sorted(set(values))
    # Раньше перебирались только 2048 (+4096): для 12 ГБ и большого ctx
    # оптимум часто 512/1024. Добавляем градации.
    if args.try_4096:
        return [512, 1024, 2048, 4096]
    # короткий список, чтобы не взрывать screening
    return [1024, 2048]


def make_screen_candidates(args, server_caps, bench_caps, allow_ot):
    screen_moe_enabled = (
        server_caps.get("n-cpu-moe")
        and bench_caps.get("moe")
        and not allow_ot
    )

    moe_values = moe_value_list(args, screen_moe_enabled)

    # Раньше перебирались только 2048 (+4096): для 12 ГБ и большого ctx
    # оптимум часто 512/1024 (см. ubatch_value_list).
    ubatch_values = ubatch_value_list(args)

    thread_values = [args.threads]
    if args.try_threads_16 and 16 not in thread_values:
        thread_values.append(16)

    kv_values = [args.kv_cache_type]
    if args.try_f16 and "f16" not in kv_values:
        kv_values.append("f16")

    fa_values = [args.flash_attn]
    # fa on/off заметно влияет на prefill на 3060 — пробуем оба если дефолт on
    if args.flash_attn == "on" and "off" not in fa_values:
        fa_values.append("off")

    out = []
    seen = set()

    for moe in moe_values:
        for ubatch in ubatch_values:
            for threads in thread_values:
                for kv in kv_values:
                    for fa in fa_values:
                        v = default_variant(args)
                        v.update({
                            "moe": moe,
                            "ubatch": ubatch,
                            "b": max(2048, ubatch),
                            "threads": threads,
                            "threads_batch": max(args.threads_batch, threads),
                            "kv_cache_type": kv,
                            "fa": fa,
                        })
                        k = variant_key(v)
                        if k in seen:
                            continue
                        seen.add(k)
                        out.append(v)

    limit = max(args.screen_top * 8, 32)
    return out[:limit], screen_moe_enabled


def expand_server_candidates(
    selected,
    args,
    server_caps,
    bench_caps,
    allow_ot,
    screen_moe_enabled,
    limit,
):
    out = []
    seen = set()

    def add(v):
        if len(out) >= limit:
            return False
        k = variant_key(v)
        if k in seen:
            return True
        seen.add(k)
        out.append(dict(v))
        return True

    server_moe_enabled = server_caps.get("n-cpu-moe") and not allow_ot

    if selected:
        if screen_moe_enabled:
            for v in selected:
                if not add(v):
                    break
        else:
            moe_values = moe_value_list(args, server_moe_enabled)
            base_count = max(1, min(3, args.screen_top))
            for base in selected[:base_count]:
                for moe in moe_values:
                    v = dict(base)
                    v["moe"] = moe
                    if not add(v):
                        break
                if len(out) >= limit:
                    break

    if not out:
        moe_values = moe_value_list(args, server_moe_enabled)
        ubatch_values = ubatch_value_list(args)
        full = False
        for moe in moe_values:
            for ubatch in ubatch_values:
                v = default_variant(args)
                v["moe"] = moe
                v["ubatch"] = ubatch
                v["b"] = max(2048, ubatch)
                if not add(v):
                    full = True
                    break
            if full:
                break

    return out


def replace_section_keys(original, section, updates):
    if not updates:
        return original

    lines = original.splitlines(keepends=True)
    header = re.compile(r"^\s*\[([^\]]+)\]\s*(?:[;#].*)?$")
    key_line = re.compile(r"^(\s*)([A-Za-z][\w-]*)(\s*=\s*)(.*?)(\r?\n)?$")

    active = False
    found = False
    seen = set()
    output = []
    insert_index = None

    for line in lines:
        match = header.match(line.rstrip("\r\n"))
        if match:
            if active:
                insert_index = len(output)
            active = match.group(1) == section
            if active:
                found = True

        if active:
            km = key_line.match(line)
            if km and km.group(2) in updates:
                key = km.group(2)
                seen.add(key)
                line = (
                    f"{km.group(1)}{key}{km.group(3)}"
                    f"{updates[key]}{km.group(5) or ''}"
                )

        output.append(line)

        if active:
            insert_index = len(output)

    if not found:
        raise RuntimeError("Секция исчезла из файла")

    if insert_index is None:
        insert_index = len(output)

    missing = set(updates) - seen
    if missing:
        new_lines = [f"{key} = {updates[key]}\n" for key in sorted(missing)]
        output[insert_index:insert_index] = new_lines

    return "".join(output)


def csv_ints(value, name, allow_empty=False):
    value = (value or "").strip()
    if not value:
        if allow_empty:
            return []
        raise SystemExit(f"--{name}: ожидается список целых чисел через запятую")
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--{name}: ожидались целые числа через запятую")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Двухэтапный подбор параметров llama.cpp: "
            "llama-bench/llama-sweep-bench screening + llama-server validation."
        )
    )
    p.add_argument("ini", type=Path)
    p.add_argument("section")
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--contexts", default="",
                   help="Дополнительные контексты, например 65536,114688")
    p.add_argument("--deep", action="store_true",
                   help="Проверять контекст почти до полного c")
    p.add_argument("--apply", action="store_true",
                   help="Записать победителя в INI; требует --deep")

    p.add_argument("--reserve", type=int, default=1024,
                   help="Минимум свободной VRAM, MiB")
    p.add_argument("--ram-reserve", type=int, default=3072,
                   help="Минимум доступной RAM, MiB")
    p.add_argument("--min-tps", type=float, default=20.0)

    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--max-cpu-moe", type=int, default=64)
    p.add_argument("--moe-step", type=int, default=4)
    p.add_argument("--moe-values", default="",
                   help="Явный список n-cpu-moe через запятую, например 20,24,28; "
                        "имеет приоритет над диапазоном 0..--max-cpu-moe с шагом --moe-step")

    p.add_argument("--short-probe-tokens", type=int, default=4096)
    p.add_argument("--max-probe-tokens", type=int, default=8192)
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--probe-repeats", type=int, default=3)
    p.add_argument("--warmup-tokens", type=int, default=512)
    p.add_argument("--tokenize-timeout", type=int, default=240)

    p.add_argument("--ngl", type=int, default=999)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--threads-batch", type=int, default=16)
    p.add_argument("--try-threads-16", action="store_true")

    p.add_argument("--try-4096", action="store_true")
    p.add_argument("--ubatch-values", default="",
                   help="Явный список ubatch-size через запятую, например 2048,4096; "
                        "имеет приоритет над градацией по --try-4096")
    p.add_argument("--kv-cache-type", default="q8_0")
    p.add_argument("--try-f16", action="store_true")

    p.add_argument("--kv-unified", choices=("on", "off", "auto"), default="on")
    p.add_argument("--flash-attn", choices=("on", "auto", "off"), default="on")
    p.add_argument("--strict-flash-attn", action="store_true")
    p.add_argument("--jinja", choices=("on", "off"), default="on")
    # BUGFIX (тест 2026-09-30): default был mmap и всегда перетирал load-mode
    # из INI (у пресета none). Теперь default=default (=не передавать флаг,
    # сервер сам возьмёт mmap; если в INI есть значение — уважаем его).
    p.add_argument("--load-mode", default="default",
                   choices=("default", "auto", "mmap", "none", "mlock", "dio"))

    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--keep-mmproj", action="store_true",
                   help="держать mmproj во время замеров (по умолчанию гасится)")
    p.add_argument("--with-spec", action="store_true",
                   help="измерять вместе со spec-type/model-draft из секции "
                        "(по умолчанию ускоритель исключается из замера)")
    p.add_argument("--allow-vision", action="store_true",
                   help="не гасить mmproj и мерить с ним (для пресетов со зрением)")
    p.add_argument("--allow-ot", action="store_true")
    p.add_argument("--cache-reuse", type=int, default=0)

    p.add_argument("--bench",
                   choices=("auto", "llama-bench", "llama-sweep-bench", "both", "none"),
                   default="auto")
    p.add_argument("--bench-bin", type=Path, default=None)
    p.add_argument("--sweep-bin", type=Path, default=None)
    p.add_argument("--screen-top", type=int, default=8)
    p.add_argument("--prefilter-ctx", type=int, default=0,
                   help="Контекст быстрого предфильтра (только короткая проба, без deep); "
                        "0 = автоматически --screen-ctx, если он меньше максимального контекста")
    p.add_argument("--no-prefilter", action="store_true",
                   help="Отключить предфильтр: сразу полная проверка на максимальном контексте")
    p.add_argument("--screen-ctx", type=int, default=32768)
    p.add_argument("--screen-pp", type=int, default=4096)
    p.add_argument("--screen-tg", type=int, default=64)
    p.add_argument("--screen-repeats", type=int, default=2)
    p.add_argument("--bench-timeout", type=int, default=1800)
    p.add_argument("--skip-screening", action="store_true")
    p.add_argument("--screen-only", action="store_true")

    p.add_argument("--load-timeout", type=int, default=240)
    p.add_argument("--request-timeout", type=int, default=7200)
    p.add_argument("--optimize-for", choices=("balanced", "gen", "prefill", "vram"),
                   default="balanced",
                   help="Приоритет выбора победителя: balanced=TTFT+gen+запас, "
                        "gen=скорость декодинга, prefill=скорость промпта, vram=запас памяти")
    p.add_argument("--out", type=Path, default=Path("tune-results"))

    args = p.parse_args()

    if args.apply and not args.deep:
        p.error("--apply допускается только с --deep")
    if args.max_candidates < 1:
        p.error("--max-candidates должен быть >= 1")
    if args.moe_step < 1:
        p.error("--moe-step должен быть >= 1")
    if str(args.ubatch_values or "").strip():
        try:
            _ub = [int(x.strip()) for x in args.ubatch_values.split(",") if x.strip()]
        except ValueError:
            p.error("--ubatch-values: ожидались целые числа через запятую")
        if not _ub or any(v < 128 for v in _ub):
            p.error("--ubatch-values: нужен непустой список значений >= 128")
    if str(args.moe_values or "").strip():
        try:
            _moe = [int(x.strip()) for x in args.moe_values.split(",") if x.strip()]
        except ValueError:
            p.error("--moe-values: ожидались целые числа через запятую")
        if not _moe or any(v < 0 for v in _moe):
            p.error("--moe-values: нужен непустой список неотрицательных целых чисел")
    if args.max_cpu_moe < 0:
        p.error("--max-cpu-moe должен быть >= 0")
    if args.n_predict < 16:
        p.error("--n-predict должен быть >= 16")
    if args.probe_repeats < 1:
        p.error("--probe-repeats должен быть >= 1")
    if args.short_probe_tokens < 512:
        p.error("--short-probe-tokens должен быть >= 512")
    if args.cache_reuse < 0:
        p.error("--cache-reuse должен быть >= 0")
    if args.screen_top < 1:
        p.error("--screen-top должен быть >= 1")

    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        p.error(f"Не найден исполняемый llama-server: {args.server}")

    args.ini = args.ini.resolve()
    args.server = args.server.resolve()

    try:
        original_bytes = args.ini.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать INI: {exc}")

    try:
        original = original_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise SystemExit("INI должен быть UTF-8")

    original_hash = hashlib.sha256(original_bytes).hexdigest()

    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.read_string(original)

    if args.section not in ini:
        p.error("Секция не найдена")

    opts = dict(ini[args.section])

    unknown = set(opts) - allowed_keys_for(args.server)
    if unknown:
        p.error(
            "Неизвестные ключи INI; добавьте явную поддержку или удалите: "
            + ", ".join(sorted(unknown))
        )

    if not opts.get("model"):
        p.error("В секции нет model")
    if "c" not in opts:
        p.error("В секции нет c")

    model_path = Path(str(opts["model"])).expanduser()
    if not model_path.is_absolute():
        model_path = (args.ini.parent / model_path).resolve()
    if not model_path.is_file():
        p.error(f"Не найден модельный файл: {model_path}")
    opts["model"] = str(model_path)

    if args.keep_mmproj and opts.get("mmproj"):
        mmproj_path = Path(str(opts["mmproj"])).expanduser()
        if not mmproj_path.is_absolute():
            mmproj_path = (args.ini.parent / mmproj_path).resolve()
        if not mmproj_path.is_file():
            p.error(f"Не найден mmproj: {mmproj_path}")
        opts["mmproj"] = str(mmproj_path)

    # Спекулятивное декодирование по умолчанию НЕ измеряется, и это
    # осознанно: проба строится из повторяющегося текста, на котором ngram
    # выигрывает искусственно, а draft-модель в памяти искажает замер VRAM.
    # Поэтому параметры ускорителя подбираются отдельно, а --with-spec
    # включает его в замер сознательно.
    for key in ("model-draft", "spec-type"):
        if str(opts.get(key, "")).strip() and not args.with_spec:
            print(f"ВНИМАНИЕ: {key} игнорируется в этом тесте; MTP/speculative "
                  f"сравнивай отдельно (или --with-spec, чтобы мерить с ним).")
            opts.pop(key, None)

    server_caps = detect_server_caps(args.server)

    allow_ot = bool(args.allow_ot and opts.get("ot") and server_caps.get("override-tensor"))
    if opts.get("ot") and not allow_ot:
        if args.allow_ot:
            print("ВНИМАНИЕ: сборка не поддерживает --override-tensor; ot игнорируется.")
        else:
            print("ВНИМАНИЕ: ot игнорируется; добавьте --allow-ot для явного теста.")
        opts.pop("ot", None)

    if not args.keep_mmproj and opts.get("mmproj"):
        print("ВНИМАНИЕ: mmproj исключён из текстового бенча; используйте --keep-mmproj для VLM.")

    if not server_caps.get("n-cpu-moe"):
        print("ВНИМАНИЕ: сборка не поддерживает --n-cpu-moe; MoE offload sweep выключен.")

    moe_sweep_server = bool(server_caps.get("n-cpu-moe")) and not allow_ot

    # BUGFIX (тест 2026-09-30): --ngl 999 по умолчанию игнорировал n-gpu-layers
    # из INI (у пресета 99) и приводил к OOM. Берём базу из INI если CLI дефолтный.
    if args.ngl == 999 and opts.get("n-gpu-layers") is not None:
        try:
            args.ngl = int(str(opts["n-gpu-layers"]).strip())
            print(f"ngl из INI: {args.ngl}")
        except ValueError:
            pass

    try:
        base_ctx = int(opts["c"])
    except ValueError:
        p.error("Ключ c должен быть целым числом")

    extra_contexts = csv_ints(args.contexts, "contexts", allow_empty=True)
    contexts = sorted({base_ctx, *extra_contexts})

    if any(ctx < 4096 or ctx > 262144 for ctx in contexts):
        p.error("Контекст должен быть в пределах 4096..262144")

    if args.deep:
        args.max_probe_tokens = max(contexts)

    if args.max_probe_tokens < args.short_probe_tokens:
        args.short_probe_tokens = max(512, args.max_probe_tokens // 2)

    args.out = args.out / (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    args.out.mkdir(parents=True, exist_ok=False)

    total, used, free = gpu_mem(args.gpu)
    print(f"GPU {args.gpu}: всего {total} MiB, свободно {free} MiB")
    print(f"RAM доступно: {ram_available_mib()} MiB")
    print(f"CPU: {os.cpu_count()} логических потоков")
    print(f"Результаты: {args.out}")

    selected = []
    bench_caps = {}
    screen_moe_enabled = False

    if args.bench != "none" and not args.skip_screening:
        bench_tool = None
        sweep_tool = None

        if args.bench in ("auto", "llama-bench", "both"):
            bench_tool = find_tool(args.server, "llama-bench", args.bench_bin)

        if args.bench in ("auto", "llama-sweep-bench", "both"):
            sweep_tool = find_tool(args.server, "llama-sweep-bench", args.sweep_bin)

        screen_report = {
            "bench_tool": str(bench_tool) if bench_tool else None,
            "sweep_tool": str(sweep_tool) if sweep_tool else None,
            "candidates": [],
            "selected": [],
        }

        candidates, screen_moe_enabled = make_screen_candidates(
            args,
            server_caps,
            detect_bench_caps(bench_tool) if bench_tool else {},
            allow_ot,
        )

        needed_ctx = args.screen_pp + args.screen_tg + 64
        screen_ctx = min(max(contexts), max(args.screen_ctx, needed_ctx))
        if screen_ctx < needed_ctx:
            screen_ctx = max(contexts)

        scored = []

        if bench_tool:
            bench_caps = detect_bench_caps(bench_tool)
            if not bench_caps:
                print("ВНИМАНИЕ: не удалось определить флаги llama-bench; screening пропущен.")
            else:
                if not screen_moe_enabled and moe_sweep_server:
                    print(
                        "ВНИМАНИЕ: llama-bench не поддерживает --n-cpu-moe; "
                        "screening будет только по ubatch/threads/KV, затем MoE sweep в llama-server."
                    )

                for idx, variant in enumerate(candidates, start=1):
                    print(f"[screen {idx}/{len(candidates)}] llama-bench {variant}", flush=True)

                    entry = {
                        "variant": variant,
                        "ok": False,
                        "score": 0.0,
                        "pp_tps": 0.0,
                        "tg_tps": 0.0,
                        "error": None,
                    }

                    try:
                        rows, raw = run_llama_bench(
                            bench_tool,
                            bench_caps,
                            opts["model"],
                            screen_ctx,
                            variant,
                            args,
                        )
                        score, pp, tg = bench_score(rows)

                        entry.update({
                            "ok": score > 0,
                            "score": round(score, 4),
                            "pp_tps": round(pp, 2),
                            "tg_tps": round(tg, 2),
                            "rows": len(rows),
                        })

                        if score > 0:
                            scored.append((score, variant, entry))

                    except Exception as exc:
                        entry["error"] = str(exc)

                    screen_report["candidates"].append(entry)

                scored.sort(key=lambda x: x[0], reverse=True)
                selected = [v for _, v, _ in scored[:args.screen_top]]

        elif sweep_tool:
            print("ВНИМАНИЕ: llama-bench не найден; использую llama-sweep-bench только как эвристику.")
            sweep_caps = detect_bench_caps(sweep_tool)

            for idx, variant in enumerate(candidates, start=1):
                print(f"[screen {idx}/{len(candidates)}] llama-sweep-bench {variant}", flush=True)

                log_path = args.out / f"screen-sweep-{idx:03d}.log"
                entry = {
                    "variant": variant,
                    "ok": False,
                    "score": 0.0,
                    "sweep": None,
                    "error": None,
                }

                try:
                    sweep = run_llama_sweep_bench(
                        sweep_tool,
                        sweep_caps,
                        opts["model"],
                        screen_ctx,
                        variant,
                        args,
                        log_path,
                    )
                    median = sweep.get("median_tps")
                    score = float(median) if median else 0.0

                    entry.update({
                        "ok": score > 0,
                        "score": round(score, 4),
                        "sweep": sweep,
                    })

                    if score > 0:
                        scored.append((score, variant, entry))

                except Exception as exc:
                    entry["error"] = str(exc)

                screen_report["candidates"].append(entry)

            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [v for _, v, _ in scored[:args.screen_top]]

        else:
            print("ВНИМАНИЕ: llama-bench/llama-sweep-bench не найдены; иду сразу в llama-server.")

        screen_report["selected"] = selected
        (args.out / "screening.json").write_text(
            json.dumps(screen_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if args.screen_only:
            print("\nSCREENING ЗАВЕРШЁН. llama-server не запускался, INI не изменён.")
            print(json.dumps({
                "bench_tool": screen_report["bench_tool"],
                "sweep_tool": screen_report["sweep_tool"],
                "selected": selected,
            }, indent=2, ensure_ascii=False))
            return

    candidate_limit = max(args.max_candidates, args.screen_top) * max(1, len(contexts))
    server_candidates = expand_server_candidates(
        selected,
        args,
        server_caps,
        bench_caps,
        allow_ot,
        screen_moe_enabled,
        candidate_limit,
    )

    if not server_candidates:
        server_candidates = [default_variant(args)]

    results = []
    seen = set()
    run_id = 0

    # Быстрый предфильтр на уменьшенном контексте: отбрасываем OOM/медленные
    # варианты до запуска дорогой full-context валидации.
    max_ctx = max(contexts)
    prefilter_ctx = None
    if not args.no_prefilter:
        want = args.prefilter_ctx or args.screen_ctx
        if 4096 <= want < max_ctx:
            prefilter_ctx = want

    if prefilter_ctx and server_candidates:
        prefilter_results = []

        for pf_idx, variant in enumerate(server_candidates, start=1):
            run_id += 1
            print(
                f"[P{pf_idx}/{len(server_candidates)}] prefilter ctx={prefilter_ctx} {variant}",
                flush=True,
            )
            result = evaluate(
                args.server,
                opts,
                prefilter_ctx,
                variant,
                args,
                run_id,
                server_caps,
                short_only=True,
            )
            prefilter_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            (args.out / "prefilter.json").write_text(
                json.dumps(prefilter_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        survivors = [
            v for v, r in zip(server_candidates, prefilter_results) if r.get("ok")
        ]
        if survivors:
            print(
                f"\nПредфильтр прошли {len(survivors)}/{len(server_candidates)}; "
                "далее полная проверка на максимальном контексте.",
                flush=True,
            )
            server_candidates = survivors
        else:
            print("\nПредфильтр не прошёл ни один вариант.", flush=True)
            results = prefilter_results
            server_candidates = []

    # Larger contexts first, because --apply requires deep validation.
    # BUGFIX: раньше был общий лимит max_candidates на все контексты —
    # второй контекст почти никогда не тестировался. Теперь бюджет на контекст.
    validation_contexts = sorted(contexts, reverse=True)
    per_ctx_budget = max(1, args.max_candidates // max(1, len(validation_contexts)))

    full_idx = 0
    total_budget = args.max_candidates
    for ctx in validation_contexts:
        ctx_count = 0
        for variant in server_candidates:
            if len(results) >= total_budget:
                break
            if ctx_count >= per_ctx_budget and len(validation_contexts) > 1:
                break

            key = (ctx, variant_key(variant))
            if key in seen:
                continue
            seen.add(key)

            run_id += 1
            full_idx += 1
            ctx_count += 1
            print(f"[{full_idx}/{total_budget}] ctx={ctx} {variant}", flush=True)

            result = evaluate(
                args.server,
                opts,
                ctx,
                variant,
                args,
                run_id,
                server_caps,
            )
            results.append(result)

            print(json.dumps(result, ensure_ascii=False), flush=True)
            (args.out / "results.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if len(results) >= args.max_candidates:
            break

    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    good = [r for r in results if r.get("ok")]
    if not good:
        raise SystemExit("Нет прошедших конфигураций. Проверьте логи: " + str(args.out))

    largest_ctx = max(int(r["config"]["c"]) for r in good)
    comparable = [r for r in good if int(r["config"]["c"]) == largest_ctx]

    # BUGFIX: старый отбор брал max() по кортежу (prefill, gen, free, ...),
    # где +1 t/s prefill перевешивал гигабайты VRAM. Теперь: фильтр top-10%
    # по gen + взвешенный rank_score с учётом стабильности и запаса памяти.
    weights_map = {
        "balanced": (0.55, 0.30, 0.15),
        "gen": (0.80, 0.10, 0.10),
        "prefill": (0.20, 0.65, 0.15),
        "vram": (0.35, 0.20, 0.45),
    }
    weights = weights_map.get(args.optimize_for, weights_map["balanced"])
    fastest_gen = max(float(r["deep"]["gen_tps"]) for r in comparable)
    close = [
        r for r in comparable
        if float(r["deep"]["gen_tps"]) >= fastest_gen * 0.90
    ]
    for r in close:
        r["_rank"] = rank_score(r, weights)
    ranked = sorted(close, key=lambda r: r["_rank"], reverse=True)
    winner = ranked[0]

    print(f"\nТОП-{min(5, len(ranked))} (optimize-for={args.optimize_for}):")
    for i, r in enumerate(ranked[:5], 1):
        d = r.get("deep", {})
        print(f"  {i}. rank={r['_rank']} gen={d.get('gen_tps')} "
              f"prefill={d.get('prefill_tps')} free={r.get('min_observed_free_mib')} "
              f"cfg={r.get('config')}")
    print("\nВЫБРАНО:\n" + json.dumps(winner, indent=2, ensure_ascii=False))
    # готовая команда запуска
    if winner.get("command_preview"):
        print("\nЗАПУСК:\n" + winner["command_preview"])
    (args.out / "ranking.json").write_text(
        json.dumps(ranked[:10], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # BUGFIX: если выжил только предфильтр (short_only на урезанном ctx),
    # его c == prefilter_ctx, а не целевой контекст — запрещаем --apply,
    # иначе в INI запишется заниженный контекст.
    only_prefilter = all(r.get("short") is not None and r.get("deep") is r.get("short")
                         for r in good) if good else False
    if only_prefilter and int(winner["config"]["c"]) != max(contexts):
        print("ВНИМАНИЕ: есть только результаты предфильтра "
              f"(c={winner['config']['c']}, целевой max={max(contexts)}). "
              "Полная валидация не выполнена — INI не изменяю.")
        return

    if not args.deep:
        print("Это короткий тест: INI не изменён.")
        return

    if not args.apply:
        print("INI не изменён; для записи добавьте --apply.")
        return

    if hashlib.sha256(args.ini.read_bytes()).hexdigest() != original_hash:
        raise SystemExit("INI изменился во время теста — запись отменена")

    updates = {}
    for key, value in winner["config"].items():
        if key not in WRITABLE_KEYS:
            continue
        if key == "n-cpu-moe" and not server_caps.get("n-cpu-moe"):
            continue
        if key == "threads-batch" and not server_caps.get("threads-batch"):
            continue
        if key == "kv-unified" and not server_caps.get("kv-unified"):
            continue
        if key == "cache-reuse" and not server_caps.get("cache-reuse"):
            continue
        if key == "load-mode":
            if not server_caps.get("load-mode") or args.load_mode == "default":
                continue
        updates[key] = str(value)

    if "ubatch-size" in updates and "b" in updates:
        if int(updates["b"]) < int(updates["ubatch-size"]):
            updates["b"] = updates["ubatch-size"]

    edited = replace_section_keys(original, args.section, updates)
    if edited == original:
        print("Выбран исходный пресет; изменений не требуется.")
        return

    backup = args.ini.with_name(
        args.ini.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    )
    shutil.copy2(args.ini, backup)

    fd, temporary = tempfile.mkstemp(
        prefix=f".{args.ini.name}.",
        dir=args.ini.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(edited)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, args.ini.stat().st_mode)
        os.replace(temporary, args.ini)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(f"Обновлён: {args.ini}")
    print(f"Резервная копия: {backup}")


if __name__ == "__main__":
    main()