# VRAM budget, measurements, and slot planning

## What `budget` shows

```bash
llamastery budget                              # every preset + slot forecast for --models-max
llamastery budget <preset> --models-max 2 --reserve 1024 --explain
```

It breaks VRAM into components: weights (accounting for `n-cpu-moe`), KV cache,
mmproj, compute buffer. It also answers a separate question — will N presets
fit at once? The router keeps every loaded model as its own process, so memory
adds up across all of them.

```bash
llamastery calibrate <preset> --free-mib 1147
llamastery calibrate a=1147,b=992 --free-mib     # one by one, per section
llamastery calibrate --from-tune                # from tuner runs
llamastery calibrate --from-log                 # from the server's own report
```

Until `calibrate` has run, the estimate is too low — the tool prints a warning,
and that is not a bug. A large spread of residuals means the structural model
cannot be trusted and you should rely on measurements.

## Measurement beats forecast

If a preset has a real `used_mib`, the "does it fit" decision uses it, and the
estimate is shown next to it for cross-checking. Without that, a working preset
with an imprecise model looks like it does not fit: that is what happened to
`tiel-coder-nanoplus-128ctx-mmproj-moe16`, where the estimate said 11.54 GiB
against 10.42 GiB actually used.

```bash
llamastery measure                    # read the card and record
llamastery measure --recalibrate      # also recompute the compute buffer
```

Measuring only makes sense with a preset loaded.

`calibrate --from-log` is more accurate than `--from-tune`: it reads
`CUDA0 compute buffer size` lines next to `n_ubatch` straight from the server
log, so the buffer is named by the server itself. The catch is that the server
does not print that breakdown on every load (in router mode the instance report
sometimes does not reach the shared log), so you get few data points. The older
method, taking the remainder from the total, includes constant overhead and
therefore overestimates the buffer by roughly 0.3 GiB.

## Speed at real depth

The tuner saves money by using shallow contexts, but real work happens at long
contexts. So depth is measured separately, on a live server.

```bash
llamastery probe --tokens 110000 --max-tokens 96 --repeats 2
llamastery probe --tokens 4096 --repeats 1          # short, for the curve
llamastery probe --from-file server.cpp --tokens 110000   # prompt from a real file
llamastery probe --image ~/photo.jpg                # check vision (mmproj)
llamastery probe ... --record --preset <section>   # store in measurements.json
```

The measurement is an ordinary request: the server counts timings itself,
hence `prompt_n / predicted_n / *_ms`.

**The prompt cache is switched off explicitly:** the request carries
`cache_prompt: false`. Without it, a repeat of the same text arrives from warm
KV with `prompt_n = 4` instead of 110000, and that run's prefill is several
times lower — on a graph it looks like a sudden speed-up. Such runs are marked
🔥 (warm) and excluded from prefill statistics. If every run in a series is warm,
the tool says plainly that there is no cold prefill in it.

**What you measure matters more than the repeat count.** A "no repeats" probe
built from a made-up word list partially loops the model, so the result depends
on how the build survives looping rather than on real speed. For an honest
number use `--from-file` with real code or prose; a continuation tail is added
at the cut point, otherwise the model considers the file finished and emits EOS
on the first token. Tails are tried in turn until the model speaks, because the
same word-like hints do not work on every build.

Measured difference at 114688 (Qwen3.8 and Tiel-Coder NanoPlus, faks):

| prompt | ngram-mod | without ngram |
|---|---|---|
| repeating text | 96–107 t/s | 26–28 t/s |
| real code at 110k | 25.6 t/s | 28.8 t/s |

The accelerator gives ×3–4 on repeating text and **hurts** on unique text: it
pays for every miss. Enable it for workloads with repetition, not by default.

## Crash journal

```bash
llamastery crashes                       # which presets bring the server down, and why
llamastery crashes --forget <preset>     # clear the entry: verified working
```

A preset can load, pass schema validation, and still fall over only at deep
context — that is how `ubatch 2048 + ngram-mod` kills CUDA at 114688. Such
crashes are remembered automatically (on a failed load, and when an instance
dies during `probe`), after which:

- `validate` warns about a preset that has taken the server down;
- `presets annotate` writes a 💥 (this preset has taken the server down)
  line into the preset block;
- a successful measurement clears the entry by itself — a verified preset must
  not keep being listed as crashing.

Stored in `~/.local/state/llamastery/crashes.json`.

## Collecting measurements

```bash
llamastery ingest             # what was found in tune-results and models.ini comments
llamastery ingest --apply     # store in ~/.local/state/llamastery/measurements.json
```

Measurements from `tune-results/*/results.json` (produced by the tuner) and from
comments above `models.ini` sections are merged into one store and substituted
into `budget` and `validate`. The key is the configuration signature, so a
preset differing by a single flag lands in its own record and does not inherit
someone else's numbers.

## The speed degradation law

An earlier `models.ini` header had `tg(D) = 1000 / (16.1 + 0.0025·D)`. The
formula is wrong: at D = 114688 it gives 3.3 t/s, while the same header
recorded 32.5 t/s — the coefficient is about 20× too low.

Refitting on four Tiel-Coder measurements (128k, RTX 3060, no ngram):

| D | measured |
|---|---|
| 9 864 | 48.6 t/s |
| 32 327 | 42.1 t/s |
| 63 932 | 36.1 t/s |
| 114 012 | 32.5 t/s |

gives `tg(D) = 1 / (0.01961 + 9.786e-8 · D)`:
8k → 49.0, 32k → 43.8, 65k → 38.4, 114688 → 32.4 t/s.

**But rely on measurements of your own model, not on the formula.** The Qwen3.8
family measures 39–41.5 t/s at 114688, higher than the Tiel fit predicts. The
law is good for order of magnitude and for comparing two configurations of one
model, but not for carrying numbers between models.

Practical consequence: **doubling the context costs about 1 t/s** at large
depths. So a 128k preset for long sessions loses almost nothing in speed.

## How the calculation works

```
bytes_per_token = n_attention_layers × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
cache_size      = bytes_per_token × ctx × slots
```

where `bytes(q8_0) = 1.0625`, `bytes(f16) = 2`, `bytes(q4_0) = 0.5625`.

Two traps:

* **GQA.** `n_kv_heads` is usually much smaller than `n_head` (16 versus 2 on
  Qwen3-35B-A3B), so KV is several times smaller than naive maths suggests.
* **Hybrid architectures.** In qwen35moe / Nemotron-H / Jamba some layers are
  SSM (Mamba-like); their state does not grow with context and they keep no KV.
  Full attention appears only every `full_attention_interval`-th layer. `budget`
  accounts for this; "count all the layers" underestimates memory several-fold.

`kv-unified` (enabled in most modern presets) means one shared KV pool for the
whole context: `parallel` stops multiplying the cache. Turn it off and the
multiplication by slot count returns.