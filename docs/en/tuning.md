# Tuning: method, traps, and established laws

The engine is `tools/tune_models.py`. It works in two stages, and that is the
main reason it is worth having: a cheap screening pass throws out the garbage,
an expensive validation pass throws out what looks plausible but is not.

## Two stages

1. **Screening** — `llama-bench` / `llama-sweep-bench` at a short context.
   Cheap; removes obviously weak combinations of `n-cpu-moe` × `ubatch` × `t`.
2. **Validation** — a live `llama-server`: full context, real KV, free VRAM
   reading, a retrieval test (needle in a haystack) at both short and deep
   context, and a draft-quality check.

Artifacts of each run: `~/.config/llama/tune-results/<stage>/<time-pid>/` —
`results.json` (machine-readable), `screening.json`, `run-NNN.log`.

`results.json` is what `llamastery ingest` collects into `measurements.json`.
Fields worth reading:

| Field | Meaning |
|---|---|
| `short.gen_tps` / `deep.gen_tps` | generation speed at 4k and at full context |
| `short.prefill_tps` | prompt processing speed |
| `min_observed_free_mib` | the lowest free VRAM during the run — the key number |
| `needle_ok` | did the retrieval test pass (model not broken at long context) |
| `ok` / `error` | did the run hold up, with the reason |
| `config`, `config_extra` | the candidate's full flag set |

The **worst** value is taken from measurements, not the average: you plan by
the worst case.

## What the tuner does not do

* It does not search over `c` — that is a user decision, not a grid search.
* It does not choose between models.
* It does not check whether several presets fit at once — that is
  `llamastery budget`.

## Traps found in practice

### `spec-type` with several values plus an external MTP draft

In the Faks build, `--spec-type ngram-mod` together with `--model-draft` (an
MTP head) segfaults on load. An external MTP head is itself twice as slow
(24 t/s versus 48), so `ngram-mod` without a draft is the only working
accelerator on such a model.

### `ubatch-size`

`ubatch-size = 2048` at 12 GiB VRAM is a known cause of OOM; 1024 works. But
that is a heuristic, not a law: if a successful measurement exists for a
specific configuration, `llamastery validate` stays quiet.

The measured nuance: `ubatch-size = 2048` alone is fine — it was measured at
26.4 t/s with 1616 MiB free. It is `2048` **together with** a speculative
decoder that falls over, because the verification graph needs a larger compute
buffer than was allocated.

### `parallel`

`parallel > 1` multiplies the KV pool. On 12 GiB, keep `parallel = 1`.

### `mmproj` on CPU

`mmproj-offload = 0` (projector in RAM) barely affects generation speed —
measured: 38.9 versus 38.8 t/s. It uses no VRAM. For vision models on a tight
card this is almost always the best choice.

### `cache-reuse` is incompatible with mmproj

Verified in the sources, `tools/server/server-context.cpp:1220`:

```cpp
if (params_base.n_cache_reuse) {
    params_base.n_cache_reuse = 0;
    SRV_WRN("cache_reuse is not supported by multimodal, it will be disabled");
}
```

When a multimodal model is loaded, `cache-reuse` is forcibly disabled with a
warning in the log. The second barrier (same file, ~3164): `can_cache_reuse`
requires `!slot.prompt.tokens.has_mtmd`, so any request with an image disables
reuse for that slot.

**An important correction to the wording:** the limitation applies to mmproj, not
to `ngram-mod` and not to speculative decoding. An older entry in models.ini
conflated these.

### `spec-draft-*` has no effect together with `ngram-mod`

Verified: `common/speculative.cpp`, function `common_speculative_n_max()`
only iterates over the **selected** types from `spec->types`:

```cpp
case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:      // and EAGLE3, DFLASH
    n_max = max(n_max, spec->draft.n_max);   // draft-model knobs
    break;
case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
    n_max = max(n_max, spec->ngram_mod.n_max);  // its own knobs
    break;
```

So with `spec-type = ngram-mod`, `spec-draft-n-max` and `spec-draft-p-min` are
inert — they control a different draft model. ngram-mod's own knobs are
`--spec-ngram-mod-n-max` (64), `--spec-ngram-mod-n-min` (48),
`--spec-ngram-mod-n-match` (24). A preset with `spec-draft-n-max = 1` and
`spec-type = ngram-mod` silently runs on defaults.

### What ngram-mod actually gives

Measured A/B on llama-upstream 0.5.0-dev, Qwen3.8 MiniPlus, 114688 tokens,
same prompt, the only difference being the `spec-type` line:

| probe text | with ngram-mod | without ngram |
|---|---|---|
| lorem (repeating) | **84–92 t/s** | 26.3–26.5 t/s |
| non-repeating | 30.0 t/s | 28.5–30.5 t/s |
| 4k, repeating | 98.6 t/s | — |
| 4k, non-repeating | 49.5 t/s | — |

Conclusion: on repeating text (code, templates, dialogs) the accelerator gives
×3.2; on unique text it gives exactly nothing and does no harm. So
`spec-type = ngram-mod` is worth enabling for repeating workloads, not
"always just in case".

A methodological caveat: the tuner's probe (needle in a haystack) and the
`probe` command give different numbers for the same preset — 48 versus 26.5 t/s.
These are different tasks, not a measurement discrepancy: only compare within a
single method. That is why the A/B above is `probe` against `probe`.

Extended measurement on two models (Qwen3.8 and Tiel-Coder NanoPlus, faks):

| prompt | with ngram-mod | without ngram |
|---|---|---|
| repeating text | 96–107 t/s | 26–28 t/s |
| real code at 110k | 25.6 t/s | 28.8 t/s |

The accelerator **hurts** on real code: it pays for every miss, and it also
makes the measurement less stable (spread 5.3–6.6 t/s between repeats versus
0.1–0.5 without it).

### How `--n-cpu-moe` works

Not a numeric field in params, but per-layer buffer overrides
(`common/arg.cpp:2607`): for every layer `i < N`, `llm_ffn_exps_block_regex(i)`
is added to `tensor_buft_overrides` with a forced CPU buffer. That is why
searching the code for `n_cpu_moe` finds nothing — look for
`tensor_buft_overrides` / `ffn_exps`.

Measured effect on free VRAM (Tiel, c=114688, b=2048, runs with `ok=True`):
moe24 → 2963 MiB, moe32 → 5028, moe40 → 7129. The slope is 258–263 MiB per
layer. The analytic share of expert weights (~0.978) overstates the effect by
about 13%, hence the `offload_realization = 0.89` coefficient that
`llamastery calibrate --from-tune` fits automatically.

### Antispam

`repeat-last-n = 256` + `repeat-penalty = 1.02` is a working combination
against looping. On broken JSON or tool calls, lower it to 1.05; below that is
useless.

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

## Measurement artifacts

* Prefill measured on a warm cache gives falsely high numbers. A real cold
  prefill at 65k takes about 50 minutes (~20 t/s).
* A draft acceptance of 1.00 is an artifact of a repeating prompt; do not
  believe it. On varied text acceptance is 0.93.
* The first request of a session does not benefit from drafting (the index is
  empty).

## Tuner limitations worth knowing

* **The fact-extraction check trips reasoning models.** The tuner asks the
  model to insert a secret into a long prompt and expects it back within 64
  tokens (`--n-predict`). Models with `<think>` spend the whole budget on
  reasoning, the answer is empty, `finish_reason: length` — and the run is
  discarded even though its speed was fine. Fix with `--n-predict 256`. Tell
  "the model is bad" from "the model is thinking" by `gen_tps`, not by `ok`.
* **The deep stage is expensive.** At 114688 a cold prefill takes tens of
  minutes, so `--deep --apply` at 128k means hours. The practical substitute:
  pick parameters with the tuner at a short context, then measure speed at
  real depth separately — `llamastery probe --tokens 110000` on a live server.
  That is cheaper and measures what actually matters.
* **The set of allowed keys grows with the build's schema.** The tuner takes
  flags from the target binary's `--help`, so a preset with fork flags
  (`load-mode`, `ctx-checkpoints`, `no-mmproj-offload`) passes, and that
  knowledge is not duplicated in the tuner code.
* **`--with-spec` keeps the accelerator in the measurement.** By default
  `spec-type` is dropped from tuner runs: the probe is built from repeating
  text, on which ngram wins artificially, and a draft model in memory skews the
  VRAM reading. Pass `--with-spec` to measure with it deliberately.

## Order of work when picking a preset

1. `llamastery schema --grep <substring>` — check the flag exists in this build
   at all (forks add their own, and `--draft` / `--draft-min` are declared
   removed in newer versions).
2. Fix `c` (the task's capacity) and the model.
3. `llamastery budget <section>` — estimate memory; if short, lower
   `n-cpu-moe` (the strongest lever on MoE) or `c`.
4. `llamastery tune ... --extra deep` — the tuner sweeps
   `n-cpu-moe`/`ubatch`/`t`.
5. `llamastery ingest --apply` — record the measurements.
6. `llamastery budget` across all sections — check that several presets fit at
   once under `--models-max`.