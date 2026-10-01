# Preset format in llama.cpp

Three levels of configuration. Confusion between them is the source of almost
every misunderstanding, so start with this table.

| Level | File | Key | What it is |
|---|---|---|---|
| Per-model presets | `models.ini` / `presets.ini` | `--models-preset PATH` | section = model, `[*]` = shared defaults |
| Named, shareable | `preset.ini` in an empty HF repo | `-hf user/repo` | a preset treated as a "model" with a tag |
| Common to all binaries | `/etc/llama.cpp/config.ini`, `~/.config/llama.cpp/config.ini` | no flag | applies to `llama-cli` too; only `[*]` and the section before the first heading are read |

Order of application (weakest to strongest):
`config.ini` → environment variables → model preset → the router's own CLI.
That is, an argument given on the command line **beats** a value from a preset.

The source of truth on "who controls what" is `tools/server/server-models.cpp`,
function `unset_reserved_args()`:

* stripped from every preset: `ssl-key-file`, `ssl-cert-file`, `api-key`,
  `models-dir`, `models-max`, `models-preset`, `models-autoload`;
* overwritten by the router when spawning a child process: `port`, `host`,
  `alias`;
* in a per-model preset, `model` / `mmproj` / `hf-repo` **are** the model itself
  and are therefore legal (the router strips them only from the base preset).

## Three ways to write one key

Equivalent, mixable in one file:

```ini
ctx-size = 32768     # long
c = 32768            # short
LLAMA_ARG_CTX_SIZE = 32768   # environment variable name
```

Logic: first the forms are deduplicated (short and long count as one argument),
then merged with the router's base arguments.

Boolean values: `on` / `off`, `true` / `false`, `1` / `0`, `enabled` /
`disabled`. A flag with no value (`kv-unified`, `jinja`) — just leave the value
empty.

## Example

```ini
version = 1

; defaults shared by all models
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

## Where the router finds models

1. `~/.cache/llama.cpp` (or `LLAMA_CACHE`) — cached HF models;
2. `--models-dir PATH` — direct children only, **no recursion**;
3. `--models-preset` — sections with explicit paths (`m = ...`).

On name collisions the priority is: preset > models-dir > cache. For
multi-part models and `mmproj` the files go into a subdirectory, and the
projector filename should start with `mmproj`.

Related router flags: `--models-max N` (4 by default, `0` for no limit),
`--models-autoload` / `--no-models-autoload`.

## What matters for memory

Every loaded model is a **separate `llama-server` process** on its own free
port on `127.0.0.1`; the router only proxies requests. Hence:

* VRAM adds up across all simultaneously loaded models, not per "preset size";
* `LLAMA_SERVER_ROUTER_PORT` is passed down to the child processes;
* `parallel` inside one preset is a separate axis: it multiplies the KV pool
  inside one process.

`llamastery budget` answers exactly the question "will N presets fit at once",
which no existing GUI launcher answers.

## KV cache structure

```
bytes_per_token = n_attention_layers × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
cache_size      = bytes_per_token × ctx × slots
```

where `bytes(q8_0) = 1.0625`, `bytes(f16) = 2`, `bytes(q4_0) = 0.5625`.

Two traps:

* **GQA.** `n_kv_heads` is usually much smaller than `n_head` (16 versus 2 on
  Qwen3-35B-A3B), so KV is several times smaller than "naive" calculations.
* **Hybrid architectures.** In qwen35moe / Nemotron-H / Jamba some layers are
  SSM (Mamba-like), their state does not grow with context and they keep no KV.
  Full attention appears only every `full_attention_interval`-th layer.
  `llamastery budget` accounts for this; "count all the layers" underestimates
  memory several-fold.

`kv-unified` (enabled in most modern presets) means one shared KV pool for the
whole context: `parallel` stops multiplying the cache. Turn it off and the
multiplication by slot count returns.

## Preset operations

```bash
llamastery presets list                      # all sections
llamastery presets show <section>             # the section's keys
llamastery presets globals                   # the [*] section
llamastery presets export -o - <section>...   # export a subset
llamastery presets annotate --dry-run        # what the formatter would change
llamastery presets annotate --apply          # apply it
```

### Importing foreign presets

Sources: a local file, a URL, or `git-repo#branch:path/inside`.

```bash
llamastery presets import --source ./foreign-presets.ini --dry-run
llamastery presets import --source 'https://github.com/u/repo#main:presets.ini' --dry-run
llamastery presets import --source git@github.com:u/repo.git --only my-model-128k
llamastery presets import --source ./p.ini --on-conflict new      # do not overwrite
llamastery presets import --source ./p.ini --on-conflict overwrite --conflicts
```

Behaviour: `--dry-run` by default; `skip` on conflicts; a `.bak-<timestamp>`
is made before writing. `--rename old=new` renames sections. A missing target
file is created.

### Validation

```bash
llamastery validate                          # all sections of the current models.ini
llamastery validate <section> --json
llamastery validate --build ik               # validate against a specific build
llamastery validate --no-paths                # skip disk access (faster)
```

It catches: unknown keys, router control arguments (`api-key`, `models-max` are
stripped; `port`/`host`/`alias` are overwritten), GGUF files that do not
exist, `c` beyond the trained context, `n-cpu-moe` above the layer count,
known fork traps, and presets present in the crash journal.

Every build has its own flag schema, so a preset written for a fork will warn
on ik_llama: some flags simply do not exist there. The hint names the flag.

## Loading and unloading

```bash
llamastery runtime start --build faks      # start the router
llamastery runtime status                  # port, pid, build, what is in VRAM
llamastery load <preset>                   # load into VRAM
llamastery runtime unload [model]          # unload, the server keeps running
llamastery runtime restart --build faks
llamastery runtime stop
llamastery runtime logs -n 100
```

`llamastery` runs the server itself — separate managers (`llama`, `llama-faks`,
`llama-ik`) are neither required nor used. That is deliberate: per-build scripts
exist for very few forks, and the tool has to work for someone who never
installed them.

Before loading, it validates against **that** build's schema and forecasts
VRAM:

```
preset: qwen3.8-35B-A3B-miniplus-21-128ctx-ngram-mmproj
build: faks   binary: /home/axel/git/llama-faks/build/bin/llama-server
  VRAM: 8.67 GiB of 12.00 GiB, headroom +2.96 GiB
```

On errors the load does not happen (`--force` overrides).

Other models are unloaded first: there is one pool of VRAM. How many models can
be loaded at once is the router's `--models-max`.

Comment formatting above sections is covered in [comments.md](comments.md).