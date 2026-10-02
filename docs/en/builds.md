# Builds, forks, and the flag schema

## Registry

Source directory, remote, router support (`--models-preset`), environment
variables, and port. Not every build has a router: `ik_llama` runs in single
mode, and its preset has to be translated into argv by hand.

```bash
llamastery builds list                        # what is registered, built, which version
llamastery builds detect                      # find forks in the usual directories (read only)
llamastery builds detect --apply              # write to the registry
llamastery builds show faks
llamastery builds add mine --path ~/git/mine-fork \
              --remote https://github.com/u/mine --no-router --port 8098
```

`--port` matters for a build without a router when 8099 is taken: without it
the second build dies with `couldn't bind to server socket`. A build's
environment variables are set in the same place and applied at start:

```bash
llamastery builds add faks --path ~/git/llama-faks \
              --env GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1
```

## Freshness

```bash
llamastery builds stale
llamastery builds stale --no-fetch            # no network access
```

It distinguishes three states that are easy to confuse:

| State | Meaning | Fixed by |
|---|---|---|
| behind upstream | commits exist that are not upstream | `git pull` |
| ahead | your own commits are not upstream | nothing: rebuilding keeps them |
| binary older than the tree | HEAD moved, the build was not redone | rebuild |

It also reports whether files carrying flags changed (`arg.cpp`,
`server-context.cpp`): in forks, new commits change the flag set, not just
speed, and it is better to learn that before measuring.

The same line appears in `doctor`, so the check is part of the routine.
Nothing is rebuilt automatically: the decision stays with the person.

After updating:

```bash
cd ~/git/llama-upstream && git pull --ff-only
cmake --build build -j$(nproc)
llamastery schema --build upstream --refresh   # refresh the flag cache
```

## Flag schema

The schema is parsed from that binary's `--help`, so fork-only flags
(`load-mode`, `image-min-tokens`, `ctx-checkpoints`, `spec-type`, `kv-unified`)
show up automatically. The cache is keyed on the binary's mtime, so a rebuild
invalidates the old schema on its own.

```bash
llamastery schema --build faks                # how many flags this build has
llamastery schema --build faks --grep moe     # everything matching a word
llamastery schema --build faks --json | jq '."--spec-type"'
llamastery schema --refresh                   # ignore the cache
```

If you need to know what a fork can do at all, start here.

The schema is used in three places: translating preset keys into argv,
`validate`, and the tuner. Thanks to that, knowledge about fork flags is not
duplicated in the code.

## Builds without a router

`ik_llama` has no `--models-preset`. `llamastery load --build ik` translates
the preset section into argv and restarts a single server:

```bash
llamastery load <preset> --build ik --dry-run   # see the argv without running
llamastery load <preset> --build ik
```

The translation follows that build's schema: flags it does not have are dropped
with a warning, and `gpu-layers` is recognised as the same key as
`n-gpu-layers`.

Such a build has its own addressing: the server address comes from the
registry and from `LLAMA_SERVER`, not from the default 8099. `probe` and
`measure` recover the build from the pid file — every CLI invocation is a
separate process.

## Process identification

A build is recognised by comparing `/proc/<pid>/exe` against `server_bin` from
the registry. So if you rebuild a build at a new path without fixing the
registry entry, `status` stops recognising whoever holds the port.

## xing4_0 port

The `xing4_0` architecture does not exist in stock llama.cpp — neither in any
local build nor upstream. The only working option is the `xing4_0-port` branch
of the `jmarceno/llama.cpp-xing4` fork (previously maintained by
`shuxiaoqiong`). In `detect` it is known as `xing4` (`~/git/llama-xing4`).

Three consequences to keep in mind:

1. **No fork flags.** This is almost pure upstream: no `n-cpu-moe`, no fork
   `load-mode`/`ctx-checkpoints`. Hybrid expert offload is done with stock
   `-ot`/`override-tensor` (`blk.(…).ffn_*_exps.weight=CPU`) — `budget`
   accounts for it exactly, from the GGUF tensor table.
2. **KV follows MLA.** `xing4_0` has compressed KV (`kv_lora_rank` + rope
   part, 576 elements/token/layer); the plain GQA formula overestimates it
   almost 2× — `budget` applies the MLA formula itself.
3. **Commits affect speed.** Verified: `b2056929` ("faster decode on
   quantised MLA KV") is slower than its parent `63c16fb` everywhere on an
   RTX 3060 (gen 30.7 vs 36.7 t/s @10k, prefill 178 vs 639 t/s, collapse at
   depth). Pinning the commit is deliberate; `builds stale` warns about new
   commits.