# Install and first steps

## Requirements

* Python 3.11 or newer. That is all: no external dependencies, nothing
  installed globally.
* A built `llama-server` for at least one build or fork. This tool does not
  compile anything — it only registers what is already built.

## Getting it

```bash
git clone <repo> ~/git/llamastery
```

The repository is self-contained: `bin/llamastery` runs from anywhere. The
docs below refer to it as `$LM`.

```bash
LM=~/git/llamastery
$LM/bin/llamastery --help
```

## Installing as an agent skill

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

Use a symlink rather than a copy, so edits are visible to every agent at once.
`SKILL.md` in the repo root is the short cheat sheet; everything else is in
`docs/`.

## First run

```bash
$LM/bin/llamastery doctor
```

`doctor` shows what it can see: registered builds, the current preset, the
number of measurements, the compute buffer calibration, total VRAM, and
whether the builds are up to date.

If the registry is empty:

```bash
$LM/bin/llamastery builds detect --apply
```

`detect` looks for forks in the usual directories (`~/git/*`, `~/llama*`).
Without `--apply` it only shows what it found.

## Minimal working cycle

```bash
$LM/bin/llamastery validate                    # all sections pass the schema
$LM/bin/llamastery budget                     # how much memory each preset needs
$LM/bin/llamastery runtime start --build faks  # start the router
$LM/bin/llamastery load <preset>               # load the model into VRAM
$LM/bin/llamastery measure                     # capture real memory use
$LM/bin/llamastery probe --tokens 110000        # speed at real depth
$LM/bin/llamastery runtime stop                # stop
```

Order matters: `measure` and `probe` only make sense with a preset loaded.

## Where things live

| What | Path |
|---|---|
| build registry | `~/.config/llamastery/builds.json` |
| measurements and calibration | `~/.local/state/llamastery/` |
| crash journal | `~/.local/state/llamastery/crashes.json` |
| server log | `~/.local/state/llamastery/router.log` |
| flag schema cache | `~/.cache/llamastery/` |
| router presets | `~/.config/llama/models.ini` and neighbours |

Overridable through `LLAMASTERY_CONFIG_DIR`, `LLAMASTERY_STATE_DIR` and
`LLAMASTERY_CACHE_DIR` (see the README). The prefix from the tool's former
name is deliberately not supported.

## Sanity check on your own data

Before trusting any number, make sure measurements exist at all:

```bash
$LM/bin/llamastery ingest            # what was found in tune-results and comments
$LM/bin/llamastery budget --explain  # VRAM broken into components
```

Until `calibrate` has run, the VRAM estimate is too low — the tool prints a
warning about it. That is not a bug, it means calibration has not happened yet.