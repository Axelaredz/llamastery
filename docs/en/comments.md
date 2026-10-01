# Preset comment standard

Applies to `models.ini` (and any other router preset). The goal: any number in
the file can either be checked, or is explicitly labelled as an estimate.

## Three levels

| Level | Marker | What it describes |
|---|---|---|
| File | `; ====` | hardware, build, global rules, model families |
| Family | `; ====` | a group of related presets for one model |
| Preset | `; ----` | one specific preset |

Different markers are not decoration: `=` separates a group, `-` separates an
element inside a group. Fixed width: 75 characters for `-`, 75 for `=`.

## The preset block

### Symbol legend

The tool writes these field labels as Russian words, because that is what
`llamastery presets annotate` emits and what the parser looks for. Below they
are written as emoji so this document stays readable in English. **The files
themselves contain the words, not the emoji.**

| Symbol | Word in the file | Meaning |
|---|---|---|
| ✅ | `[замер]` | measured on a real run |
| 🧮 | `[оценка]` | computed by `llamastery budget`, not measured |
| 🔬 | `Замер:` | provenance: the conditions of the run |
| ⚠️ | `Нюанс:` | traps and conditions of use |
| 💬 | `Заметка:` | old prose kept verbatim |
| 🔥 | `ПРОГРЕТО` | this run came from warm KV, excluded from prefill |
| 💥 | `ПАДАЕТ` | this preset has taken the server down |

```
; ---------------------------------------------------------------------------
; [Tiel-Coder NanoPlus] 65k | Text | ngram-mod — FASTEST 65k preset
; tg 48.7 t/s  vram 9.9 GiB ✅  ctx 114688  ub 512
; 🔬 tiel-apply2 2026-09-29, ub=2048, browser ~800 MiB, needle 3/3
; ⚠️ ngram-mod only works on repeating text (code, templates);
;    on unique text the speed equals the base preset.
; ---------------------------------------------------------------------------
```

### Fields

* **Title** — one line: `[Model] ctx | mode | what differs`. Phrasings like
  "FASTEST" or "SAFE" are kept: preset selection happens by name, not by
  reading comments.
* **Metrics line** — always one line, always in this order:
  `tg <value> t/s  vram <value> GiB <tag>  ctx <N>  ub <N>`
  * `<tag>` is ✅ for an actual run, 🧮 for a `llamastery budget` calculation.
    It is written **once**, at the end, when every marked value shares a
    source; with mixed sources each value is tagged in place, because
    otherwise the line reads as if both were measured.
  * If there is no value, write "not measured". Making one up is forbidden: a
    wrong number in a comment is worse than no number, because people cite it.
  * `ctx` and `ub` are printed only when they differ from neighbouring presets
    of the same family — otherwise the line turns into noise.
  * `moe` (n-cpu-moe) is printed likewise: it is the main speed lever on MoE
    models.
* **🔬 Provenance** — the conditions under which the `[measured]` number was
  obtained: tuner stage with date, what else was running, probe length,
  `needle`. If nothing was measured, this line holds the command that would
  produce it.
* **⚠️ Note** — traps and conditions of use. May span several lines.
* **💬 Remark** — old prose kept **verbatim**: it holds nuances like "short",
  "at full depth", "prefill" that metrics do not. Its numbers deliberately
  duplicate the metrics — the metric is an index for searching, the remark is
  the source of truth.

### Order

Title → metrics → Measured → Note → Remark. A skipped block is simply not
printed, no blank placeholders are left.

The `tg` value in metrics is the **speed at full context depth** if the source
has a `deep` line. A `short (4k)` probe value does not go into the tg field:
a 4k probe does not characterise the preset. If there is no `deep` but there is
a `short`, `short` is used and this is recorded in the "Measured" line.

## What `llamastery presets annotate` does

* keeps all prose verbatim, only the wrapper changes;
* converts `=`-style headings to `-` style (old files mix both);
* assembles metrics from tuner measurements first, then from old prose, and
  only then treats a `llamastery budget` calculation as an estimate — the same
  priority as in the method itself;
* does not overwrite already recorded metrics: reformatting must not lose
  measurements;
* idempotent: a second run reports 0 changes;
* does not drop unrecognised lines, but moves them into 💬 and lists them
  in the report.

Table rows inside a block are excluded from prose parsing. A comparison table
has several numbers in columns, and the second column (prefill) used to be
mistaken for tg, and its annotation for a memory measurement.

## What is forbidden

* Mixing numbers of different confidence in one block without tags.
  "~10.6 GiB (measured)" next to "~10.2 GiB (safe)" across presets is a source
  of wrong decisions.
* Writing an estimate as a fact. The `llamastery budget` calculation
  overestimates the compute buffer and underestimates the effect of
  `n-cpu-moe` by roughly 13%.
* Duplicating metrics in two places in a block. The number lives in the metrics
  line, the conditions in the 🔬 line.
* Leaving `cache-reuse` next to `mmproj`: the server zeroes it itself.

## Naming convention

`<model>-<ctx>[k]-<mode>-<accelerator>-<vision>`

```
tiel-coder-nanoplus-65ctx-ngram-mmproj
│                │      │      │      └── mmproj: vision
│                │      │      └───────── ngram: accelerator
│                │      └──────────────── text mode (no vision)
│                └─────────────────────── 65k = 65536 tokens
└───────────────────────────────────────── model family and variant
```

Mode suffixes: `mtp`, `ngram`, `dflash`. Vision suffix: `mmproj`. If a preset
has no external draft head, the word `mtp` must not appear in its name — the
name has to match the contents.

## Keeping it up to date

```bash
llamastery presets annotate --dry-run    # show what would change
llamastery presets annotate --apply      # rewrite the blocks (with a backup)
```

Formatting is idempotent: re-running on an already formatted file changes
nothing. Unrecognised lines are moved into ⚠️ with a warning rather than
dropped.