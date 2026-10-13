# Builds, Forks und das Parameter-Schema

## Registry

Quellverzeichnis, Remote, Router-Unterstützung (`--models-preset`), Umgebungsvariablen
und Port. Nicht jeder Build hat einen Router: `ik_llama` läuft im Single-Modus, und
sein Preset muss von Hand in argv übersetzt werden.

```bash
llamastery builds list                        # was registriert, gebaut, welche Version
llamastery builds detect                      # Forks in den üblichen Verzeichnissen finden (nur lesen)
llamastery builds detect --apply              # in die Registry schreiben
llamastery builds show faks
llamastery builds add mine --path ~/git/mine-fork \
              --remote https://github.com/u/mine --no-router --port 8098
```

`--port` ist bei einem Build ohne Router wichtig, wenn 8099 belegt ist: ohne ihn
stirbt der zweite Build mit `couldn't bind to server socket`. Die Umgebungsvariablen
eines Builds werden an derselben Stelle gesetzt und beim Start angewendet:

```bash
llamastery builds add faks --path ~/git/llama-faks \
              --env GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1
```

## Aktualität

```bash
llamastery builds stale
llamastery builds stale --no-fetch            # kein Netzzugriff
```

Es unterscheidet drei Zustände, die leicht zu verwechseln sind:

| Zustand | Bedeutung | Behoben durch |
|---|---|---|
| hinter Upstream | Commits, die es upstream nicht gibt | `git pull` |
| voraus | eigene Commits, die es upstream nicht gibt | nichts: ein Neubau behält sie |
| Binär älter als der Baum | HEAD ist weitergegangen, nicht neu gebaut | neu bauen |

Gemeldet wird außerdem, ob Dateien mit Parametern geändert wurden (`arg.cpp`,
`server-context.cpp`): In Forks verändern neue Commits nicht nur die Geschwindigkeit,
sondern den Parametersatz — besser erfährt man das vor dem Messen.

Dieselbe Zeile erscheint in `doctor`, die Prüfung gehört also zur Routine. Nichts wird
automatisch neu gebaut: die Entscheidung bleibt beim Menschen.

Nach dem Update:

```bash
cd ~/git/llama-upstream && git pull --ff-only
cmake --build build -j$(nproc)
llamastery schema --build upstream --refresh   # den Parameter-Cache erneuern
```

## Parameter-Schema

Das Schema wird aus dem `--help` genau dieses Binärs gelesen, dadurch erscheinen
Fork-eigene Parameter (`load-mode`, `image-min-tokens`, `ctx-checkpoints`,
`spec-type`, `kv-unified`) automatisch. Der Cache hängt an der mtime des Binärs,
ein Neubau entwertet das alte Schema also von selbst.

```bash
llamastery schema --build faks                # wie viele Parameter dieser Build hat
llamastery schema --build faks --grep moe     # alles, was zu einem Wort passt
llamastery schema --build faks --json | jq '."--spec-type"'
llamastery schema --refresh                   # den Cache ignorieren
```

Wer wissen will, was ein Fork überhaupt kann, fängt hier an.

Das Schema wird an drei Stellen benutzt: Übersetzung von Preset-Schlüsseln nach argv,
`validate` und der Tuner. So ist Wissen über Fork-Parameter nicht im Code dupliziert.

## Builds ohne Router

`ik_llama` hat kein `--models-preset`. `llamastery load --build ik` übersetzt den
Preset-Abschnitt in argv und startet einen Single-Server neu:

```bash
llamastery load <preset> --build ik --dry-run   # das argv ansehen, ohne zu starten
llamastery load <preset> --build ik
```

Die Übersetzung folgt dem Schema dieses Builds: Parameter, die er nicht hat, werden
mit einer Warnung verworfen, und `gpu-layers` wird als derselbe Schlüssel wie
`n-gpu-layers` erkannt.

Ein solcher Build hat eine eigene Adressierung: Die Serveradresse stammt aus der
Registry und aus `LLAMA_SERVER`, nicht aus dem Standardwert 8099. `probe` und
`measure` ermitteln den Build aus der PID-Datei — jeder CLI-Aufruf ist ein eigener
Prozess.

## Prozesserkennung

Ein Build wird erkannt, indem `/proc/<pid>/exe` mit `server_bin` aus der Registry
verglichen wird. Wird ein Build an neuem Pfad neu gebaut, ohne den Registry-Eintrag
anzupassen, erkennt `status` den Prozess auf dem Port nicht mehr.

## xing4_0-Port

Die `xing4_0`-Architektur gibt es im unveränderten llama.cpp nicht — weder lokal noch
upstream. Die einzige funktionierende Option ist der Branch `xing4_0-port` des Forks
`jmarceno/llama.cpp-xing4` (zuvor von `shuxiaoqiong` gepflegt). In `detect` heißt er
`xing4` (`~/git/llama-xing4`).

Drei Folgen sind zu bedenken:

1. **Keine Fork-Parameter.** Das ist fast reines Upstream: kein `n-cpu-moe`, keine
   Fork-Parameter `load-mode`/`ctx-checkpoints`. Der hybride Experten-Offload läuft
   über das unveränderte `-ot`/`override-tensor` (`blk.(…).ffn_*_exps.weight=CPU`) —
   `budget` rechnet das aus der GGUF-Tensorentabelle genau nach.
2. **KV folgt MLA.** `xing4_0` hat komprimiertes KV (`kv_lora_rank` + rope-Anteil,
   576 Elemente/Token/Layer); die einfache GQA-Formel überschätzt es um fast das
   Doppelte — `budget` wendet die MLA-Formel selbst an.
3. **Commits beeinflussen die Geschwindigkeit.** Geprüft: `b2056929` ("faster decode on
   quantised MLA KV") ist überall auf einer RTX 3060 langsamer als sein Elternteil
   `63c16fb` (generiert 30.7 vs 36.7 t/s @10k, Prefill 178 vs 639 t/s, Einbruch in der
   Tiefe). Der Commit ist bewusst festgenagelt; `builds stale` warnt vor neuen Commits.