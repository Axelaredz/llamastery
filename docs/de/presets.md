# Preset-Format in llama.cpp

Drei Ebenen der Konfiguration. Die Verwechslung zwischen ihnen ist die Ursache nahezu
jedes Missverständnisses, deshalb fängt man hier an.

| Ebene | Datei | Schalter | Was es ist |
|---|---|---|---|
| Presets je Modell | `models.ini` / `presets.ini` | `--models-preset PATH` | Abschnitt = Modell, `[*]` = gemeinsame Standardwerte |
| Benannt, teilbar | `preset.ini` in einem leeren HF-Repo | `-hf user/repo` | ein Preset, das wie ein "Modell" mit Tag behandelt wird |
| Gemeinsam für alle Binärdateien | `/etc/llama.cpp/config.ini`, `~/.config/llama.cpp/config.ini` | kein Schalter | gilt auch für `llama-cli`; gelesen werden nur `[*]` und der Abschnitt vor der ersten Überschrift |

Reihenfolge der Anwendung (schwächstes zuerst):
`config.ini` → Umgebungsvariablen → Modell-Preset → die eigene CLI des Routers.
Ein Argument aus der Kommandozeile **schlägt** also einen Wert aus einem Preset.

Die Wahrheitsquelle zu "wer kontrolliert was" ist `tools/server/server-models.cpp`,
Funktion `unset_reserved_args()`:

* aus jedem Preset entfernt: `ssl-key-file`, `ssl-cert-file`, `api-key`,
  `models-dir`, `models-max`, `models-preset`, `models-autoload`;
* vom Router beim Spawnen eines Kindprozesses überschrieben: `port`, `host`,
  `alias`;
* in einem Preset je Modell sind `model` / `mmproj` / `hf-repo` **das** Modell selbst
  und damit zulässig (der Router entfernt sie nur aus dem Basis-Preset).

## Drei Schreibweisen für einen Schlüssel

Äquivalent, in einer Datei kombinierbar:

```ini
ctx-size = 32768     # lang
c = 32768            # kurz
LLAMA_ARG_CTX_SIZE = 32768   # Name der Umgebungsvariable
```

Die Logik: erst werden die Formen zusammengefasst (kurz und lang zählen als ein
Argument), dann mit den Basis-Argumenten des Routers vereinigt.

Boolesche Werte: `on` / `off`, `true` / `false`, `1` / `0`, `enabled` /
`disabled`. Ein Parameter ohne Wert (`kv-unified`, `jinja`) — den Wert einfach leer lassen.

## Beispiel

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

## Wo der Router Modelle findet

1. `~/.cache/llama.cpp` (oder `LLAMA_CACHE`) — zwischengespeicherte HF-Modelle;
2. `--models-dir PATH` — nur direkte Kinder, **keine Rekursion**;
3. `--models-preset` — Abschnitte mit expliziten Pfaden (`m = ...`).

Bei Namenskollisionen gilt die Priorität: Preset > models-dir > Cache. Für
Mehrteil-Modelle und `mmproj` wandern die Dateien in ein Unterverzeichnis, und der
Projektorname sollte mit `mmproj` beginnen.

Verwandte Router-Schalter: `--models-max N` (voreingestellt 4, `0` ohne Grenze),
`--models-autoload` / `--no-models-autoload`.

## Was für den Speicher zählt

Jedes geladene Modell ist ein **eigener `llama-server`-Prozess** auf einem eigenen freien
Port auf `127.0.0.1`; der Router leitet lediglich Anfragen weiter. Daher:

* VRAM addiert sich über alle gleichzeitig geladenen Modelle, nicht je "Preset-Größe";
* `LLAMA_SERVER_ROUTER_PORT` wird an die Kindprozesse durchgereicht;
* `parallel` innerhalb eines Presets ist eine eigene Achse: es vervielfacht den
  KV-Pool innerhalb eines Prozesses.

`llamastery budget` beantwortet genau die Frage "passen N Presets gleichzeitig", die kein
bestehender GUI-Launcher beantwortet.

## Struktur des KV-Caches

```
bytes_per_token = n_attention_layers × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
cache_size      = bytes_per_token × ctx × slots
```

wobei `bytes(q8_0) = 1.0625`, `bytes(f16) = 2`, `bytes(q4_0) = 0.5625`.

Zwei Fallstricke:

* **GQA.** `n_kv_heads` ist meist weit kleiner als `n_head` (16 gegenüber 2 bei
  Qwen3-35B-A3B), KV ist also um ein Vielfaches kleiner als bei "naiven" Rechnungen.
* **Hybride Architekturen.** Bei qwen35moe / Nemotron-H / Jamba sind manche Schichten SSM
  (Mamba-artig), ihr Zustand wächst nicht mit dem Kontext und sie halten kein KV.
  Vollständige Aufmerksamkeit erscheint nur jede `full_attention_interval`-te Schicht.
  `llamastery budget` berücksichtigt das; "alle Schichten zählen" unterschätzt den
  Speicher um ein Vielfaches.

`kv-unified` (in den meisten modernen Presets aktiviert) bedeutet einen gemeinsamen
KV-Pool für den gesamten Kontext: `parallel` hört auf, den Cache zu vervielfachen. Schaltet
man es ab, kehrt die Vervielfachung mit der Slot-Anzahl zurück.

## Operationen an Presets

```bash
llamastery presets list                      # alle Abschnitte
llamastery presets show <section>             # die Schlüssel des Abschnitts
llamastery presets globals                   # der Abschnitt [*]
llamastery presets export -o - <section>...   # eine Teilmenge exportieren
llamastery presets annotate --dry-run        # was der Formatierer ändern würde
llamastery presets annotate --apply          # anwenden
```

### Fremde Presets importieren

Quellen: eine lokale Datei, eine URL oder `git-repo#branch:pfad/darin`.

```bash
llamastery presets import --source ./foreign-presets.ini --dry-run
llamastery presets import --source 'https://github.com/u/repo#main:presets.ini' --dry-run
llamastery presets import --source git@github.com:u/repo.git --only my-model-128k
llamastery presets import --source ./p.ini --on-conflict new      # nicht überschreiben
llamastery presets import --source ./p.ini --on-conflict overwrite --conflicts
```

Verhalten: `--dry-run` ist voreingestellt; bei Konflikten `skip`; vor dem Schreiben entsteht
eine `.bak-<zeitstempel>`. `--rename alt=neu` benennt Abschnitte um. Eine fehlende Zieldatei
wird angelegt.

### Validierung

```bash
llamastery validate                          # alle Abschnitte der aktuellen models.ini
llamastery validate <section> --json
llamastery validate --build ik               # gegen einen bestimmten Build prüfen
llamastery validate --no-paths                # Festplattenzugriff überspringen (schneller)
```

Sie fängt: unbekannte Schlüssel, Router-Steuerargumente (`api-key`, `models-max` werden
entfernt; `port`/`host`/`alias` werden überschrieben), nicht existierende GGUF-Dateien,
`c` jenseits des trainierten Kontexts, `n-cpu-moe` über der Schichtzahl, bekannte
Fork-Fallstricke und Presets, die im Absturzjournal stehen.

Jeder Build hat sein eigenes Parameter-Schema, ein für einen Fork geschriebenes Preset
warnt also auf ik_llama: manche Parameter existieren dort schlicht nicht. Der Hinweis nennt
den Parameter.

## Laden und Entladen

```bash
llamastery runtime start --build faks      # den Router starten
llamastery runtime status                  # Port, PID, Build, was im VRAM liegt
llamastery load <preset>                   # in den VRAM laden
llamastery runtime unload [model]          # entladen, der Server läuft weiter
llamastery runtime restart --build faks
llamastery runtime stop
llamastery runtime logs -n 100
```

`llamastery` führt den Server selbst aus — separate Manager (`llama`, `llama-faks`,
`llama-ik`) werden weder gebraucht noch benutzt. Das ist Absicht: Build-spezifische
Skripte gibt es für sehr wenige Forks, und das Werkzeug muss auch für jemanden
funktionieren, der sie nie installiert hat.

Vor dem Laden wird gegen das Schema **dieses** Builds geprüft und der VRAM prognostiziert:

```
preset: qwen3.8-35B-A3B-miniplus-21-128ctx-ngram-mmproj
build: faks   binary: /home/axel/git/llama-faks/build/bin/llama-server
  VRAM: 8.67 GiB of 12.00 GiB, headroom +2.96 GiB
```

Bei Fehlern wird nicht geladen (`--force` überschreibt das).

Andere Modelle werden zuerst entladen: es gibt einen einzigen VRAM-Pool. Wie viele Modelle
gleichzeitig geladen werden können, legt der Router mit `--models-max` fest.

Die Kommentarformatierung über den Abschnitten steht in [comments.md](comments.md).