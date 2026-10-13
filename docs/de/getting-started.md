# Installation und erste Schritte

## Voraussetzungen

* Python 3.11 oder neuer. Das ist alles: keine externen Abhängigkeiten, nichts
  wird global installiert.
* Ein gebauter `llama-server` für mindestens einen Build oder Fork. Dieses Tool
  baut nichts selbst — es registriert nur, was bereits gebaut wurde.

## Beschaffung

```bash
git clone <repo> ~/git/llamastery
```

Das Repository ist eigenständig: `bin/llamastery` läuft von überall. Die
Dokumentation unten bezeichnet es als `$LM`.

```bash
LM=~/git/llamastery
$LM/bin/llamastery --help
```

## Installation als Agent-Skill

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

Symlink statt Kopie, damit Änderungen für alle Agenten sofort sichtbar sind.
`SKILL.md` im Repo-Root ist die Kurzreferenz; alles Weitere steht in `docs/`.

## Erster Lauf

```bash
$LM/bin/llamastery doctor
```

`doctor` zeigt, was erkennbar ist: registrierte Builds, das aktuelle Preset, die
Anzahl der Messungen, die Compute-Buffer-Kalibrierung, den Gesamt-VRAM und ob die
Builds aktuell sind.

Falls die Registry leer ist:

```bash
$LM/bin/llamastery builds detect --apply
```

`detect` sucht in den üblichen Verzeichnissen (`~/git/*`, `~/llama*`) nach Forks.
Ohne `--apply` zeigt es nur, was gefunden wurde.

## Minimaler Arbeitszyklus

```bash
$LM/bin/llamastery validate                    # alle Abschnitte erfüllen das Schema
$LM/bin/llamastery budget                     # wie viel Speicher jedes Preset braucht
$LM/bin/llamastery runtime start --build faks  # den Router starten
$LM/bin/llamastery load <preset>               # Modell in den VRAM laden
$LM/bin/llamastery measure                     # tatsächlichen Speicher erfassen
$LM/bin/llamastery probe --tokens 110000        # Geschwindigkeit in realer Tiefe
$LM/bin/llamastery runtime stop                # anhalten
```

Die Reihenfolge ist wichtig: `measure` und `probe` ergeben nur mit geladenem Preset Sinn.

## Wo die Dinge liegen

| Was | Pfad |
|---|---|
| Build-Registry | `~/.config/llamastery/builds.json` |
| Messungen und Kalibrierung | `~/.local/state/llamastery/` |
| Absturzjournal | `~/.local/state/llamastery/crashes.json` |
| Server-Log | `~/.local/state/llamastery/router.log` |
| Parameter-Schema-Cache | `~/.cache/llamastery/` |
| Router-Presets | `~/.config/llama/models.ini` und Nachbarn |

Überschreibbar über `LLAMASTERY_CONFIG_DIR`, `LLAMASTERY_STATE_DIR` und
`LLAMASTERY_CACHE_DIR` (siehe README). Das Präfix des früheren Werkzeugnamens wird
absichtlich nicht unterstützt.

## Gegenprobe an eigenen Daten

Bevor man irgendeiner Zahl traut, sollte man sicherstellen, dass überhaupt Messungen
vorliegen:

```bash
$LM/bin/llamastery ingest            # was in tune-results und Kommentaren gefunden wurde
$LM/bin/llamastery budget --explain  # VRAM in Bestandteile zerlegt
```

Solange `calibrate` nicht gelaufen ist, ist die VRAM-Schätzung zu niedrig — das Tool
gibt eine Warnung aus. Das ist kein Fehler, sondern heißt, dass noch nicht kalibriert wurde.

## Sprache der Ausgabe

`llamastery` spricht Russisch (Quellsprache), Englisch, Deutsch und Chinesisch. Die
Sprache wird gewählt über `--lang`, über `LLAMASTERY_LANG` oder über `LANG`/`LC_ALL`,
in dieser Reihenfolge — der Schalter schlägt die Umgebung, weil er eine bewusste
Entscheidung ist.

```bash
llamastery lang                 # aktuelle und verfügbare Sprachen
llamastery lang list            # dieselbe Liste mit Markierung der aktiven
llamastery doctor --lang de     # einmalig auf Deutsch
LANG=de_DE.UTF-8 llamastery doctor   # dauerhaft für die Sitzung
```

Fehlt eine Übersetzung, erscheint an ihrer Stelle der russische Originaltext — ein
unvollständiger Übersetzer ist besser als eine Lücke mitten in der Ausgabe über VRAM.
Neue Übersetzungen pflegt man in `lib/locale/<code>.json`, die Schlüssel sind dabei
der russische Text selbst.