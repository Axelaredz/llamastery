# Standard für Preset-Kommentare

Gilt für `models.ini` (und jeden anderen Router-Preset). Das Ziel: jede Zahl in der
Datei ist entweder nachprüfbar oder ausdrücklich als Schätzung gekennzeichnet.

## Drei Ebenen

| Ebene | Markierung | Was sie beschreibt |
|---|---|---|
| Datei | `; ====` | Hardware, Build, globale Regeln, Modellfamilien |
| Familie | `; ====` | eine Gruppe verwandter Presets für ein Modell |
| Preset | `; ----` | ein bestimmtes Preset |

Verschiedene Markierungen sind keine Dekoration: `=` trennt eine Gruppe, `-` trennt
ein Element innerhalb einer Gruppe. Feste Breite: 75 Zeichen für `-`, 75 für `=`.

## Der Preset-Block

### Symbollegende

Das Werkzeug schreibt diese Feldbezeichnungen als **russische Wörter**, weil genau das
von `llamastery presets annotate` erzeugt wird und wonach der Parser sucht. Unten
stehen sie der Lesbarkeit halber als Emoji; **in den Dateien selbst stehen die
Wörter, nicht die Emojis.** Sie werden deshalb hier auch nicht übersetzt.

| Symbol | Wort in der Datei | Bedeutung |
|---|---|---|
| ✅ | `[замер]` | an einem echten Lauf gemessen |
| 🧮 | `[оценка]` | von `llamastery budget` berechnet, nicht gemessen |
| 🔬 | `Замер:` | Herkunft: die Bedingungen des Laufs |
| ⚠️ | `Нюанс:` | Fallstricke und Bedingungen der Verwendung |
| 💬 | `Заметка:` | alter Prosa wörtlich beibehalten |
| 🔥 | `ПРОГРЕТО` | dieser Lauf kam aus warmem KV, nicht im Prefill gewertet |
| 💥 | `ПАДАЕТ` | dieses Preset hat den Server heruntergefahren |

```
; ---------------------------------------------------------------------------
; [Tiel-Coder NanoPlus] 65k | Text | ngram-mod — SCHNELLSTER 65k-Preset
; tg 48.7 t/s  vram 9.9 GiB ✅  ctx 114688  ub 512
; 🔬 tiel-apply2 2026-09-29, ub=2048, browser ~800 MiB, needle 3/3
; ⚠️ ngram-mod funktioniert nur bei Text mit Wiederholungen (Code, Vorlagen);
;    bei einzigartigem Text entspricht die Geschwindigkeit dem Basis-Preset.
; ---------------------------------------------------------------------------
```

### Felder

* **Titel** — eine Zeile: `[Modell] ctx | Modus | was unterscheidet sich`. Formulierungen
  wie "SCHNELLSTES" oder "SICHER" bleiben erhalten: die Auswahl des Presets erfolgt
  nach Namen, nicht durch Lesen der Kommentare.
* **Metrikzeile** — immer eine Zeile, immer in dieser Reihenfolge:
  `tg <Wert> t/s  vram <Wert> GiB <Tag>  ctx <N>  ub <N>`
  * `<Tag>` ist ✅ für einen echten Lauf, 🧮 für eine `llamastery budget`-Berechnung.
    Er wird **einmal** am Ende geschrieben, wenn alle markierten Werte dieselbe Quelle
    haben; bei gemischten Quellen wird jeder Wert an Ort markiert, weil die Zeile
    sonst so gelesen würde, als wären beide gemessen.
  * Fehlt ein Wert, schreibt man "nicht gemessen". Ihn zu erfinden ist verboten: eine
    falsche Zahl in einem Kommentar ist schlimmer als keine, weil Leute sie zitieren.
  * `ctx` und `ub` erscheinen nur, wenn sie von benachbarten Presets derselben Familie
    abweichen — sonst wird die Zeile zu Rauschen.
  * `moe` (n-cpu-moe) ebenso: bei MoE-Modellen ist es der wichtigste Geschwindigkeitshebel.
* **🔬 Herkunft** — die Bedingungen, unter denen die `[замер]`-Zahl zustande kam:
  Tuner-Stufe mit Datum, was sonst noch lief, Probentiefe, `needle`. Wurde nichts
  gemessen, enthält diese Zeile den Befehl, der es erzeugen würde.
* **⚠️ Hinweis** — Fallstricke und Bedingungen der Verwendung. Darf mehrere Zeilen
  umfassen.
* **💬 Bemerkung** — alte Prosa **wörtlich** beibehalten: sie enthält Nuancen wie
  "kurz", "in voller Tiefe", "Prefill", die Metriken nicht enthalten. Ihre Zahlen
  duplizieren die Metriken bewusst — die Metrik ist ein Index zum Suchen, die Bemerkung
  ist die Wahrheit.

### Reihenfolge

Titel → Metriken → Herkunft → Hinweis → Bemerkung. Ein ausgelassener Block wird
einfach nicht gedruckt, leere Platzhalter bleiben nicht zurück.

Der `tg`-Wert in den Metriken ist die **Geschwindigkeit in voller Kontexttiefe**, wenn
die Quelle eine `deep`-Zeile hat. Ein `short (4k)`-Probewert gehört nicht ins
tg-Feld: eine 4k-Probe charakterisiert das Preset nicht. Gibt es kein `deep`, aber
doch ein `short`, wird `short` verwendet und das in der Herkunftszeile vermerkt.

## Was `llamastery presets annotate` tut

* behält die gesamte Prosa wörtlich, nur die Rahmung ändert sich;
* wandelt Überschriften im `=`-Stil in den `-`-Stil (alte Dateien mischen beides);
* stellt Metriken zuerst aus Tuner-Messungen zusammen, dann aus alter Prosa, und
  behandelt erst danach eine `llamastery budget`-Berechnung als Schätzung — dieselbe
  Priorität wie in der Methode selbst;
* überschreibt bereits erfasste Metriken nicht: Umformatieren darf keine Messungen
  verlieren;
* ist idempotent: ein zweiter Lauf meldet 0 Änderungen;
* verwirft nicht erkannte Zeilen nicht, sondern verschiebt sie nach 💬 und listet sie
  im Bericht auf.

Tabellenzeilen innerhalb eines Blocks sind vom Prosa-Parser ausgenommen. Eine
Vergleichstabelle hat mehrere Zahlen in Spalten, und die zweite Spalte (Prefill) wurde
früher für tg gehalten, ihre Anmerkung für eine Speichermessung.

## Was verboten ist

* Zahlen unterschiedlicher Verlässlichkeit in einem Block ohne Markierung mischen.
  "~10.6 GiB (gemessen)" neben "~10.2 GiB (sicher)" über Presets hinweg ist eine Quelle
  falscher Entscheidungen.
* Eine Schätzung als Tatsache schreiben. Die `llamastery budget`-Berechnung
  überschätzt den Compute-Buffer und unterschätzt die Wirkung von `n-cpu-moe` um
  etwa 13%.
* Metriken an zwei Stellen in einem Block duplizieren. Die Zahl steht in der
  Metrikzeile, die Bedingungen in der 🔬-Zeile.
* `cache-reuse` neben `mmproj` stehen lassen: der Server setzt es selbst zurück.

## Namenskonvention

`<modell>-<ctx>[k]-<modus>-<beschleuniger>-<vision>`

```
tiel-coder-nanoplus-65ctx-ngram-mmproj
│                │      │      │      └── mmproj: Vision
│                │      │      └───────── ngram: Beschleuniger
│                │      └──────────────── Textmodus (keine Vision)
│                └─────────────────────── 65k = 65536 Token
└───────────────────────────────────────── Modellfamilie und Variante
```

Modus-Suffixe: `mtp`, `ngram`, `dflash`. Vision-Suffix: `mmproj`. Hat ein Preset keinen
externen Draft-Kopf, darf das Wort `mtp` nicht in seinem Namen stehen — der Name muss
zum Inhalt passen.

## Aktuell halten

```bash
llamastery presets annotate --dry-run    # zeigen, was sich ändern würde
llamastery presets annotate --apply      # die Blöcke neu schreiben (mit Backup)
```

Die Formatierung ist idempotent: ein zweiter Lauf auf einer bereits formatierten Datei
ändert nichts. Nicht erkannte Zeilen werden mit einer Warnung nach ⚠️ verschoben,
nicht verworfen.