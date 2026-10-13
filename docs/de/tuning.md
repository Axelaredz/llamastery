# Tuning: Methode, Fallstricke und gesicherte Gesetzmäßigkeiten

Die Engine ist `tools/tune_models.py`. Sie arbeitet in zwei Stufen, und das ist der
Hauptgrund, sie überhaupt zu haben: ein billiger Screening-Durchlauf wirft den Müll
hinaus, ein teurer Validierungsdurchlauf wirft hinaus, was plausibel aussieht, aber nicht
haltbar ist.

## Zwei Stufen

1. **Screening** — `llama-bench` / `llama-sweep-bench` bei kurzem Kontext.
   Billig; entfernt offensichtlich schwache Kombinationen aus `n-cpu-moe` × `ubatch` × `t`.
2. **Validierung** — ein laufender `llama-server`: voller Kontext, echtes KV, Auslesen
   des freien VRAM, ein Retrieval-Test (Nadel im Heuhaufen) sowohl bei kurzem als auch
   bei tiefem Kontext und eine Prüfung der Draft-Qualität.

Artefakte jedes Laufs: `~/.config/llama/tune-results/<stufe>/<zeit-pid>/` —
`results.json` (maschinenlesbar), `screening.json`, `run-NNN.log`.

`results.json` ist das, was `llamastery ingest` in `measurements.json` sammelt. Lesenswerte
Felder:

| Feld | Bedeutung |
|---|---|
| `short.gen_tps` / `deep.gen_tps` | Generierungsgeschwindigkeit bei 4k und bei vollem Kontext |
| `short.prefill_tps` | Geschwindigkeit der Prompt-Verarbeitung |
| `min_observed_free_mib` | der niedrigste freie VRAM während des Laufs — die Schlüsselzahl |
| `needle_ok` | hat der Retrieval-Test bestanden (Modell bei langem Kontext nicht kaputt) |
| `ok` / `error` | hat der Lauf gehalten, mit Begründung |
| `config`, `config_extra` | der vollständige Parametersatz des Kandidaten |

Aus den Messungen wird der **schlechteste** Wert genommen, nicht der Durchschnitt: man
plant mit dem ungünstigsten Fall.

## Was der Tuner nicht tut

* Er sucht nicht über `c` — das ist eine Entscheidung des Menschen, keine Gittersuche.
* Er wählt nicht zwischen Modellen.
* Er prüft nicht, ob mehrere Presets gleichzeitig passen — dafür ist
  `llamastery budget` da.

## Fallstricke aus der Praxis

### `spec-type` mit mehreren Werten zusammen mit einem externen MTP-Draft

Im Faks-Build führt `--spec-type ngram-mod` zusammen mit `--model-draft` (einem
MTP-Kopf) beim Laden zu einem Segfault. Ein externer MTP-Kopf ist für sich genommen
schon doppelt so langsam (24 t/s gegenüber 48), deshalb ist `ngram-mod` ohne Draft der
einzige funktionierende Beschleuniger auf einem solchen Modell.

### `ubatch-size`

`ubatch-size = 2048` bei 12 GiB VRAM ist eine bekannte OOM-Ursache; 1024 funktioniert. Das
ist aber eine Heuristik, kein Gesetz: existiert für eine bestimmte Konfiguration eine
erfolgreiche Messung, schweigt `llamastery validate`.

Die gemessene Nuance: `ubatch-size = 2048` für sich genommen ist in Ordnung — gemessen
wurde es mit 26.4 t/s und 1616 MiB frei. Erst `2048` **zusammen** mit einem spekulativen
Decoder kippt, weil der Verifizierungsgraph einen größeren Compute-Buffer braucht als
zugeteilt wurde.

### `parallel`

`parallel > 1` vervielfacht den KV-Pool. Bei 12 GiB `parallel = 1` beibehalten.

### `mmproj` auf der CPU

`mmproj-offload = 0` (Projektor im RAM) wirkt kaum auf die Generierungsgeschwindigkeit —
gemessen: 38.9 gegenüber 38.8 t/s. Es nutzt keinen VRAM. Für Vision-Modelle auf einer
engen Karte ist das fast immer die beste Wahl.

### `cache-reuse` ist mit mmproj unvereinbar

In den Quellen geprüft, `tools/server/server-context.cpp:1220`:

```cpp
if (params_base.n_cache_reuse) {
    params_base.n_cache_reuse = 0;
    SRV_WRN("cache_reuse is not supported by multimodal, it will be disabled");
}
```

Beim Laden eines multimodalen Modells wird `cache-reuse` mit einer Warnung im Log
erzwungen abgeschaltet. Die zweite Schranke (gleiche Datei, ~3164): `can_cache_reuse`
verlangt `!slot.prompt.tokens.has_mtmd`, jede Anfrage mit Bild schaltet die Wiederverwendung
für diesen Slot also ab.

**Eine wichtige Korrektur an der Formulierung:** die Einschränkung gilt für mmproj, nicht
für `ngram-mod` und nicht für spekulatives Decodieren. Ein älterer Eintrag in models.ini
hatte das vermischt.

### `spec-draft-*` hat zusammen mit `ngram-mod` keine Wirkung

Geprüft: `common/speculative.cpp`, Funktion `common_speculative_n_max()` iteriert nur über
die **ausgewählten** Typen aus `spec->types`:

```cpp
case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:      // und EAGLE3, DFLASH
    n_max = max(n_max, spec->draft.n_max);   // Regler des Draft-Modells
    break;
case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
    n_max = max(n_max, spec->ngram_mod.n_max);  // eigene Regler
    break;
```

Mit `spec-type = ngram-mod` sind `spec-draft-n-max` und `spec-draft-p-min` also wirkungslos
— sie steuern ein anderes Draft-Modell. Die eigenen Regler von ngram-mod sind
`--spec-ngram-mod-n-max` (64), `--spec-ngram-mod-n-min` (48),
`--spec-ngram-mod-n-match` (24). Ein Preset mit `spec-draft-n-max = 1` und
`spec-type = ngram-mod` läuft still auf Vorgabewerten.

### Was ngram-mod tatsächlich bringt

Gemessener A/B auf llama-upstream 0.5.0-dev, Qwen3.8 MiniPlus, 114688 Token,
derselbe Prompt, einziger Unterschied die `spec-type`-Zeile:

| Probetext | mit ngram-mod | ohne ngram |
|---|---|---|
| lorem (mit Wiederholungen) | **84–92 t/s** | 26.3–26.5 t/s |
| ohne Wiederholungen | 30.0 t/s | 28.5–30.5 t/s |
| 4k, mit Wiederholungen | 98.6 t/s | — |
| 4k, ohne Wiederholungen | 49.5 t/s | — |

Fazit: bei Text mit Wiederholungen (Code, Vorlagen, Dialoge) liefert der Beschleuniger
×3.2; bei einzigartigem Text liefert er genau nichts und schadet nicht.
`spec-type = ngram-mod` lohnt sich also für Lasten mit Wiederholungen, nicht als
"immer sicherheitshalber".

Methodische Einschränkung: die Probe des Tuners (Nadel im Heuhaufen) und der Befehl
`probe` liefern für dasselbe Preset verschiedene Zahlen — 48 gegenüber 26.5 t/s. Das sind
verschiedene Aufgaben, keine Messabweichung: nur innerhalb einer Methode vergleichen. Deshalb
ist der obige A/B `probe` gegen `probe`.

Erweiterte Messung an zwei Modellen (Qwen3.8 und Tiel-Coder NanoPlus, faks):

| Prompt | mit ngram-mod | ohne ngram |
|---|---|---|
| Text mit Wiederholungen | 96–107 t/s | 26–28 t/s |
| echter Code bei 110k | 25.6 t/s | 28.8 t/s |

Der Beschleuniger **schadet** bei echtem Code: er bezahlt für jeden Fehlschlag und macht
die Messung zudem instabiler (Streuung 5.3–6.6 t/s zwischen Wiederholungen gegenüber
0.1–0.5 ohne ihn).

### Wie `--n-cpu-moe` funktioniert

Kein numerisches Feld in params, sondern Puffer-Überschreibungen je Schicht
(`common/arg.cpp:2607`): für jede Schicht `i < N` wird `llm_ffn_exps_block_regex(i)` mit
erzwungenem CPU-Puffer zu `tensor_buft_overrides` hinzugefügt. Deshalb findet eine Suche
im Code nach `n_cpu_moe` nichts — man muss nach `tensor_buft_overrides` / `ffn_exps` suchen.

Gemessene Wirkung auf freien VRAM (Tiel, c=114688, b=2048, Läufe mit `ok=True`):
moe24 → 2963 MiB, moe32 → 5028, moe40 → 7129. Die Steigung beträgt 258–263 MiB je Schicht.
Der analytische Anteil der Expertengewichte (~0.978) überschätzt die Wirkung um etwa 13 %,
daher der Koeffizient `offload_realization = 0.89`, den `llamastery calibrate --from-tune`
automatisch anpasst.

### Antispam

`repeat-last-n = 256` + `repeat-penalty = 1.02` ist eine funktionierende Kombination gegen
Schleifen. Bei kaputtem JSON oder Tool-Aufrufen auf 1.05 senken; darunter ist es nutzlos.

### Denkmodelle bei niedrigen Quants: Denk-Schleifen

Denkmodelle (auch Qwen-artige) auf niedrigen Quants (IQ4, IQ3, Q3) geraten häufig in ihrer
Begründung in Schleifen: Das gesamte Budget geht an `<think>`, die Antwort bleibt leer,
`finish_reason: length`. Eine funktionierende Kombination (gegen die Build-Schemata geprüft):

```ini
temp = 0.6
top-p = 0.95
top-k = 20
reasoning-format = deepseek
reasoning-budget = 2048
reasoning-budget-message = Wrap up and give the final answer.
dry-multiplier = 0.5
dry-allowed-length = 6
dry-penalty-last-n = 4096
```

Warum das wirkt: `reasoning-budget` begrenzt die Denklänge hart, und
`reasoning-budget-message` bringt das Modell zum Abschluss und zur Antwort;
`reasoning-format = deepseek` trennt die Begründung vom Endtext (Gedanken landen in
`message.reasoning_content`); DRY bestraft wiederholte Sequenzen. Das Sampling
`0.6 / 0.95 / 20` liegt nahe anQwens offiziellen Empfehlungen.

**Zuerst die tatsächlichen Parameter prüfen.** Der Parametersatz fürs Denken unterscheidet
sich je Build (live geprüft): `--reasoning-effort` gibt es in upstream/xing4, aber **nicht**
in faks/ik; `--reasoning-budget-conclusion` gibt es **nirgends** — ein Preset, das es
mitführt, ignoriert den Schlüssel stillschweigend. Also vor dem Parametertuning:

```bash
llamastery schema --build <build> --grep reasoning   # was wirklich existiert
llamastery validate <section> --build <build>         # Zusammenspiel der Kombination:
  # Budget ohne Nachricht, Nachricht ohne Budget, DRY ohne Multiplikator,
  # effort auf einem nicht unterstützten Build, ungültiges Format — alles wird gefangen
```

`validate` prüft gegen das Schema **dieses** Builds, nicht gegen eine fest verdrahtete Liste:
Nach einem Fork-Update folgt die Prüfung automatisch dem `--help`. Der Tuner übernimmt die
Kombination wörtlich (Passthrough) in die Messung: reasoning-budget verändert die
Generierungsdynamik, und eine Messung ohne es würde eine andere Konfiguration messen als die
ausgelieferte. Die Mess-Signatur deckt ebenfalls Denken/DRY/Sampling ab — Läufe mit
verschiedenen Kombinationen werden nie zusammengeführt.

Ebenfalls einen Versuch wert: `min-p = 0.0`, Wiederholungs-/Anwesenheitsstrafe.

## Das Gesetz der Geschwindigkeitsabnahme

Ein früherer `models.ini`-Kopf hatte `tg(D) = 1000 / (16.1 + 0.0025·D)`. Die Formel ist
falsch: bei D = 114688 liefert sie 3.3 t/s, während derselbe Kopf 32.5 t/s verzeichnete —
der Koeffizient ist etwa 20× zu klein.

Neu angepasst an vier Tiel-Coder-Messungen (128k, RTX 3060, kein ngram):

| D | gemessen |
|---|---|
| 9 864 | 48.6 t/s |
| 32 327 | 42.1 t/s |
| 63 932 | 36.1 t/s |
| 114 012 | 32.5 t/s |

ergibt `tg(D) = 1 / (0.01961 + 9.786e-8 · D)`:
8k → 49.0, 32k → 43.8, 65k → 38.4, 114688 → 32.4 t/s.

**Verlasse dich aber auf Messungen deines eigenen Modells, nicht auf die Formel.** Die
Qwen3.8-Familie misst 39–41.5 t/s bei 114688, mehr als die Tiel-Anpassung vorhersagt. Das
Gesetz taugt für Größenordnung und für den Vergleich zweier Konfigurationen desselben
Modells, nicht dafür, Zahlen zwischen Modellen zu übertragen.

Praktische Folge: **eine Verdopplung des Kontexts kostet etwa 1 t/s** in großen Tiefen.
Ein 128k-Preset für lange Sitzungen verliert also fast nichts an Geschwindigkeit.

## Messartefakte

* Prefill auf warmem Cache gemessen liefert zu hohe Zahlen. Ein echter kalter Prefill bei
  65k dauert etwa 50 Minuten (~20 t/s).
* Eine Draft-Annahme von 1.00 ist ein Artefakt eines wiederholenden Prompts; nicht
  glauben. Bei gemischtem Text liegt die Annahme bei 0.93.
* Die erste Anfrage einer Sitzung profitiert nicht vom Drafting (der Index ist leer).

## Grenzen des Tuners, die man kennen sollte

* **Die Fakt-Extraktions-Prüfung bringt Denkmodelle zum Scheitern.** Der Tuner lässt das
  Modell ein Geheimnis in einen langen Prompt einfügen und erwartet es innerhalb von 64
  Token (`--n-predict`) zurück. Modelle mit `<think>` verbrauchen das ganze Budget fürs
  Denken, die Antwort ist leer, `finish_reason: length` — und der Lauf wird verworfen,
  obwohl seine Geschwindigkeit in Ordnung war. Mit `--n-predict 256` beheben. "Das Modell
  ist schlecht" und "das Modell denkt" unterscheidet man an `gen_tps`, nicht an `ok`.
* **Die tiefe Stufe ist teuer.** Bei 114688 dauert ein kalter Prefill zehntausende
  Sekunden, `--deep --apply` bei 128k bedeutet also Stunden. Der praktische Ersatz: die
  Parameter mit dem Tuner bei kurzem Kontext wählen und die Geschwindigkeit in realer
  Tiefe getrennt messen — `llamastery probe --tokens 110000` auf einem laufenden Server.
  Das ist billiger und misst das, worauf es ankommt.
* **Die Menge erlaubter Schlüssel wächst mit dem Schema des Builds.** Der Tuner nimmt die
  Parameter aus dem `--help` des Ziel-Binärs, ein Preset mit Fork-Parametern
  (`load-mode`, `ctx-checkpoints`, `no-mmproj-offload`) besteht also, und dieses Wissen ist
  im Tuner-Code nicht dupliziert.
* **`--with-spec` behält den Beschleuniger in der Messung.** Voreingestellt wird
  `spec-type` aus den Tuner-Läufen entfernt: die Probe entsteht aus Text mit
  Wiederholungen, auf dem ngram künstlich gewinnt, und ein Draft-Modell im Speicher
  verzerrt die VRAM-Auslesung. `--with-spec` übergeben, um bewusst damit zu messen.

## Reihenfolge der Arbeit bei der Preset-Wahl

1. `llamastery schema --grep <teilzeichenkette>` — prüfen, ob der Parameter in diesem
   Build überhaupt existiert (Forks fügen eigene hinzu, und `--draft` / `--draft-min` sind
   in neueren Versionen als entfernt erklärt).
2. `c` (die Kapazität der Aufgabe) und das Modell festlegen.
3. `llamastery budget <section>` — Speicher schätzen; wenn es nicht reicht, `n-cpu-moe`
   senken (der stärkste Hebel bei MoE) oder `c`.
4. `llamastery tune ... --extra deep` — der Tuner fährt `n-cpu-moe`/`ubatch`/`t` durch.
5. `llamastery ingest --apply` — die Messungen festhalten.
6. `llamastery budget` über alle Abschnitte — prüfen, ob mehrere Presets unter
   `--models-max` gleichzeitig passen.