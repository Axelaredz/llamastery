# VRAM-Budget, Messungen und Slot-Planung

## Was `budget` zeigt

```bash
llamastery budget                              # jedes Preset + Slot-Prognose für --models-max
llamastery budget <preset> --models-max 2 --reserve 1024 --explain
```

Es zerlegt den VRAM in Bestandteile: Gewichte (unter Berücksichtigung von `n-cpu-moe`),
KV-Cache, mmproj, Compute-Buffer. Es beantwortet außerdem eine eigene Frage — passen N
Presets gleichzeitig? Der Router hält jedes geladene Modell als eigenen Prozess, der
Speicher addiert sich also über alle hinweg.

```bash
llamastery calibrate <preset> --free-mib 1147
llamastery calibrate a=1147,b=992 --free-mib     # einzeln, je Abschnitt
llamastery calibrate --from-tune                # aus Tuner-Läufen
llamastery calibrate --from-log                 # aus dem Bericht des Servers selbst
```

Solange `calibrate` nicht gelaufen ist, ist die Schätzung zu niedrig — das Tool gibt
eine Warnung aus, und das ist kein Fehler. Eine große Streuung der Residuen heißt,
dass dem Strukturmodell nicht zu trauen ist und man sich auf Messungen verlassen sollte.

## Messung schlägt Prognose

Hat ein Preset eine echte `used_mib`, entscheidet dieser Wert über die Frage "passt es",
und die Schätzung steht daneben zum Abgleich. Ohne das sieht ein funktionierendes Preset
mit ungenauem Modell so aus, als passte es nicht: so erging es
`tiel-coder-nanoplus-128ctx-mmproj-moe16`, wo die Schätzung 11.54 GiB gegen tatsächlich
genutzte 10.42 GiB stand.

```bash
llamastery measure                    # die Karte auslesen und festhalten
llamastery measure --recalibrate      # zusätzlich den Compute-Buffer neu berechnen
```

Messen ergibt nur mit geladenem Preset Sinn.

`calibrate --from-log` ist genauer als `--from-tune`: es liest Zeilen mit
`CUDA0 compute buffer size` neben `n_ubatch` direkt aus dem Server-Log, den Puffer benennt
also der Server selbst. Der Haken: der Server gibt diese Aufschlüsselung nicht bei jedem
Laden aus (im Router-Modus erreicht der Instanzbericht das gemeinsame Log manchmal nicht),
man bekommt also wenige Datenpunkte. Die ältere Methode, den Rest vom Gesamtbetrag zu
nehmen, enthält konstante Überhänge und überschätzt den Puffer deshalb um etwa 0.3 GiB.

## Geschwindigkeit in realer Tiefe

Der Tuner spart mit flachen Kontexten, gearbeitet wird aber mit langen. Deshalb wird
die Tiefe getrennt gemessen, an einem laufenden Server.

```bash
llamastery probe --tokens 110000 --max-tokens 96 --repeats 2
llamastery probe --tokens 4096 --repeats 1          # kurz, für die Kurve
llamastery probe --from-file server.cpp --tokens 110000   # Prompt aus echter Datei
llamastery probe --image ~/photo.jpg                # Vision prüfen (mmproj)
llamastery probe ... --record --preset <section>   # in measurements.json speichern
```

Die Messung ist eine gewöhnliche Anfrage: der Server zählt die Zeiten selbst, daher
`prompt_n / predicted_n / *_ms`.

**Der Prompt-Cache ist ausdrücklich abgeschaltet:** die Anfrage trägt `cache_prompt: false`.
Ohne das kommt eine Wiederholung desselben Textes aus warmem KV mit `prompt_n = 4` statt
110000, und der Prefill dieses Laufs wäre um ein Vielfaches niedriger — auf der Grafik
sieht das wie ein plötzlicher Geschwindigkeitssprung aus. Solche Läufe werden mit 🔥
(warm) markiert und aus den Prefill-Statistiken ausgeschlossen. Ist jeder Lauf einer
Serie warm, sagt das Tool ausdrücklich, dass darin kein kalter Prefill vorkommt.

**Was gemessen wird, ist wichtiger als die Wiederholungszahl.** Eine "ohne Wiederholungen"-
Probe aus einem erfundenen Wortverzeichnis lässt das Modell teilweise in Schleifen
geraten, das Ergebnis hängt also davon ab, wie der Build Schleifen übersteht, und nicht
von der echten Geschwindigkeit. Für eine ehrliche Zahl `--from-file` mit echtem Code
oder echter Prosa verwenden; an der Schnittstelle wird ein Fortsetzungsanhang ergänzt,
sonst hält das Modell die Datei für beendet und liefert beim ersten Token EOS. Anhänge
werden nacheinander probiert, bis das Modell spricht, weil dieselben wortartigen Hilfen
nicht bei jedem Build wirken.

Gemessener Unterschied bei 114688 (Qwen3.8 und Tiel-Coder NanoPlus, faks):

| Prompt | ngram-mod | ohne ngram |
|---|---|---|
| Text mit Wiederholungen | 96–107 t/s | 26–28 t/s |
| echter Code bei 110k | 25.6 t/s | 28.8 t/s |

Der Beschleuniger bringt ×3–4 bei Text mit Wiederholungen und **schadet** bei einzigartigem
Text: er bezahlt für jeden Fehlschlag. Man schaltet ihn für Lasten mit Wiederholungen ein,
nicht pauschal.

## Absturzjournal

```bash
llamastery crashes                       # welche Presets den Server runterbringen, und warum
llamastery crashes --forget <preset>     # Eintrag löschen: geprüft und funktionsfähig
```

Ein Preset kann laden, die Schema-Prüfung bestehen und trotzdem erst im tiefen Kontext
umfallen — so bringt `ubatch 2048 + ngram-mod` CUDA bei 114688 zu Fall. Solche Abstürze
werden automatisch gemerkt (beim fehlgeschlagenen Laden und wenn eine Instanz während
`probe` stirbt), danach gilt:

- `validate` warnt vor einem Preset, das den Server runtergebracht hat;
- `presets annotate` schreibt eine 💥-Zeile (dieses Preset hat den Server runtergebracht)
  in den Preset-Block;
- eine erfolgreiche Messung löscht den Eintrag selbst — ein geprüftes Preset darf nicht
  weiter als abstürzend geführt werden.

Gespeichert in `~/.local/state/llamastery/crashes.json`.

## Messungen sammeln

```bash
llamastery ingest             # was in tune-results und models.ini-Kommentaren gefunden wurde
llamastery ingest --apply     # in ~/.local/state/llamastery/measurements.json speichern
```

Messungen aus `tune-results/*/results.json` (vom Tuner erzeugt) und aus Kommentaren
über `models.ini`-Abschnitten werden in einen Speicher zusammengeführt und in `budget` und
`validate` eingesetzt. Der Schlüssel ist die Konfigurationssignatur: ein Preset, das sich
in einem einzelnen Parameter unterscheidet, landet in einem eigenen Eintrag und erbt keine
Zahlen von jemand anderem.

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

## Wie die Rechnung funktioniert

```
bytes_per_token = n_attention_layers × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
cache_size      = bytes_per_token × ctx × slots
```

wobei `bytes(q8_0) = 1.0625`, `bytes(f16) = 2`, `bytes(q4_0) = 0.5625`.

Zwei Fallstricke:

* **GQA.** `n_kv_heads` ist meist weit kleiner als `n_head` (16 gegenüber 2 bei
  Qwen3-35B-A3B), KV ist also um ein Vielfaches kleiner, als naive Rechnung vermuten lässt.
* **Hybride Architekturen.** Bei qwen35moe / Nemotron-H / Jamba sind manche Schichten SSM
  (Mamba-artig); ihr Zustand wächst nicht mit dem Kontext und sie halten kein KV.
  Vollständige Aufmerksamkeit erscheint nur jede `full_attention_interval`-te Schicht.
  `budget` berücksichtigt das; "alle Schichten zählen" unterschätzt den Speicher um ein
  Vielfaches.

`kv-unified` (in den meisten modernen Presets aktiviert) bedeutet einen gemeinsamen
KV-Pool für den gesamten Kontext: `parallel` hört auf, den Cache zu vervielfachen. Schaltet
man es ab, kehrt die Vervielfachung mit der Slot-Anzahl zurück.