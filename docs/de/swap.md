# llama-swap: Export und Backend-Empfehlungen

`llamastery` ist die Wahrheitsquelle (validate + budget + Messungen gehen direkt an
`llama-server`); `llama-swap` ist nur ein Laufzeit-Proxy: ein Port, Hot-Swap, `ttl`.
Diese Seite behandelt den Export und die Frage, welcher Build dahinter laufen soll.

## Befehle

```bash
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml --dry-run  # erst prüfen
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml             # schreiben (.bak-<zeit> automatisch)
llamastery swap export --build faks --only qwen3.8-35B-A3B-miniplus-v2.1-128ctx     # Teilmenge nach stdout
llamastery swap up                           # als Daemon starten, Port wird automatisch gewählt
llamastery swap status                       # Proxy plus direkter Server
llamastery swap down                         # den Daemon beenden
llamastery swap install                      # v261-Binärdatei nach ~/.local/bin laden (nur Linux x64)
```

## Der Port wählt sich selbst

Voreingestellt ist `127.0.0.1:8087` (8080 belegt hier vom searxng-Container). Ist der Port
belegt, wird der nächste freie genommen, und die Ausgabe nennt den Verdrängten:

```
swap поднят на 127.0.0.1:8088 (pid 392842)
  порт 8087 занят (pid 392830; /usr/bin/python3.14) — выбран свободный
```

Der eigene swap wird nie verdrängt. Einen Port erzwingt man mit `--listen 127.0.0.1:8090`;
automatische Auswahl unterbindet `--no-auto-port`. Der Zustand (Port, PID, Konfiguration)
liegt in `~/.local/state/llamastery/swap.json`, damit kennt `swap status` ohne `--url`
die Adresse.

## Regeln

1. `models.ini` bearbeiten; das Swap-YAML entsteht ausschließlich durch `export`.
   Niemals von Hand bearbeiten.
2. `validate / budget / measure / probe / tune` gehen direkt (`:8099`/`:8098`),
   niemals über `:8080`. Messen durch den Proxy verzerrt die Zeiten.
3. Eine Swap-Konfiguration — ein Build. `faks` und `ik` niemals in einer Datei mischen:
   die Parameterschemata unterscheiden sich, und manche Parameter fallen stillschweigend weg.
4. Builds mit Router (`faks`, `upstream`) werden im Single-Modus über `preset_to_argv`
   exportiert. Eine "swap → Router → Instanz"-Matrjoschka ist verboten.

## Welcher Build hinter llama-swap laufen sollte (Empfehlung)

Referenzhardware: RTX 3060 12GB + Ryzen 5700X + 32GB RAM.

| Build | Wann | Swap-Status |
|---|---|---|
| `faks` | **Vorgabe.** Die `models.ini`-Presets sind für ihn geschrieben: `n-cpu-moe`, `fa`, `ctx-checkpoints`, `kv-unified`, env `GGML_CUDA_REGISTER_HOST=1` + `GGML_SCHED_PREFETCH_EXPERTS=1`. Sauberer Export, keine Warnungen. | ✅ primär |
| `upstream` | Reserve, wenn `faks` kaputtgeht oder das neueste `ggml-org`-Feature gebraucht wird. | ✅ Rückfallebene |
| `ik` | Höchste rohe MoE-Geschwindigkeit (IQK/Trellis, FlashMLA), aber ein altes Schema (synchronisiert August 2024): der Export verwirft `n-gpu-layers`, `load-mode`, `kv-unified`, `n-predict`. Für Aufzeichnungen direkt mit `llamastery load --build ik` laden; hinter swap nur als eigene Konfiguration und nach Prüfung der Warnungen. | ⚠️ getrennt |
| `xing4` | Nur für Modelle der `xing4`-Architektur. | ❌ nicht mischen |

Kurz gesagt: **swap nimmt `faks`** (`swap export --build faks`).
`ik` ist für manuelle Aufzeichnungen außerhalb von swap gedacht.

## Ausführen

```bash
~/.local/bin/llama-swap --config ~/.config/llama-swap/config.yaml --listen 0.0.0.0:8080
```

Installiert wird bewusst eine Binärdatei: CUDA auf dem Host läuft bereits, in docker gibt
es keine nvidia-Laufzeit, und das einheitliche Image verlöre die Faks-Patches und
erforderte ein Neuzuordnen der HF-Cache-Pfade.

## Aufräumen

`llamastery swap up` findet selbst einen freien Port und stört damit keinen laufenden
Dienst. Das war nicht selbstverständlich: `llama-swap` bringt sein eigenes `install` mit,
das den Dienst global installiert — und genau daran scheitern die Nachlade-Instanzen, weil
sie sich hinter einem Proxy verstecken, den es gar nicht gibt. Deshalb der eigene Export
statt `llama-swap --models-preset`: nur so bleibt sichtbar, welches argv am Ende wirklich
läuft.