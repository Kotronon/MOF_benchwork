# Kurzer Trainingszweig ab Stage 1

Start aus dem Projektverzeichnis:

```bash
bash scripts/run_umof_stage1_followup.sh
```

Der Hauptlauf ist vorbereitet und noch nicht gestartet. Das bestehende
25-%-Training wird als Ausgangspunkt genutzt; die vollständige Matrix wird
nicht neu gerechnet.

| Einstellung | Wert |
|---|---|
| Ausgangsmodell | Ausgewählter Stage-1-Export, Adsorptions-Head `Default` |
| Neue Epochen | 5 |
| Lernrate | 0,00003, vorher 0,0001 |
| Seed | 12345 |
| Adsorptionsdaten | Dieselben 211 Trainings- und 46 Validierungsstrukturen |
| Replay | Dieselben 200 Trainings- und 100 Validierungsstrukturen |
| Stage 2 / SWA | Deaktiviert |
| GPU | 0; durch `CUDA_VISIBLE_DEVICES` wählbar |
| Auswertung | Validierung gegen den ursprünglichen Stage-1-Export |

Mit den bisherigen rund 36 Minuten pro Epoche ist grob mit **drei Stunden** zu
rechnen. Es werden anschließend keine Isothermen gestartet. Die Auswahl und
Bewertung verwendet keine Testdaten.

## Was fortgesetzt wird

Dies ist ein **neuer Trainingszweig mit den gelernten Stage-1-Gewichten**, kein
exaktes Fortsetzen des alten Optimizerzustands. Der Optimizer beginnt neu. MACE
initialisiert den Replay-Head aus dem ausgewählten Adsorptions-Head neu; der
ursprüngliche Replay-Head wird nicht als separater Head übernommen. Diese
Initialisierung unterscheidet sich von einem Optimizer-Resume und ist im Manifest
festgehalten. Innerhalb dieses neuen Zweigs werden spätere Unterbrechungen mit
Checkpoint und Optimizerzustand fortgesetzt.

Ein normales `resume` des früheren Laufs würde dagegen den letzten
Stage-2-Checkpoint laden und erneut unter Stage-2-Bedingungen weiterrechnen.
Die vorhandenen Modelle, Trainingsdaten und Isothermenergebnisse bleiben hier
unverändert; alle neuen Dateien liegen unter
`outputs/umof_polar_stage1_followup/`.

## Status und Ergebnis

```bash
bash scripts/run_umof_stage1_followup.sh status
```

Während des Trainings:

```bash
tail -f outputs/umof_polar_stage1_followup/training.log
```

Nach Abschluss enthält `validation_metrics.csv` den Vergleich von
`parent_stage1` und `followup`, jeweils insgesamt und nach MOF/Gasgruppe. Kleinere
Energie- und Kraftfehler sind ein positives Signal; unterschiedliche Entwicklungen
beider Metriken müssen getrennt beurteilt werden. Diese Validierung belegt noch
keine bessere Isotherme.

Das neue Modell liegt unter
`outputs/umof_polar_stage1_followup/models/umof_polar_adsorption_finetuned.model`.

## Technische Prüfung

Beim Head-Export in MACE 0.3.16 werden POLAR-Basispuffer mit `cl_scale` im Namen
fälschlich wie Head-abhängige Skalierungen behandelt. Der lokale Adapter erhält
diese gemeinsamen Puffer, lädt alle nichtleeren Parameter strikt und vergleicht
Energie und Kräfte mit dem ursprünglichen Modell auf einer Validierungsstruktur.
Das Ergebnis ist in `head_extraction_check.json` gespeichert. Die installierte
MACE-Bibliothek wird nicht verändert.

Ein separater Mini-Trainingslauf verwendet ausschließlich wenige kleine
Trainings-/Validierungsstrukturen und ist vom eigentlichen Folgeversuch getrennt.
