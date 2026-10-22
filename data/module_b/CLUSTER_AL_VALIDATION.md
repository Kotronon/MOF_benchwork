# Module B AL: technischer Abschlussbericht

Geprüft am 22.10.2026. Die drei CO2/N2-Kampagnen für MOF-5, ZIF-8 und Cu-BTC sind technisch startbereit. Externe quantitative Validierung und wissenschaftliche Paper-Reproduktion bleiben offen.

## Tests

- 156 unterschiedliche Regressionstests bestanden: 152 in der Hauptsuite; vier dort wegen des Standard-PATH übersprungene LAMMPS-Tests anschließend mit dem LAMMPS-Environment bestanden.
- Alle sechs Paper-Profile statisch geprüft; bestehende Reingas-, Gemisch-, Referenz-, t-/Fieller-, Material- und Wiederaufnahmetests erfolgreich.
- 36 echte AL-Seed-Läufe: je MOF drei Trainingszustände und ein gesperrter Prüfzustand, jeweils drei Seeds und 100+100 Schritte.
- Für jeden MOF: GP-Training, nächste adaptive Auswahl, Prüfbericht, identische vorgemerkte Anforderung bei Wiederaufnahme.
- Abgeschlossene Seeds wurden aus ihren realen Logs wiederverwendet; der Test verbietet dabei neue LAMMPS-Prozesse.
- In allen 36 gespeicherten Trajektorien bleiben Framework-Koordinaten und Zellgrenzen unverändert.
- Verzeichnistransfer ohne ursprünglichen CRAFTED-Baum, Prüfsummen, Schreibschutz, Budgets, Wiederholungsgrenze und getrennte Rollen geprüft.
- Keine AL-Produktionskampagne gestartet. MOF-5 enthält ausschließlich die vier importierten längeren Pilotpunkte; ZIF-8 und Cu-BTC haben noch keine Produktionslabels.

Ein alter Reingas-Kurztest übernahm zunächst 120000 Schritte aus benchmark.json. Dieser Versuch wurde abgebrochen; der Test ist nun ausdrücklich auf 100 Equilibrierungs- plus 100 Produktionsschritte begrenzt und hat bestanden. Die vollständigen Logs dokumentieren diesen Vorgang.

## Rechenaufwand der 36 AL-Kurzläufe

| Material | Seed-Läufe | Summe der Seed-Laufzeiten / s |
|---|---:|---:|
| mof5 | 12 | 273.00 |
| zif8 | 12 | 3268.93 |
| cubtc | 12 | 739.45 |

Diese Zeiten sind lokale Messwerte und keine Prognose für den Cluster. Die kurzen Testlabels dienen ausschließlich der Funktionsprüfung; sie erreichen überwiegend nicht das wissenschaftliche Präzisionsziel.

## Artefakte und Start

- [Clusteranleitung](CLUSTER_AL.md)
- `outputs/module_B_active_learning/validation/`: vollständige Testlogs und `summary.json`.
- `outputs/module_B_active_learning/{production,smoke}/<mof>/`: Zustände, Modelldateien, Manifeste, Eingaben und englische Berichte als PNG/SVG/CSV.
- `outputs/module_B_active_learning_prepared.tar.gz`: vorbereitete Ausgabeordner für den Transfer zusammen mit dem aktuellen Repository-Code.

```bash
bash scripts/run_module_b_al.sh prepare all
bash scripts/run_module_b_al.sh start all --max-requests 2
bash scripts/run_module_b_al.sh resume all --max-requests 2
bash scripts/run_module_b_al.sh holdout all --max-requests 16
bash scripts/run_module_b_al.sh report all
```

`--max-requests` gilt je MOF. Drei Seeds pro Anforderung; bei Präzisionswiederholungen vierfache Schrittzahl. Der längere MOF-5-Pilot bleibt eingeordnet als **„Pilot ausgewertet; quantitative Referenzvalidierung offen“**.
