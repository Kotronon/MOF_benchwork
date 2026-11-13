# Modul C Auswertung und Clusterrechnungen

## Startbereiter GPU-Pilot vom 11. November 2026

Der [begrenzte Drei-Modell-Pilot](docs/umof_guarded_pilot.md) vergleicht MH-Foundation, POLAR-Foundation und POLAR-Stage-1 auf derselben MOF-5-Struktur bei 1 bar. Er verwendet eingefrorene Modellgewichte, gemeinsame float64-Präzision, eine explizite vdW-Ausschlussregel, feste Gasreferenzen und Akzeptanzberechnung im Logarithmus. Es ist eine neue Protokollstudie; die historischen Ergebnisse werden nicht überschrieben.

Das Paket [umof_mof5_pilot.tar.gz](reports/umof_mof5_pilot.tar.gz) enthält Code, Gewichte und Eingaben. `bash run_gpu.sh` führt in einer GPU-Allokation zunächst Energie- und Funktionstests aus und startet nur bei bestandenen Prüfungen maximal neun Pilotjobs. Der generische Campaign-Pfad der ursprünglichen Drei-Modell-JSON ist für diesen Vergleich blockiert. Lokal bestanden 36 Tests sowie ein echter MH-Energiescan und ein kurzer MH-GCMC-Lauf. Die GPU-Prüfungen stehen beim Clusterstart noch an.

Die vorhandenen Untersuchungen zu uMOF-MH, uMOF-POLAR und den weiteren Potentialklassen sind gemeinsam ausgewertet. Der [Abschlussbericht](reports/module_c_closure/REPORT.md) enthält Ergebnisse, Guideline-Abgleich, Referenzgrenzen und verbleibende Sampling-Fragen. Die quantitativen Isothermenrangfolgen bleiben bis zur Absicherung offen.

## Clusterrückläufe: aktualisiert am 10. November 2026

**Neue Diagnose:** In 16 von 18 MH-Endzuständen finden sich auffällig kurze intermolekulare Kontakte. Drei echte lokale Energiechecks des MH-Foundation-Modells reproduzieren die gespeicherten Energien, einschließlich eines extrem günstigen Nahkontaktzustands. Für MH ist daher zusätzlich zum Sampling das Potential-/Ausschlussprotokoll zu klären. Der [vorbereitete Clustercheck](reports/umof_cluster_diagnostic_2026-11-10/START_HERE.md) prüft zwölf ausgewählte Zustände mit den ursprünglichen Gewichten; weitere lange GCMC-Läufe sollten erst nach dieser Diagnose geplant werden.

Alle **18 MH- und 24 POLAR-Läufe** liegen vor und wurden aus den Übergangslogs nachgerechnet. Der [aktuelle Ergebnisbericht](reports/module_c_cluster_return_2026-11-10/REPORT.md) ergänzt die früheren Auswertungen um drei MC-Seeds je Modell und Druck, jeweils mit 20000 Equilibrierungs- und 40000 Produktionsschritten.

Nur **6 von 42 Ketten** und **0 von 14 Modell-Druckpunkten** bestehen die jeweils vollständigen Sampling-Prüfungen. Die Rechnungen fehlen damit nicht mehr; die Absicherung der Isothermen bleibt offen. POLAR vergleicht die Foundation mit **Stage 1**, nicht mit dem separaten Folgemodell. Alle vier eingefrorenen Clustergewichte bestehen die Prüfsummenprüfung.

- [Isothermen mit Seedintervallen](reports/module_c_cluster_return_2026-11-10/01_returned_isotherms.png)
- [Samplingdiagnostik](reports/module_c_cluster_return_2026-11-10/02_sampling_failures.png)
- [Referenzabweichungen](reports/module_c_cluster_return_2026-11-10/reference_metrics.csv)

Die Rückläufe werden getrennt von den historischen Piloten ausgewertet. Die folgende Auswertung vom 9. November hatte den Rücklauf-Unterordner noch nicht berücksichtigt; ihre Aussagen über ausstehende Clusterläufe und Ein-Seed-Isothermen sind für den aktuellen Bestand durch diesen Nachtrag ersetzt.

## Lokale Auswertung vom 9. November 2026

Der vorhandene Bestand wurde erneut geprüft und als Text für die Arbeit aufbereitet:

- [Ergebnisse und Diskussion](reports/module_c_local_review_2026-11-09/ERGEBNISSE_UND_DISKUSSION.md), mit sieben Abbildungen als PNG und SVG.
- [Methodik](reports/module_c_local_review_2026-11-09/METHODIK.md), einschließlich Datenaufteilung, Modellselektion, Fehlermaßen und Sampling-Grenzen.
- [Datenprüfung](reports/module_c_local_review_2026-11-09/DATENPRUEFUNG.md), mit 24 aus Einzelvorhersagen nachgerechneten Fehlerkennzahlen und dokumentierten veralteten MH-Statusübersichten.

Die Auswertung enthält keine neuen Clusterrechnungen. Der POLAR-Folgelauf verbessert die Validierungskräfte gegenüber Stage 1, erhöht jedoch die Isothermenabweichung im ZIF-8-Piloten. Unterschiedliche Systeme und begrenztes Sampling erlauben daraus keine allgemeine Korrelationsaussage. Quellkampagnen und der frühere Abschlussbericht bleiben unverändert.

Die lokale POLAR-Folgemodelldatei hat eine andere Prüfsumme als im archivierten Auswahlbericht. Die Aussagen zum Folgelauf beziehen sich deshalb auf die archivierten Kennzahlen; ihre Reproduktion mit der heutigen Datei bleibt ungeklärt. Die drei ursprünglichen Fine-Tuning-Checkpoints bestehen die Identitätsprüfung. Details und beide Prüfsummen stehen in der Datenprüfung.

Reproduzieren in einem neuen Ausgabeordner:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mof-c-review-mpl \
python -m analysis.module_c_local_review --as-of 2026-11-09 --output /tmp/module_c_review_reproduced
```

## Abbildungen

Alle Grafiken liegen als hochauflösende PNG und editierbare SVG in `reports/module_c_closure/` vor.

| Abbildung | Aussage |
|---|---|
| [Testfehler vor und nach Fine-Tuning](reports/module_c_closure/01_finetuning_test.png) | MH 10/25 % und POLAR 25 %, Energie und Kräfte |
| [Effekte je MOF und Gasgruppe](reports/module_c_closure/02_material_gas_effects.png) | Fehlerverhältnisse relativ zur jeweiligen Foundation |
| [Isothermen](reports/module_c_closure/03_isotherms.png) | MH/MOF-5 und POLAR/ZIF-8 mit klassischen Referenzen |
| [Henry-Konstanten und Adsorptionswärmen](reports/module_c_closure/04_widom_potential_classes.png) | Drei Potentialklassen bei 273 K mit Seedintervallen |
| [DFT-Test und Dispersion](reports/module_c_closure/05_golddac_dispersion.png) | GoldDAC-Energiefehler, Bias und Laufzeit |
| [Sampling-Grenzen](reports/module_c_closure/06_sampling_limits.png) | Autokorrelation und Änderungen zwischen Messhälften |

## Archiviertes Clusterpaket und Reproduktion

Das [Transferarchiv](reports/module_c_closure/cluster_bundle.tar.gz) enthält die vier eingefrorenen Modellgewichte, Strukturen, Referenzen, Python-Code und die [Startanleitung](reports/module_c_closure/cluster_bundle/START_HERE.md). Die 18 MH- und 24 POLAR-Jobs mit insgesamt 2,52 Millionen MC-Schritten sind abgeschlossen und lokal vorhanden. Die folgenden Befehle dokumentieren das ursprüngliche Ausführungsprotokoll; ein erneuter Start ist für die Auswertung nicht nötig.

Nach dem Entpacken auf dem Cluster im Verzeichnis `module_c_cluster`, mit aktiviertem kompatiblem Environment:

```bash
python -m modules.module_c_mlips.closure_campaign check --directory .
# In einer GPU-Allokation; zuerst die getrennten technischen Kurztests:
python -m modules.module_c_mlips.closure_campaign smoke --directory . --system mh
python -m modules.module_c_mlips.closure_campaign smoke --directory . --system polar
# Danach die begrenzten wissenschaftlichen Läufe:
python -m modules.module_c_mlips.closure_campaign run --directory . --system mh --max-jobs 18
python -m modules.module_c_mlips.closure_campaign run --directory . --system polar --max-jobs 24
python -m modules.module_c_mlips.closure_campaign report --directory . --system mh
python -m modules.module_c_mlips.closure_campaign report --directory . --system polar
```

Einzelne Array-Jobs sind über `--job-index` möglich; Näheres steht in der Startanleitung. Unterbrochene Jobs starten in neuen Versuchspfaden aus ihrem ursprünglichen Seed. Fertige, unveränderte Resultate werden wiederverwendet. Die Ausgabeordner `results/mh` und `results/polar` zusammen mit beiden Konfigurationen und dem Bundle-Manifest zurückübertragen.

Die neue Bestätigung verwendet in beiden Profilen denselben expliziten MC-Adapter und dieselbe Überlappungsnäherung. Die MH-Überlappungsregel unterscheidet sich vom früheren Standardadapter; historische Pilotdaten bleiben getrennt. Unterschiedliche MOFs erlauben weiterhin keine direkte MH/POLAR-Architekturrangfolge. Die Budgets und operationalen Präzisionsprüfungen garantieren kein Gleichgewicht.

## Lokal reproduzieren

Die zurückgekehrten Rechnungen ohne Inferenz oder Simulation in einen neuen Ordner auswerten:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mof-c-return-mpl \
python -m analysis.module_c_return_review --as-of 2026-11-10 --output /tmp/module_c_return_reproduced
```

Die ältere Zusammenfassung der Pilotdaten lässt sich separat erzeugen:

```bash
MPLCONFIGDIR=/tmp/mof-c-mpl python -m analysis.module_c_closure
```

Das erzeugt Bericht, CSV/JSON und Abbildungen aus vorhandenen Daten; es trainiert und simuliert nichts. Quellenprüfsummen und die Prüfung identischer MH/POLAR-Geometrien und Referenzlabels stehen in `reports/module_c_closure/provenance.json`. Quellkampagnen wurden nicht verändert.
