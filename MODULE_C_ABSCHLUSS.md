# Modul C Auswertung und Clusterrechnungen

Die vorhandenen Untersuchungen zu uMOF-MH, uMOF-POLAR und den weiteren Potentialklassen sind gemeinsam ausgewertet. Der [Abschlussbericht](reports/module_c_closure/REPORT.md) enthält Ergebnisse, Guideline-Abgleich, Referenzgrenzen und verbleibende Sampling-Fragen. Die quantitativen Isothermenrangfolgen bleiben bis zur Absicherung offen.

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

## Clusterpaket

Das [Transferarchiv](reports/module_c_closure/cluster_bundle.tar.gz) enthält die vier eingefrorenen Modellgewichte, Strukturen, Referenzen, Python-Code und die [Startanleitung](reports/module_c_closure/cluster_bundle/START_HERE.md). Es startet keine Rechnung automatisch. Für eine vollständige Ausführung sind 18 MH- und 24 POLAR-Jobs mit jeweils drei neuen Seeds vorgesehen, insgesamt 2,52 Millionen MC-Schritte.

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

```bash
MPLCONFIGDIR=/tmp/mof-c-mpl python -m analysis.module_c_closure
```

Das erzeugt Bericht, CSV/JSON und Abbildungen aus vorhandenen Daten; es trainiert und simuliert nichts. Quellenprüfsummen und die Prüfung identischer MH/POLAR-Geometrien und Referenzlabels stehen in `reports/module_c_closure/provenance.json`. Quellkampagnen wurden nicht verändert.
