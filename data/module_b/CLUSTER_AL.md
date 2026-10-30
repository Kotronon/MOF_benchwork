# CO2/N2 active learning: MOF-5, ZIF-8, Cu-BTC

Drei getrennte `noise_aware`-Kampagnen in starren Frameworks. Die technische
Einsatzbereitschaft wird mit echten LAMMPS-Kurztests geprüft. NIST-Vergleich und
wissenschaftliche Hauptpaper-Reproduktion sind weiterhin offen. ZIF-8 ist hier
**kein NIST-Referenzmodell**. Flexible Frameworks gehören zu Module D.

## Start am Cluster

Das Repository **einschließlich `outputs/module_B_active_learning/`** übertragen.
Dieser Ordner ist von Git ausgeschlossen; ein Git-Clone allein enthält weder
vorbereitete Modelle noch Pilotdaten oder Prüfergebnisse. Seine relative Lage im
Repository beibehalten; der absolute Pfad zum Repository darf sich ändern.
Das vorbereitete Bundle enthält die verwendeten CIFs, Kraftfeld- und Gastdateien,
LAMMPS-Eingaben, Modellmanifeste, Kampagnenzustände und die archivierte Pilotevidenz.

Im Python-/LAMMPS-Environment müssen `lmp` (GCMC und PPPM), NumPy, SciPy, ASE,
Matplotlib und scikit-learn verfügbar sein; siehe `requirements.txt` und
`requirements-module-b-al.txt`. Der Helfer nutzt `python`, alternativ
`export PYTHON_BIN=/path/to/environment/bin/python`. `lmp` wird über `PATH` aufgelöst.

```bash
cd /path/to/MOF_benchwork
bash scripts/run_module_b_al.sh prepare all
bash scripts/run_module_b_al.sh status all

tmux new -s module-b-al
bash scripts/run_module_b_al.sh start all --max-requests 2
```

`prepare` prüft Abhängigkeiten und eingefrorene Modelle, erzeugt nur bei Bedarf
Kampagnen und Eingaben und startet keine Simulation. Ein vorhandener Zustand wird
nicht zurückgesetzt. `start` und `resume` bearbeiten dieselbe wiederaufnehmbare
Warteschlange. `all --max-requests 2` bedeutet **bis zu zwei Anforderungen je MOF**,
mit drei Seeds pro Anforderung; die MOFs werden nacheinander bearbeitet.

```bash
# Fortsetzen; ein unterbrochener Auftrag bleibt derselbe Auftrag:
bash scripts/run_module_b_al.sh resume all --max-requests 2

# Optional drei unabhängige Seed-Prozesse gleichzeitig (ausreichende Ressourcen nötig):
bash scripts/run_module_b_al.sh resume mof5 --max-requests 2 --jobs 3

# Separat gesperrte Prüfpunkte simulieren, nach Bedarf in kleineren Kontingenten:
bash scripts/run_module_b_al.sh holdout all --max-requests 16

# Aktueller Stand und englische Berichte/Grafiken:
bash scripts/run_module_b_al.sh status all
bash scripts/run_module_b_al.sh report all
```

Standard ist ein LAMMPS-Prozess; `--jobs` akzeptiert 1–3 parallele Seeds.
OpenMP-/BLAS-Threads sind auf eins begrenzt. Keine Slurm-Abhängigkeit. Pro
Kampagne ist nur ein schreibender Prozess erlaubt; Sperren werden auch nach einem
Prozessabbruch vom Betriebssystem freigegeben. Ein vorgemerkter Prüfauftrag muss
mit `holdout`, ein Trainingsauftrag mit `start`/`resume` fortgesetzt werden.

## Konfiguration und Budgets

| MOF | Struktur/Ladungen | T / K | Zelle | Atome |
|---|---|---:|---|---:|
| MOF-5 (IRMOF-1) | CRAFTED DDEC | 298.15 | konventionell | 424 |
| ZIF-8 | CRAFTED PACMOF | 300 | 2×2×2 Ausgangszelle | 2208 |
| Cu-BTC | RASPA-CIF, eingetragene Ladungen | 300 | konventionell | 624 |

UFF, vorhandene CRAFTED CO2/N2-Gastmodelle, Abschneideradius 12.8 Å, PPPM 1e-4.
Die gemeinsamen Module-A-Prüfungen bleiben aktiv. Bei Cu-BTC werden die sechs
asymmetrischen Ladungseinträge über die CIF-Symmetrie eindeutig expandiert.
Widersprüchliche Ladungen auf äquivalenten Positionen verhindern die Aufbereitung.

Versionskontrollierte Ausgangskonfigurationen:
`input_json_files/benchmark_module_b_al_{mof5,zif8,cubtc}.json`.
Die tatsächlich aufgelösten Konfigurationen und SHA-256-Manifeste liegen je
Kampagne in `production/<mof>/config.json` und `model_manifest.json`.
Sie wählen sämtliche archivierten Modelldateien explizit aus.

Je MOF: 4096 Sobol-Kandidaten (Seed 2026), logarithmischer Druck 0.1–10 bar,
zusätzliche reine Randzustände; zwölf reproduzierbare Maximin-Anfangszustände;
24 unterschiedliche Trainingszustände und 16 gesperrte Prüfpunkte. Seeds:
12345, 23456, 34567. Je Seed zunächst 5000 Equilibrierungs- und 50000
Produktionsschritte. Rohlogs, Dumps und Restarts werden gespeichert.

MOF-5 enthält bereits vier Punkte des längeren Piloten. Sie zählen zu den zwölf
Anfangs- und 24 Trainingszuständen. Der kürzere, überlappende Pilot wird nicht
gepoolt. Import ist inhaltlich geprüft, unterbrechbar und idempotent.

**Präzisionswiederholungen sind längere Läufe:** Ein unpräziser Trainingszustand
bekommt höchstens einen weiteren Auftrag mit 20000 Equilibrierungs- und 200000
Produktionsschritten je Seed. Dieser ersetzt die aktive Beobachtung; die frühere
bleibt in `history`. Eine Wiederholung verbraucht eine Anforderung, aber keinen
zusätzlichen Trainingszustand. Die ersten MOF-5-Aufträge können daher die bereits
importierten 0.1-/1-bar-Punkte verbessern. Prüfpunkte werden nicht automatisch
wiederholt und steuern weder Training noch Auswahl oder Abbruch.

## Unsicherheit und Interpretation

Die klassischen komponentenweisen t-Intervalle und gepaarten Fieller-Intervalle
bleiben unter `classic_seed_components` / `classic_seed_selectivity` erhalten.
Für die neuen Kampagnen werden zwischen den Seeds gemessene Varianz und
Autokorrelations-/Batch-Varianz aus den Rohlogs konservativ addiert. Das kann
Samplingrauschen doppelt berücksichtigen; es ist keine kalibrierte zusätzliche
Validierung. Fehlende Batch-Unsicherheit verhindert den Import ins GP.

Identische Seed-Mittelwerte bei schwankenden Rohlogs ergeben so weiterhin
positives GP-Rauschen. Konstante Zeitreihen bei positiver Fugazität bestehen das
Präzisionsziel nicht und erhalten eine dokumentierte Untergrenze von 0.05
Molekülen für den GP-Standardfehler. Reine Randzustände haben exakt null Beladung
der abwesenden Komponente und undefinierte Selektivität; dort wird nur die
vorhandene Komponente auf Präzision geprüft.

Das Präzisionsziel verlangt mindestens drei Seeds und eine relative
Intervallhalbbreite ≤20 % für beide vorhandenen Komponenten und die definierte
Selektivität. Es ist **kein Nachweis vollständiger Gleichgewichtskonvergenz**.
Für den importierten längeren Pilot erfüllen 5 und 10 bar das Ziel; 0.1 und 1 bar
verfehlen es. Status: **„Pilot ausgewertet; quantitative Referenzvalidierung offen“**.

`execution_budget_exhausted`, `budget_exhausted`, `simulation_precision_unmet`,
`surrogate_uncertainty_target_met` und `technical_error` bleiben unterschiedliche
Status. Fehler werden mit Meldung und Logverzeichnis in `campaign.json` gespeichert.
Ein erreichtes GP-Ziel bedeutet keine externe physikalische Validierung.

## Berichte und Techniktest

Unter `production/<mof>/report/` bzw. `smoke/<mof>/report/`:

- `observations.csv`, `uncertainties.csv`: komponentenweise Beladung, Gesamtwert,
  Selektivität, klassische und konservative Unsicherheiten; Massenherkunft im Zustand.
- `surrogate_predictions.csv`: Vorhersagen und latente GP-Intervalle.
- `holdout_predictions.csv`, `validation_report.json`: GP-Fehler gegenüber
  unabhängigen LAMMPS-Prüfpunkten; unpräzise Prüfpunkte ausdrücklich gekennzeichnet.
- Englisch beschriftete PNG und SVG für ausgewählte Zustände, Beladungen und
  Selektivität, Unsicherheit, Prüfpunkt-Parität und Lernkurven.
- `learning_curve.csv`: Auswertungsstände, soweit Prüflabels vorliegen. Vor deren
  Berechnung wird kein Genauigkeitswert erfunden.

`overview.csv` / `overview.json` fassen Trainings- und Prüfpunktzahlen,
Seed-Läufe einschließlich ersetzter Beobachtungen, Rechenzeit und offene
Präzisionsprüfungen zusammen. Zeiten sind summierte Prozesszeiten der Seed-Läufe,
keine Prognose für die Clusterlaufzeit.

```bash
# Optional den technischen Test am Cluster wiederholen/fortsetzen:
bash scripts/run_module_b_al.sh smoke all
```

Der isolierte Test umfasst je MOF `(P/bar,yCO2)=(0.1,0.1),(10,0.1),(1,0.9)`
als Training und `(5,0.5)` als Prüfpunkt, jeweils drei Seeds und 100+100 Schritte,
Thermoausgabe bei jedem Schritt. Keine Präzisionswiederholungen. Insgesamt 36
kurze Seed-Läufe, GP-Training, nächste Auswahl und Wiederaufnahme. Ein erneuter
Aufruf verwendet die abgeschlossenen Testdaten. Testdaten können nicht als
Produktionsbeobachtungen importiert werden. Die vorgeschlagene vierte
Trainingsanforderung bleibt als Wiederaufnahmetest vorgemerkt und wird vom
`smoke`-Helfer nicht ausgeführt.

Für eine vollständig neue technische Testserie einen neuen relativen Ordner
wählen, z. B. `--root outputs/module_B_al_cluster_check`.

Das mitgelieferte `outputs/module_B_active_learning_prepared.tar.gz` enthält den
vorbereiteten Ausgabeordner. Zusätzlich zum aktuellen Repository-Code auf den
Cluster übertragen und **im Repository-Stamm** entpacken:

```bash
tar -xzf /path/to/module_B_active_learning_prepared.tar.gz
bash scripts/run_module_b_al.sh prepare all
```

Das Paket ist ein Anfangszustand. Auf einem bereits laufenden Clusterprojekt
würde erneutes Entpacken die dortigen Kampagnenstände überschreiben; für solche
Kampagnen ausschließlich `resume` verwenden. Die Versionskontrolle allein
überträgt dieses von Git ausgeschlossene Paket nicht.

## Separate MOF-5 continuation: 0.1–20 bar

Activate `MOF_sim` before running the commands below. Preparation copies verified
completed observations and evidence; it does not launch production simulations.
The source campaign must have no active writer during preparation.

```bash
bash scripts/run_module_b_al.sh prepare mof5 \
  --root outputs/module_B_active_learning_mof5_20bar \
  --pressure-min 0.1 --pressure-max 20 \
  --import-from outputs/module_B_active_learning/production/mof5/campaign.json

bash scripts/run_module_b_al.sh resume mof5 \
  --root outputs/module_B_active_learning_mof5_20bar \
  --max-requests 2 --jobs 3

bash scripts/run_module_b_al.sh report mof5 \
  --root outputs/module_B_active_learning_mof5_20bar

# Compute up to two outstanding holdouts after finishing a pending training request:
bash scripts/run_module_b_al.sh holdout mof5 \
  --root outputs/module_B_active_learning_mof5_20bar \
  --max-requests 2 --jobs 3
```

Pressure options and `--import-from` are preparation-only. Subsequent commands
inherit the stored bounds. Repeating preparation with the same source path and
bounds preserves progress; different settings require another destination root.
The import is a snapshot, not an ongoing synchronization with the old campaign.

The new design retains the original candidate pool and twelve initial states,
adds 4,096 logarithmic-pressure Sobol candidates over 10–20 bar, and schedules
20 bar / 15% CO2 as the next unobserved initial state. Adaptive GP selection begins
only after the initial design is complete. Twenty-four distinct training states
(including imported states) and at most one precision retry per state are allowed.
`--max-requests` counts requests including retries, not necessarily new states.

The sixteen previous holdouts remain locked; four additional holdouts cover
12/18 bar at 15/50% CO2. Existing inspected holdouts are marked as such and must
not be presented as a fresh final test set. Reports separate the old domain,
the extension, and the full domain, explicitly marking missing holdout data.
The new learning curve starts with the extended domain; old reports and original
simulation files are retained byte-for-byte under `provenance/source`. Working evidence is copied separately under `evidence`, with internal paths and
summary checksums adjusted. The provenance archive retains original paths and
checksums. No claims of external physical
validation or advantage over random sampling follow from this extension alone.

## Parallel MOF-5 comparison and validation

Use `bash scripts/run_module_b_comparison.sh prepare` to prepare a separate random
control and independent holdout worker while the 20-bar AL campaign runs.
Detailed protocol, commands and limitations: `data/module_b/MOF5_AL_COMPARISON.md`.
The shared report never writes holdout labels into a live training campaign.
