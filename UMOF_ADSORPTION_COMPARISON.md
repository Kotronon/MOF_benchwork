# Startfertiger uMOF-Isothermenvergleich

## Längerer Vergleich nach dem Pilotlauf

Für den gezielten Vergleich von **Ausgangsmodell und Stage 1** starten:

```bash
bash scripts/run_umof_adsorption_refinement.sh
```

Dieser separate Lauf verwendet einen Seed (12345), die Druckpunkte **0,4655 und
0,9905 bar** sowie je **5.000 Einlauf- und 10.000 Messschritte**. Zwei Rechnungen
laufen parallel auf GPU 0 und 2; insgesamt sind es vier GCMC-Jobs mit **60.000
Schritten**. Widom wird nicht erneut gerechnet. Jeder Job startet mit leerem
Gerüst; die kurzen Pilotketten werden nicht verlängert.

Die bisherigen Jobs mit 3.000 Schritten brauchten jeweils etwa zehn Minuten.
Eine lineare Hochrechnung ergibt für diesen längeren Vergleich etwa **100
Minuten insgesamt** auf zwei GPUs. Mit veränderter Beladung und GPU-Auslastung
kann es länger dauern; dies ist eine Schätzung, kein Zeitlimit.

Status anzeigen:

```bash
bash scripts/run_umof_adsorption_refinement.sh status
```

Ergebnisse liegen separat unter `outputs/umof_polar_adsorption_refinement/`.
`report/comparison.png` vergleicht die längeren Isothermenpunkte mit dem Pilotlauf
und zeigt zusätzlich die Messblockmittelwerte. In `report/isotherm_summary.csv`
stehen die Mittelwerte der ersten und zweiten Messhälfte. Ähnliche Hälften und
stabile Blöcke sprechen für bessere Durchmischung, beweisen allein aber keine
Konvergenz. Ein Seed liefert keine Unsicherheit zwischen unabhängigen Läufen.
Die Auswertung vergleicht nur vollständig gerechnete gemeinsame Druckpunkte.

Die Einstellungen stehen in
`input_json_files/umof_polar_mgmof74_co2_refinement.json`. Bei erneutem Start werden
abgeschlossene Teiljobs übersprungen. Die ursprünglichen Einstellungen und
Ergebnisse bleiben unverändert.

## Ursprünglicher kurzer Vergleich

Aus dem Projektverzeichnis starten:

```bash
bash scripts/run_umof_adsorption_comparison.sh
```

Das Skript nutzt die vorhandene Python-Umgebung `MOF_sim` automatisch. Der
ursprüngliche kurze Vergleich ist abgeschlossen. Für die längere Prüfung ist
der separate Startbefehl im ersten Abschnitt vorgesehen.

## Umfang und Laufzeit

Verglichen werden uMOF-POLAR Ausgangsmodell, Stage 1 und Stage 2 für
**Mg-MOF-74 / CO₂ bei 298 K**, jeweils mit **einem Seed (12345)**.

| Rechnung | Pro Modell | Insgesamt über drei Modelle |
|---|---:|---:|
| Gemeinsame Widom-Einfügungen | 2.000 | 6.000 |
| GCMC: 4 Druckpunkte × 1.000 Einlaufschritte | 4.000 | 12.000 |
| GCMC: 4 Druckpunkte × 2.000 Messschritte | 8.000 | 24.000 |
| Summe | **14.000** | **42.000** |

Druckpunkte: **0,0175; 0,1015; 0,4655; 0,9905 bar**. Diese liegen exakt auf
Punkten der archivierten experimentellen Referenz; Interpolation ist dafür
nicht erforderlich.

Die drei Modelle rechnen parallel auf **GPU 0, 2 und 3**. GPU 1 wird für diesen
Lauf nicht verwendet. Zunächst laufen die drei Widom-Rechnungen, anschließend
die GCMC-Rechnungen in Gruppen von bis zu drei Jobs.

Im kleinen Probelauf benötigten 64 Widom-Einfügungen ungefähr 11 Sekunden pro
Modell und 80 GCMC-Schritte ungefähr 10 Sekunden pro Modell. Eine lineare
Hochrechnung ergibt rund **30 Minuten reine Simulationszeit**, hinzu kommen
Modellladen und Auswertung. **Etwa 30–60 Minuten** sind eine grobe Orientierung;
höhere Beladung und andere GPU-Auslastung können die Laufzeit verändern.

## Status, Ergebnisse und erneuter Start

```bash
bash scripts/run_umof_adsorption_comparison.sh status
```

Die Konsole meldet Start, Abschluss und Logpfad jedes Teiljobs. Die Dateien liegen
unter `outputs/umof_polar_adsorption_comparison/`:

- `status.json`: aktueller Zustand.
- `report/isotherm_comparison.png` und `.svg`: Vergleich mit der Referenzisotherme.
- `report/isotherm_summary.csv`: Beladung je Modell und Druck.
- `report/widom_seeds.csv`: Henry-Konstante, Adsorptionswärme und effektive
  Anzahl der Boltzmann-gewichteten Stichproben.
- `report/reference_metrics.csv`: Abweichungen von der experimentellen Referenz,
  sobald derselbe Druckpunkt für alle Modelle fertig ist.
- `report/comparison.json`: Blockmittelwerte, Drift zwischen erster und zweiter
  Messhälfte sowie Akzeptanzstatistiken.
- `jobs/…/attempt-…/worker.log`: Einzeljob-Logs.

Bei erneutem Start mit demselben Befehl werden abgeschlossene Teiljobs nach
Prüfsummenkontrolle übersprungen. Ein unterbrochener Teiljob beginnt mit demselben
Seed erneut; frühere Versuche bleiben erhalten. Die installierte Engine speichert
keinen vollständigen Zufallszustand für eine exakte Fortsetzung.

Für ein Terminal, das geschlossen werden soll, kann der Startbefehl in einer
eigenen `tmux`-Sitzung ausgeführt werden. Das Skript selbst startet im Vordergrund.

Andere GPUs können explizit gewählt werden:

```bash
bash scripts/run_umof_adsorption_comparison.sh run --gpus 0,2,3
```

Bei geänderten Recheneinstellungen ist ein neues Ausgabeverzeichnis erforderlich.
Die vorbereiteten Einstellungen stehen in
`input_json_files/umof_polar_mgmof74_co2_adsorption.json`.

## Wissenschaftliche Einordnung

Dies ist ein **kurzer Pilotvergleich**. Ein Seed und 2.000 Messschritte liefern
keinen belastbaren Konvergenznachweis. Zwischen-Seed-Fehlerbalken werden deshalb
nicht behauptet. Produktionsmittelwerte berücksichtigen auch abgelehnte
Monte-Carlo-Schritte; Blockmittelwerte helfen, fortbestehende Drift zu erkennen.

Alle Modelle verwenden dieselbe starre Gerüststruktur und CO₂-Geometrie mit
1,16 Å Bindungslänge. Die Struktur stammt aus dem veröffentlichten uMOF-Datensatz;
der eindeutig getrennte CH₄-Gast von Ausgangsframe 1 wurde entfernt. Eine
1×1×2-Superzelle mit 324 Gerüstatomen erfüllt die geprüfte lokale Cutoff-Bedingung.
Es erfolgt keine modellabhängige Gerüstrelaxation.

Jedes Modell verwendet seine eigene, einmal berechnete CO₂-Energie im Vakuum.
Diese Referenz bleibt für alle Positionen und Orientierungen gleich. Die
Widom-Geometrien und gültigen Versuchszahlen müssen per Prüfsumme zwischen den
Modellen übereinstimmen. Grobe Überlappungen werden durch eine gemeinsame harte
Grenze bei der Summe von 0,8-fach skalierten ASE-Kovalenzradien ausgeschlossen.
Das erhält den Zugang zu engen Mg–O-Kontakten, die der übliche vdW-Filter der
Engine ausschließen würde. Dieser Filter ist eine dokumentierte Näherung;
auffällige Energieausreißer oder sehr kleine effektive Stichprobenzahlen müssen
bei der Interpretation berücksichtigt werden.

Die experimentelle Referenz stammt aus dem lokalen NIST-ISODB-Spiegel:
[Bernini et al., Dalton Transactions (2015), Figure 4](https://doi.org/10.1039/C5DT03137K),
unfunktionalisiertes Mg-MOF-74, reines CO₂, 298 K. Das Original und seine
Prüfsumme sind archiviert. Die Quelle kennzeichnet die Beladung als mmol/g,
aber nicht eindeutig als absolut oder Überschuss. Die numerische
Übereinstimmung mit dieser einzelnen Referenz ist deshalb vorläufig zu bewerten;
es wird weder eine Skalierung an die Messung angepasst noch ein Energieoffset
aus den Testdaten abgezogen.

## Verifikation

Ein separater Probelauf mit allen drei echten Modellen, 64 Widom-Einfügungen und
16+64 GCMC-Schritten ist unter
`outputs/umof_polar_adsorption_comparison_smoke/` abgeschlossen. Die gemeinsamen
Widom-Geometrien stimmen überein. Die Zahlen dieses kurzen Funktionstests sind
keine Isothermenbewertung.

Zwölf gezielte Tests prüfen unter anderem Akzeptanzwahrscheinlichkeiten samt
Fugazität, die konstante Gasreferenz, die Statistik abgelehnter MC-Schritte,
Geometriegleichheit und die Prüfsummenkontrolle abgeschlossener Jobs.
