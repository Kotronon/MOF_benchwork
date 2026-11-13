# MIL-53(Al)/CO₂: klassischer Clusterpilot ohne RASPA

Dieses Paket führt drei Vergleiche mit demselben Potential aus: starres Gerüst,
bewegliches Gerüst bei fester Zelle (NVT) und bewegliches Gerüst mit verformbarer
Zelle (NPT, `tri`). Es enthält 1216 Gerüstatome, 300 K, 1 bar CO₂ mit expliziter
Näherung f=p und 1 bar mechanischem Außendruck. Drei Seeds ergeben neun Fälle.
Je Fall: 2000 Schritte Vorlauf und 20 Blöcke à 1000 Schritte bei 0,25 fs,
insgesamt 5,5 ps. Die ersten vier Messblöcke werden verworfen.
**Das ist ein technischer Pilot, keine konvergierte Isotherme oder bestätigte
Breathing-Transition.** Die vollständige Pilotlänge wurde nicht lokal validiert.

## Auf das Cluster übertragen

Lokal im Projektordner:

```bash
rsync -avP reports/module_d_mil53_co2_pilot_v3.tar.gz reports/module_d_mil53_co2_pilot_v3.tar.gz.sha256 ge64zuq@10.152.112.111:/home/ge64zuq/MOF_benchwork/reports/
```

Auf dem Cluster:

```bash
cd /home/ge64zuq/MOF_benchwork/reports
sha256sum -c module_d_mil53_co2_pilot_v3.tar.gz.sha256
tar -xzf module_d_mil53_co2_pilot_v3.tar.gz
cd module_d_mil53_co2_pilot_v3
python3 run_mil53.py check --command lmp
```

Python ≥3.10 genügt; keine Python-Zusatzpakete nötig. `check` prüft Prüfsummen
und benötigte LAMMPS-Styles, startet keine Simulation. YAFF ist zwingend für
`mm3`, `cross` und `distharm`. Ein bestandener synthetischer Modul-D-Test
beweist nicht, dass diese Styles in deinem LAMMPS vorhanden sind.

Innerhalb einer CPU-Rechenallokation zunächst:

```bash
python3 run_mil53.py smoke --command lmp
```

Erst nach `PASS` den Pilot starten, in derselben entpackten Arbeitskopie:

```bash
python3 run_mil53.py pilot --command lmp
```

Ein expliziter Pfad funktioniert ebenfalls, z. B. `--command /home/ge64zuq/lammps-stable_22Jul2025_update4/src/lmp_serial`.
Für Smoke und Pilot muss die Befehlsliste identisch bleiben. Die lokal geprüfte
Ausführung verwendet einen CPU-Prozess. GPU-Beschleunigung und mehrere MPI-Ranks
sind für dieses Potential nicht geprüft. GPU-Verfügbarkeit ist keine Voraussetzung.
Die Befehle reichen keine Scheduler-Jobs ein; Account/Partition nach Clusterregeln
wählen. Logs und Teilresultate bleiben erhalten; ein fehlgeschlagener Lauf kann
nicht einfach erneut in denselben Ergebnisordner schreiben. Für einen neuen
Versuch das Archiv in ein neues übergeordnetes Verzeichnis entpacken, niemals
über bestehende Ergebnisse entpacken. Automatische Fortsetzung fehlt derzeit.

## Falls YAFF fehlt

Eigenes CPU-LAMMPS im Benutzerverzeichnis bauen; bestehende Installation bleibt
erhalten. Benötigt C++-Compiler, make, curl, tar; Netzwerk nur beim Download.
Die folgenden Befehle setzen voraus, dass das Zielverzeichnis noch nicht existiert:

```bash
cd "$HOME"
curl -L --fail -o lammps-stable_22Jul2025_update4.tar.gz https://codeload.github.com/lammps/lammps/tar.gz/refs/tags/stable_22Jul2025_update4
echo '411088d9c03339e025f6a975e0a5741bb9e3f351cc39eda220ab22ac318fe2fb  lammps-stable_22Jul2025_update4.tar.gz' | sha256sum -c -
tar -xzf lammps-stable_22Jul2025_update4.tar.gz
cd lammps-stable_22Jul2025_update4/src
make yes-molecule yes-kspace yes-rigid yes-mc yes-yaff yes-extra-molecule
make serial -j4
```

Bei abweichender Prüfsumme nicht fortfahren. Danach im Pilotordner `check`,
`smoke` und `pilot` mit dem vollständigen Pfad zu `lmp_serial` ausführen.

## Modellherkunft und Grenzen

Das Gerüstkraftfeld und die Ausgangsgeometrie stammen unverändert aus
[Neirynck, Geerinckx und Rogge (2026)](https://doi.org/10.1021/acsphyschemau.5c00141),
[Autorenrepository](https://github.com/SvenRogge/supporting-info), Commit
`12ac39727f750a2ea12b22d1de18b62b7fc57715`. Originale, Prüfsummen und Herkunft
liegen in `source_reference/`. Lizenz: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
auch für die abgeleiteten Parameterdateien; siehe `ATTRIBUTION.txt`.

CO₂ verwendet TraPPE-Parameter nach
[Potoff/Siepmann](https://doi.org/10.1002/aic.690470719).
**Die Gerüst–CO₂-Kreuzwechselwirkungen sind eine Projektadaption, kein
publiziertes, für MIL-53/CO₂ validiertes Gesamtpotential.** MM3-Gerüstradien
werden auf LJ-Minimumspositionen abgebildet und mit Lorentz–Berthelot gemischt;
die Abbildung wurde aus einer [MIL-53/Wasser-Arbeit](https://doi.org/10.1039/D0FD00025F)
übernommen und hier auf CO₂ angewendet. Gaußförmige Gerüstladungen erhalten
eine Korrektur gegenüber den Punktladungen des Gastes. Details und sämtliche
Kreuzparameter: `campaign/assets/validation` (JSON, ausdrücklich `not_validated`).

Die drei starren CO₂-Atome sind durch drei energiefreie Ausschlusskanten verbunden,
damit alle intramolekularen Paarenergien ausgeschlossen werden. Ihre Geometrie
erzwingt `rigid/nvt/small`. Der Host behält seine veröffentlichten Ausschlussregeln.
`reinit no` erhält die internen Körperdefinitionen zwischen Vorlauf und Messung.
Bei der nativen NPT-Zellskalierung weichen die ausgegebenen CO₂-Abstände im
lokalen 1000-Schritte-Test trotzdem maximal um 0,0001181 Å von der Vorlage ab;
in den beiden Zweigen mit fester Zelle liegt die Abweichung unter 3×10⁻¹⁴ Å.
Das ist keine strenge Geometrieerhaltung der ausgegebenen NPT-Koordinaten.
Zeitschrittabhängigkeit und Einfluss auf die MC-Energien sind vor wissenschaftlicher
Nutzung noch zu prüfen; das Paket nimmt keine nachträgliche Koordinatenkorrektur vor.
Tabellierte Potentiale haben hier keine analytische Tail-Korrektur; MC verwirft
Überlappungen unter 0,75 Å. Die publizierten Tabellen werden unverändert verwendet.
LAMMPS meldet darin Energie/Kraft-Konsistenzwarnungen, insbesondere in der auf
vier Dezimalstellen gerundeten Abstandsspalte von `eibonded.table`.
Diese Warnungen sind dokumentiert, aber nicht physikalisch abschließend bewertet.
Auch Warnungen zu Image-Flags und GCMC-Nachbarausschlüssen stehen in den Logs;
die originale periodisch verknüpfte Hosttopologie wurde nicht umgeschrieben.
Die Ausgangsstruktur ist die veröffentlichte trockene Struktur, ohne zusätzliche
Minimierung oder vorgezogene Equilibrierung. Ein Vergleich mit dem unveränderten
Quellinput ergibt dieselbe trockene Anfangsenergie und denselben Anfangsdruck;
dieser liegt bei etwa 38183 atm und ist noch nicht auf 1 bar equilibriert.

Vor belastbaren Adsorptions- oder Breathing-Aussagen fehlen insbesondere:
Validierung der Kreuzwechselwirkungen, Prüfung der Tabellen und Energieerhaltung,
Stabilität und freie Energie der Porenphasen, Zellgrößen- und Zeitschrittstudien,
ausreichende Equilibrierung, Austauschstatistik und wiederholte Übergänge.
Die Pilotlänge und ein verändertes Zellvolumen erfüllen diese Anforderungen nicht.

## Ergebnisse lesen

`campaign/results/status.json` protokolliert den Ablauf.
`campaign/evaluation/module_d_report.json` und `rigid_vs_flexible.csv` enthalten
die Auswertung. Pro Fall gibt es `console.log`, `measurements.txt`,
`mc_attempts.txt` (kumulative Versuche/Annahmen), `cell_history.txt`, Trajektorie,
`final.data` und Restart. Zelllängen in Å, Zellspannungen in atm (`units real`).
RMSD enthält bei NPT die affine Zellverformung; es ist kein reines Maß interner
Linkerbewegung. Stationarität kurzer Blöcke beweist keine Equilibrierung.
