# MIL-53(Al)/CO₂: klassischer Clusterpilot ohne RASPA

Dieses Paket führt drei Vergleiche mit demselben Potential aus: starres Gerüst,
bewegliches Gerüst bei fester Zelle (NVT) und bewegliches Gerüst mit verformbarer
Zelle (NPT, `tri`). Es enthält 1216 Gerüstatome, 300 K, 1 bar CO₂ mit expliziter
Näherung f=p und 1 bar mechanischem Außendruck. Drei Seeds ergeben neun Fälle.
Je Fall: 2000 Schritte Vorlauf und 20 Blöcke à 1000 Schritte bei 0,25 fs,
insgesamt 5,5 ps. Die ersten vier Messblöcke werden verworfen.
**Das ist ein technischer Pilot, keine konvergierte Isotherme oder bestätigte
Breathing-Transition.** Die vollständige Pilotlänge wurde nicht lokal validiert.

## Stand nach dem Clusterpilot (14.11.2026)

Alle neun Läufe sind technisch vollständig, aber acht von neun bestehen den
Stationaritätstest noch nicht. Der lokale Ergebnisimport liegt unter
`reports/module_d_mil53_cluster_pilot_v3/`; maßgeblich ist
`evaluation/module_d_report.json` mit `status: complete` und
`validation_status: not_validated`.

| Variante | Mittlere CO₂-Beladung | Stationarität bestanden |
| --- | --- | --- |
| Starres Gerüst | 2,00 mol/kg | 1/3 Seeds |
| Flexibel, NVT | 2,03 mol/kg | 0/3 Seeds |
| Flexibel, NPT | 1,46 mol/kg | 0/3 Seeds |

Die Beladungen sind ungewichtete Mittel über die drei Seed-Mittelwerte nach
Verwerfen der ersten vier Blöcke. Es sind vorläufige Pilotwerte, keine
gesicherten Gleichgewichtswerte. Auch der eine bestandene Stationaritätstest
belegt für sich kein Gleichgewicht. Die Austauschstatistiken enthalten in jedem
Fall akzeptierte CO₂-Einfügungen und -Löschungen.

Im NPT-Zweig ist das mittlere Volumen der behaltenen Blöcke gegenüber der
Ausgangszelle um rund 3,3 % erhöht; zugleich verändert sich die Zellform weiter.
Das belegt noch keinen Breathing-Übergang. Die im Pilotbefund gemeldete kleine
Abweichung der CO₂-Geometrie bleibt ein offener numerischer Prüfpunkt; die unten
genannte maximale Abweichung von 0,0001181 Å stammt aus dem kürzeren lokalen
Test und ist kein neu bestimmtes Maximum des vollständigen Clusterpiloten.

Nach dem Pilot waren folgende Prüfungen vorgesehen (Ergebnisstand vom
15.11.2026 siehe unten):

1. **NPT-Zeitschrittvergleich:** 0,25 fs gegen 0,125 fs bei gleicher physikalischer
   Dauer, denselben Seeds, Ausgangsbedingungen und Dämpfungszeiten in ps.
   Beim kleineren Zeitschritt Vorlauf und Schritte pro Block verdoppeln
   (4000 bzw. 2000), ebenso das Austauschintervall (40 statt 20 Schritte).
   So bleiben 5,5 ps Gesamtdauer und fünf MC-Austauschversuche alle 5 fs gleich.
   CO₂-Bindungslängen und O–C–O-Winkel, Zelllängen, Tilt-Faktoren, Volumen,
   Temperatur, Drucktensor und Austauschstatistik vergleichen. Zeitreihen bei
   gleichem Zeitabstand ausgeben; Thermostat- und Barostatkopplung berücksichtigen,
   wenn Energieverläufe beurteilt werden.
2. **Trockene NPT-Kontrolle:** dieselbe Ausgangsstruktur ohne CO₂ und ohne
   GCMC bei denselben beiden Zeitschritten und derselben physikalischen Dauer
   prüfen. Damit lässt sich untersuchen, welcher Anteil der Zellrelaxation
   bereits im unbeladenen Gerüst auftritt.
3. **Danach gezielt verlängern:** nach Bewertung der Numerik längeren Vorlauf
   und längere Messung für alle drei Varianten planen; Stationarität von
   Beladung und Struktur sowie die Übereinstimmung unabhängiger Seeds erneut
   prüfen. Bisherige Ergebnisse und Pakete bleiben als Pilotstand erhalten.

### Ergebnis der numerischen Kontrollen (15.11.2026)

Für Seed 12345 sind CO₂/trocken × 0,25/0,125 fs jetzt technisch vollständig
abgeschlossen, jeweils mit 5,5 ps und 20 Messblöcken. Die CO₂-Ergebnisse liegen
in `reports/module_d_npt_checks_return_v1/`, die trockenen Ergebnisse in
`reports/module_d_npt_checks_return_v2/`; die CO₂-Ordner in v2 enthalten nur Inputs.
Die geprüften Input-/Asset-Hashes stimmen überein.

Über Messblöcke 5–20 beträgt die mittlere Volumenzunahme gegenüber der
Ausgangszelle trocken +3,487 % bzw. +3,401 % und mit CO₂ +3,289 % bzw. +3,226 %
(jeweils 0,25/0,125 fs). Deutliche Zellrelaxation tritt somit bereits ohne CO₂
auf. Alle vier Läufe verfehlen die RMSD-Stationaritätsgrenze; CO₂ bei 0,125 fs
zusätzlich die Beladungsgrenze. Gleichgewicht und Zeitschrittkonvergenz sind
damit weiterhin nicht belegt.

Der maximale CO₂-Paarabstandsfehler in den Endkonfigurationen liegt bei beiden
Zeitschritten bei rund 1,9×10⁻⁵ Å. In den jeweils 21 Dumpframes sind es rund
1,2×10⁻⁴ Å; diese Werte sind durch die gerundete Koordinatenausgabe begrenzt.
Die Geometrieprüfung benötigt deshalb höhere Ausgabepräzision und dichtere
Abtastung; beides kann direkt im nächsten längeren Erkundungslauf erfolgen.
Tabellenwarnungen und Energieerhaltung bleiben separate offene Prüfungen für
die numerische Validierung. Weitere Seeds werden anschließend für einen
belastbaren Vergleich benötigt.

Vollständige Auswertung, Quellen und Zeitreihen:
[NPT-Kontrollbericht v2](../reports/module_d_npt_review_v2_2026-11-15/README.md).

### Ergebnis der längeren Kontrollen (16.11.2026)

Beide Fälle in `reports/module_d_npt_relaxation_20ps_return_v1/` sind technisch
vollständig (20 ps, je 60 Messblöcke und 301 präzise Dumpframes). Alle
Input-/Asset-Hashes stimmen mit dem vorbereiteten Paket überein. Im Intervall
5–20 ps beträgt die mittlere Volumenzunahme mit CO₂ +3,362 % und trocken
+3,506 %. Die mittlere CO₂-Beladung ist 1,666 mol/kg; die drei 5-ps-Fenster
ergeben 1,551, 1,606 und 1,841 mol/kg. Das sind weiterhin keine bestätigten
Gleichgewichtswerte. Beide Fälle verfehlen den bisherigen RMSD-Fenstertest,
der CO₂-Fall zusätzlich die Beladungsgrenze; auch längere Vergleichsfenster
liefern keinen bestandenen Test.

Die präzisen CO₂-Dumps zeigen praktisch lineare Moleküle, aber kleine
Paarabstandsabweichungen bis 5,22×10⁻⁵ Å in den gespeicherten Frames. Eine
strenge Geometrieerhaltung und Zeitschrittkonvergenz sind damit nicht bestätigt.
Es handelt sich um LAMMPS-Ergebnisse; ein FLAMES-Modul-D-Lauf wurde noch nicht
durchgeführt. Zahlen, Einschränkungen und Zeitreihen stehen im
[20-ps-Kontrollbericht](../reports/module_d_npt_relaxation_review_2026-11-16/README.md).

### Vorbereitung der zwei Läufe mit je 20 ps

Das vorbereitete Paket
[`reports/module_d_npt_relaxation_20ps_v1/`](../reports/module_d_npt_relaxation_20ps_v1/README.md)
enthält trocken und CO₂ mit Seed 12345 und 0,25 fs: 5 ps Vorlauf und 15 ps
Beobachtung. Beide starten erneut aus der ursprünglichen Struktur. Die
Zeitschrittkonvergenz wird dadurch nicht bestätigt; 20 ps sind eine gewählte
Erkundungsdauer, keine garantierte Equilibrierungszeit. Präzisere Dumps
(`%.16g`) werden nach dem Vorlauf alle 0,05 ps geschrieben. Befehle für
Übertragung, Start und Ergebnisrücktransfer stehen in der Paket-README.

`prepare` unterstützt dafür `--timesteps-fs`, `--warmup-ps`,
`--measurement-ps`, `--dump-interval-ps` und `--precise-dump`.
Ohne diese Optionen bleiben die bisherigen vier 5,5-ps-Inputs unverändert.

### Numerische Kontrollen starten

`scripts/module_d_npt_checks.py` bereitet aus den abgeschlossenen v3-NPT-Inputs
vier unabhängige Läufe vor: CO₂/trocken × 0,25/0,125 fs. Standardmäßig wird
Seed 12345 als erster numerischer Vergleich verwendet. Jeder Lauf startet aus
der archivierten trockenen Ausgangsstruktur; CO₂ wird im jeweiligen CO₂-Zweig
per GCMC eingefügt. Die trockenen Zweige enthalten weder GCMC noch den
CO₂-Starrkörperintegrator. Alle vier Läufe umfassen 5,5 ps.

Lokal, im Repository:

```bash
rsync -avP scripts/module_d_npt_checks.py ge64zuq@10.152.112.111:/home/ge64zuq/MOF_benchwork/reports/
```

Auf dem Cluster, aus dem bisherigen entpackten Pilotordner:

```bash
cd /home/ge64zuq/MOF_benchwork/reports/module_d_mil53_co2_pilot_v3
python3 ../module_d_npt_checks.py prepare --pilot campaign --directory ../module_d_npt_checks_v1
```

In einer CPU-Rechenallokation mit dem bereits geprüften YAFF-LAMMPS starten:

```bash
python3 ../module_d_npt_checks.py run --directory ../module_d_npt_checks_v1 --command /home/ge64zuq/lammps-stable_22Jul2025_update4/src/lmp_serial
```

Das Skript startet die vier Läufe nacheinander und reicht keine Scheduler-Jobs
ein. `prepare` verweigert existierende Zielordner. `run` überspringt bereits
erfolgreiche Fälle und verweigert das Überschreiben unvollständiger Versuche.
Für alle drei Seeds bei der Vorbereitung `--seeds 12345 23456 34567` ergänzen
und einen neuen Zielordner verwenden; daraus entstehen zwölf Läufe.

Pro Fall liegen `console.log`, `cell_history.txt`, `measurements.txt`,
`trajectory.lammpstrj` und `final.data` vor; nur die CO₂-Zweige schreiben
`mc_attempts.txt`. `completed.json` bestätigt ausschließlich die technische
Ausführung. Das Skript bewertet weder Geometriefehler noch Stationarität
automatisch. Für den Vergleich entspricht die physikalische Zeit in fs dem
jeweiligen Schrittindex multipliziert mit 0,25 bzw. 0,125.

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
