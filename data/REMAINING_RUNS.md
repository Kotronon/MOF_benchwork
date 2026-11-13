# Ein Aufruf für die verbleibenden Rechnungen

Im Repositorywurzelverzeichnis mit aktiviertem `MOF_sim`:

```bash
bash scripts/run_remaining.sh all --gpu 0
```

Der Aufruf führt die vorbereitete MOF-5-Modul-B-Kampagne und die eingefrorene
uMOF-Abschlusskampagne weiter. Er kann schon während der vorhandenen Aufrufe
gestartet werden: belegte Schreibsperren und die gewählte GPU werden abgewartet.
Er verändert keine Modellparameter, Punktbudgets oder Wiederholungsgrenzen.

Für eine reine Bestandsaufnahme ohne Simulationen:

```bash
bash scripts/run_remaining.sh all --gpu 0 --plan
```

Nur einen Bereich weiterführen:

```bash
bash scripts/run_remaining.sh module-b
bash scripts/run_remaining.sh umof --gpu 0
```

Modul B führt drei getrennte Zweige aus: AL, gemeinsame Holdouts und Zufallskontrolle.
Die Zufallskontrolle wartet automatisch auf das vollständige gemeinsame Startdesign
einschließlich vorgesehener Präzisionswiederholungen. Danach können alle drei
Zweige parallel rechnen: maximal neun CPU-Kerne mit dem Standard `--jobs 3`.
Mit `--jobs 1` sind es maximal drei parallele Seedprozesse. Die Stoppkriterien,
das Budget von 24 Trainingszuständen und die erlaubte Zahl von Wiederholungen
bleiben gültig. Abgeschlossene Trainingsstände erhalten Checkpoints; nach Ende
aller drei Zweige wird der Vergleichsbericht erneuert.

uMOF beendet zunächst POLAR und führt anschließend die fehlenden MH-Jobs aus.
Die Reihenfolge folgt den eingefrorenen Konfigurationen. Fertige Ergebnisse
werden mit den vorhandenen Identitäts- und Prüfsummenprüfungen wiederverwendet.
Ein unterbrochener uMOF-Job beginnt wie bisher in einem neuen Versuch mit dem
ursprünglichen Seed. Der eingefrorene Bundle-Code wird nicht verändert.
Nach jedem System wird dessen Bericht erzeugt.

`--gpu` bezeichnet den physischen Index aus `nvidia-smi`. Vor jedem neuen
uMOF-Job wartet der Starter auf eine GPU ohne Compute-Prozess und auf freie
Bundle-Schreibsperren. Das ist **keine systemweite GPU-Reservierung**: andere
Benutzer müssen sich weiterhin daran halten, die gewählte GPU währenddessen
nicht zu belegen. Andere GPUs werden nicht verwendet. Ein Rechenfehler wird
nicht automatisch wiederholt.

Logs und maschinenlesbare Zustände:

```bash
tail -f outputs/remaining_runs/al.log
tail -f outputs/remaining_runs/holdouts.log
tail -f outputs/remaining_runs/random.log
tail -f outputs/remaining_runs/umof.log
cat outputs/remaining_runs/*.json
```

Ein zweiter Sammelstarter für dieselben Zweige wird abgewiesen. Nach einer
Unterbrechung denselben Aufruf wiederholen. Ein gespeicherter technischer
Module-B-Fehler muss zuerst untersucht und über den bisherigen begrenzten
Resume-Befehl behoben werden. Wenn AL fehlschlägt, beendet der Starter auch die
wartende Zufallskontrolle; unabhängige Zweige dürfen fertiglaufen. Bei Strg-C
oder SIGTERM beendet er seine eigenen Prozessgruppen einschließlich ihrer
Simulationsprozesse. Bereits vor dem Sammelstarter laufende Prozesse gehören
nicht zu diesen Gruppen.

Für einen Lauf über mehrere Tage den Aufruf in einer bestehenden tmux-/screen-
Sitzung starten. Der Starter ist ein Vordergrundprozess und überlebt nicht
automatisch das Schließen seiner Sitzung.

Ein Ende ohne technischen Fehler bedeutet: Die konfigurierten Rechnungen und
Berichte sind abgearbeitet. Verbleibende Präzisions- und Samplingprobleme werden
berichtet; externe physikalische Validierung ist dadurch nicht abgeschlossen.
