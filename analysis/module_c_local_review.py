"""Re-audit local Module C predictions and write manuscript sections without inference.

python -m analysis.module_c_local_review --as-of 2026-11-09
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path

import numpy as np

from analysis.module_c_closure import (
    ROOT, MH, POLAR, Sources, collect, digest, dump, error_metrics,
    plot_all, table, write_csv,
)

RUNS = (("MH 10 %", MH, "010pct"), ("MH 25 %", MH, "025pct"),
        ("POLAR 25 %", POLAR, "025pct"))
# POLAR float32 scalars were serialized to CSV with fewer significant digits.
METRIC_RTOL = 1e-7
METRIC_ATOL = 1e-12


def audit_predictions(sources, data):
    """Recompute all-group errors from stored predictions and verify model/data identity."""
    from ase.io import read

    followup_root = 'outputs/umof_polar_followup_evaluation'
    followup = sources.json(followup_root + '/selected_model.json')
    if Path(followup['model']).stem != 'followup_epoch_1':
        raise ValueError('POLAR follow-up checkpoint differs from the model discussed here')
    followup_actual_sha = digest(sources.path(followup_root + '/' + followup['model']))
    followup_identity = {'recorded_sha256': followup['sha256'], 'local_sha256': followup_actual_sha,
                         'matches': followup_actual_sha == followup['sha256'],
                         'scope': 'Archived metrics only; local checkpoint equivalence is not established if hashes differ'}
    checks, statuses, splits = [], [], []
    for label, root, fraction in RUNS:
        prefix = f"{root}/{fraction}/seed-12345"
        evaluation = sources.json(prefix + "/results/evaluation.json")
        test_path = sources.path(prefix + "/datasets/test.extxyz")
        if digest(test_path) != evaluation['test_sha256']:
            raise ValueError(f"Test dataset identity changed for {label}")
        selected = evaluation['selected_model']
        model_path = sources.path(f"{root}/{selected['path']}")
        if digest(model_path) != selected['sha256']:
            raise ValueError(f"Selected checkpoint identity changed for {label}")
        selection = sources.json(prefix + "/results/selection.json")
        if selected['sha256'] != selection['selected']['sha256']:
            raise ValueError(f"Selected model and evaluated model differ for {label}")
        frame_sets = {}
        frames = None
        for split in ('train', 'valid', 'test'):
            current = read(sources.path(prefix + f'/datasets/{split}.extxyz'), index=':')
            frame_sets[split] = {a.info['umof_frame_id'] for a in current}
            if split == 'test':
                frames = current
        overlaps = {f'{a}_{b}': len(frame_sets[a] & frame_sets[b])
                    for a, b in (('train', 'valid'), ('train', 'test'), ('valid', 'test'))}
        if any(overlaps.values()):
            raise ValueError(f"Split overlap in frame IDs for {label}: {overlaps}")
        splits.append({'run': label, **overlaps, 'test_entries': len(frames),
                       'unique_test_frame_ids': len(frame_sets['test'])})
        e = sources.csv(prefix + '/results/energy_parity.csv')
        f = sources.csv(prefix + '/results/force_parity.csv')
        if [r['frame_id'] for r in e] != [a.info['umof_frame_id'] for a in frames]:
            raise ValueError(f"Energy prediction row order does not match test frames for {label}")
        expected_e = [a.info['REF_energy'] / len(a) for a in frames]
        expected_f = np.concatenate([a.arrays['REF_forces'] for a in frames]).ravel()
        for rows, expected, unit in ((e, expected_e, 'eV_per_atom'), (f, expected_f, 'eV_per_A')):
            if not np.allclose([float(r[f'reference_{unit}']) for r in rows], expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Parity reference labels differ from the test data for {label}")
        for model in ('baseline', 'finetuned'):
            recorded = evaluation['metrics'][model]['all']
            for rows, quantity, unit in ((e, 'energy', 'eV_per_atom'), (f, 'force', 'eV_per_A')):
                recomputed = error_metrics([float(r[f'{model}_{unit}']) for r in rows],
                                           [float(r[f'reference_{unit}']) for r in rows])
                for metric in ('mae', 'rmse'):
                    key = f'{quantity}_{metric}_{unit.replace("eV", "ev")}'
                    if not math.isclose(recorded[key], recomputed[metric], rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL):
                        raise ValueError(f"Stored and recomputed {key} differ for {label}/{model}")
                    consolidated = next(r for r in data['fine_tuning'] if r['run'] == label and
                                        r['model'] == model and r['group'] == 'all')
                    if not math.isclose(consolidated[key.replace('_ev_', '_meV_')], recomputed[metric] * 1000,
                                        rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL * 1000):
                        raise ValueError('JSON and CSV metrics disagree')
                    checks.append({'run': label, 'model': model, 'metric': key,
                                   'observations': len(rows), 'recomputed': recomputed[metric],
                                   'recorded': recorded[key], 'absolute_difference': abs(recorded[key] - recomputed[metric]),
                                   'passed': True})
        summary = sources.csv(root + '/matrix_summary.csv')
        row = next(r for r in summary if round(float(r['fraction']) * 100) == int(fraction[:3]) and int(r['seed']) == 12345)
        actual = sources.json(prefix + '/status.json')['status']
        statuses.append({'run': label, 'matrix_status': row['status'], 'run_status': actual,
                         'consistent': row['status'] == actual,
                         'resolution': 'use verified per-run results; source files unchanged'})
    return {'prediction_checks': checks, 'status_checks': statuses, 'split_checks': splits,
            'followup_selected_model': followup,
            'followup_checkpoint_identity': followup_identity,
            'metric_tolerance': {'relative': METRIC_RTOL, 'absolute_eV_units': METRIC_ATOL,
                                 'reason': 'Float32 prediction CSV serialization precision'},
            'scope': 'Existing predictions re-evaluated; no model inference. Frame-ID disjointness does not establish independence from foundation pretraining.'}


def paired_metrics(data):
    rows = []
    for label, _, _ in RUNS:
        pair = {r['model']: r for r in data['fine_tuning'] if r['run'] == label and r['group'] == 'all'}
        for key in ('energy_rmse_meV_per_atom', 'force_rmse_meV_per_A'):
            b, f = pair['baseline'][key], pair['finetuned'][key]
            rows.append({'run': label, 'metric': key, 'baseline': b, 'finetuned': f,
                         'ratio': f / b, 'relative_change_percent': 100 * (f / b - 1)})
    return rows


def followup_figure(data, output):
    import matplotlib.pyplot as plt

    models = ('baseline', 'parent_stage1', 'followup_epoch_1')
    rows = {r['model']: r for r in data['polar_validation'] if r['group'] == 'all'}
    if not set(models).issubset(rows):
        raise ValueError('Expected POLAR validation models missing')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    labels = ['Foundation', 'Stage 1', 'Folgelauf']
    values = [float(rows[m]['force_rmse_ev_per_A']) * 1000 for m in models]
    bars = axes[0].bar(range(3), values, color=['#2676a6', '#d27031', '#4c9277'])
    axes[0].bar_label(bars, fmt='%.2f', padding=4)
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='Validierungs-Kraft-RMSE (meV/Å)', ylim=(0, 72),
                title='HKUST-1 und MIL-53-Al · 46 Einträge')
    x = np.arange(3)
    for i, ref in enumerate(('UFF', 'DREIDING')):
        values = [float(next(r for r in data['polar_isotherm_metrics'] if r['model'] == m and ref in r['reference'])['rmse_mol_per_kg'])
                  for m in ('baseline', 'parent_stage1', 'followup_selected')]
        bars = axes[1].bar(x + (i - .5) * .36, values, .34, label=f'CRAFTED {ref}', color=['#2676a6', '#8b6cab'][i])
        axes[1].bar_label(bars, fmt='%.3f', fontsize=9, padding=4)
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel='Isothermen-RMSE (mmol/g)', ylim=(0, 1.95),
                title='ZIF-8 / CO₂ · 298 K · vier Druckpunkte')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('POLAR-Folgelauf: bessere Validierungskräfte, höhere Isothermenabweichung',
                 x=.06, ha='left', fontsize=15, weight='bold')
    fig.text(.06, .025, 'Unterschiedliche Systeme und Messgrößen; keine Korrelationsanalyse. Isothermen: ein Seed, begrenztes Sampling.\nCRAFTED ist eine klassische Simulationsreferenz. Checkpointwahl über Validierungskräfte, nicht über Isothermen.\nArchivierte Kennzahlen; Identität der lokalen Folgemodelldatei siehe DATENPRUEFUNG.md.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.23, wspace=.30)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'07_validation_and_adsorption.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    return '07_validation_and_adsorption'


def write_sections(data, audit, sources, output, as_of):
    summary = [r for r in data['fine_tuning'] if r['group'] == 'all']
    ratios = paired_metrics(data)
    mh = {r['model']: r for r in data['mh_assessment']['reference_metrics']}
    valid = {r['model']: r for r in data['polar_validation'] if r['group'] == 'all'}
    before, after = valid['parent_stage1'], valid['followup_epoch_1']
    force_change = 100 * (float(after['force_rmse_ev_per_A']) / float(before['force_rmse_ev_per_A']) - 1)
    energy_change = 100 * (float(after['energy_rmse_ev_per_atom']) / float(before['energy_rmse_ev_per_atom']) - 1)
    entry_count = audit['split_checks'][0]['test_entries']
    unique_count = audit['split_checks'][0]['unique_test_frame_ids']
    followup_note = ('Die Prüfsumme der lokalen POLAR-Folgemodelldatei weicht vom archivierten Auswahlbericht ab. '
        'Die folgenden Aussagen beziehen sich auf die archivierten Kennzahlen; eine Reproduktion mit der heutigen '
        'Datei oder deren Gleichwertigkeit ist nicht bestätigt. Die Ursache der Abweichung bleibt offen.'
        if not audit['followup_checkpoint_identity']['matches'] else
        'Die lokale POLAR-Folgemodelldatei stimmt mit der archivierten Auswahlprüfsumme überein.')
    readable_ratios = [{**r, 'metric': 'Energie-RMSE' if r['metric'].startswith('energy') else 'Kraft-RMSE'} for r in ratios]
    readable_isotherms = [{**r, **{k: float(r[k]) for k in (
        'rmse_mol_per_kg', 'mean_signed_deviation_mol_per_kg', 'mean_absolute_relative_deviation_percent')}}
        for r in data['polar_isotherm_metrics']]
    result = f'''# Ergebnisse und Diskussion des Modul-C-Benchmarks

Datenprüfung vom {as_of}. Dieser Text wertet den lokal verfügbaren Bestand aus; ausstehende Clusterergebnisse sind nicht enthalten. Die zugehörige [Methodik](METHODIK.md) beschreibt Auswahl, Fehlermaße und Aussagegrenzen. Alle Zahlen sind in [results.json](results.json) und den begleitenden Tabellen nachvollziehbar.

## Fine-Tuning verbessert die untersuchten Transferfehler nicht durchgängig

Für die abgeschlossenen Läufe MH 10 %, MH 25 % und POLAR 25 % wurde das jeweils über Validierungsdaten ausgewählte Modell mit seiner unveränderten Foundation verglichen. Auf dem Transfer-Testsatz steigen in allen drei Fällen sowohl der aggregierte Energie-RMSE als auch der Kraft-RMSE. Ein genereller Genauigkeitsgewinn durch das angewandte Fine-Tuning ist für diesen Testsatz damit nicht belegt.

{table(summary, [('run','Lauf'), ('model','Modell'), ('energy_rmse_meV_per_atom','Energie-RMSE (meV/Atom)'), ('force_rmse_meV_per_A','Kraft-RMSE (meV/Å)')])}

{table(readable_ratios, [('run','Lauf'), ('metric','Messgröße'), ('ratio','Fine-Tuning / Foundation'), ('relative_change_percent','Änderung (%)')])}

![Aggregierte Testfehler vor und nach Fine-Tuning](01_finetuning_test.png)

Die nominalen Datenanteile entsprechen infolge der stratifizierten Auswahl nicht exakt dem tatsächlichen Anteil: MH 10 % verwendet 109 von 774 Konfigurationen (14,08 %), die beiden 25-%-Läufe jeweils 211 (27,26 %). MH 25 % erzielt einen kleineren Energiefehler als MH 10 %, gleichzeitig aber einen größeren Kraftfehler. Aus diesen zwei Läufen mit einem Trainingsseed lässt sich keine monotone Lernkurve oder optimale Datenmenge ableiten. Die unterbrochenen bzw. lediglich vorbereiteten 50-/100-%-Läufe werden nicht als Ergebnisse behandelt.

Die aggregierten Ergebnisse verdecken materialabhängige Unterschiede. Beispielsweise verbessert POLAR 25 % den Energiefehler für MOF-303, während sich der Fehler für MgMOF-74 erhöht. Die MOF- und Gasgruppen überlappen und sind keine unabhängigen Stichproben. Die Gasgruppen beziehen sich jeweils auf das vollständige beladene System, nicht auf isolierte Gastmolekülkräfte.

![Fehleränderungen nach MOF und Gasgruppe](02_material_gas_effects.png)

Der Testsatz umfasst {entry_count} Einträge, jedoch nur {unique_count} unterschiedliche Frame-IDs. Die archivierte Gewichtung nach Einträgen wird beibehalten. Zwischen den verwendeten Trainings-, Validierungs- und Test-Frame-IDs wurde keine Überschneidung gefunden; eine Überschneidung mit dem Foundation-Pretraining ist dadurch nicht ausgeschlossen. Insbesondere der sehr kleine MH-Foundation-Fehler darf daher nicht ohne zusätzliche Provenienzprüfung als Nachweis außergewöhnlicher Generalisierung interpretiert werden.

## Validierungsverbesserung und Adsorptionsvorhersage sind getrennt zu beurteilen

{followup_note}

Der POLAR-Folgelauf senkt den Validierungs-Kraft-RMSE gegenüber Stage 1 von {float(before['force_rmse_ev_per_A'])*1000:.2f} auf {float(after['force_rmse_ev_per_A'])*1000:.2f} meV/Å ({force_change:.1f} %). Zugleich steigt der Energie-RMSE von {float(before['energy_rmse_ev_per_atom'])*1000:.2f} auf {float(after['energy_rmse_ev_per_atom'])*1000:.2f} meV/Atom ({energy_change:+.1f} %). Die Auswahl des Folgemodells erfolgt anhand der Validierungskräfte und nicht anhand des anschließenden CRAFTED-Vergleichs.

Im ZIF-8/CO₂-Piloten liegt der Isothermen-RMSE gegenüber CRAFTED UFF/DDEC für Stage 1 bei 0,543 mmol/g und für den Folgelauf bei 0,776 mmol/g. Die verbesserten Validierungskräfte gehen in diesem Vergleich also nicht mit einem kleineren Isothermenfehler einher. Das ist ein deskriptiver Befund für diese Modelle und Systeme, kein allgemeiner Korrelationsnachweis: Die Kraftvalidierung verwendet HKUST-1 und MIL-53-Al, die Adsorptionsrechnung ZIF-8. Außerdem sind die Isothermen bislang nur durch einen Seed und kurze Ketten abgesichert.

![Validierungskräfte und Isothermenabweichungen des POLAR-Folgelaufs](07_validation_and_adsorption.png)

## Die Adsorptionsbewertung hängt von Referenz und Fehlermaß ab

Im MH/MOF-5-Vergleich bei 298 K beträgt der deskriptive Isothermen-RMSE gegen CRAFTED UFF/DDEC {mh['umof_mh_pretrained']['rmse_mol_per_kg']:.3f} mmol/g für die Foundation und {mh['umof_mh_finetuned_025pct']['rmse_mol_per_kg']:.3f} mmol/g für das 25-%-Modell. Bei 10 bar liegen die mittleren Beladungen bei 5,924 bzw. 25,092 mmol/g. Vier von sechs Modell-Druckpunkten bestehen den archivierten Stabilitätsscreen nicht; deshalb bleibt die quantitative Modellrangfolge offen.

Für POLAR/ZIF-8 ergibt sich die folgende Bewertung:

{table(readable_isotherms, [('reference','Referenz'), ('model','Modell'), ('rmse_mol_per_kg','RMSE (mmol/g)'), ('mean_signed_deviation_mol_per_kg','Bias (mmol/g)'), ('mean_absolute_relative_deviation_percent','Mittlere absolute relative Abweichung (%)')])}

Gegen UFF/DDEC sinkt der absolute Isothermen-RMSE von der Foundation zu Stage 1 deutlich. Die mittlere relative Abweichung verbessert sich jedoch nicht. Gegen DREIDING/DDEC liegen die RMSEs von Foundation und Stage 1 nahe beieinander; die relative Abweichung nimmt ebenfalls zu. Die Aussage „Fine-Tuning verbessert die Isotherme“ benötigt deshalb immer die Angabe von Modell, Material, Referenz und Fehlermaß. CRAFTED liefert klassische Simulationen und keine experimentelle Wahrheit.

![Isothermen der getrennten MH- und POLAR-Untersuchungen](03_isotherms.png)

## Sampling und Rechenaufwand begrenzen die Interpretation

Ein formal abgeschlossener Simulationslauf ist kein Konvergenznachweis. Im MH-Piloten zeigen Autokorrelations- und Blockdiagnostiken offene Punkte; bei POLAR ändern sich teilweise die mittleren Beladungen zwischen den Produktionshälften deutlich. Diese Hälften sind abhängig und ersetzen keine unabhängigen Seeds. Daher werden für die Ein-Seed-Isothermen keine scheinbar belastbaren Seed-Konfidenzintervalle angegeben.

![Samplingdiagnostik der vorhandenen Isothermen](06_sampling_limits.png)

Die POLAR/ZIF-8-Läufe benötigen archiviert etwa 0,155 s je MC-Schritt für Foundation und Stage 1 sowie 0,164 s für den Folgelauf. Da Geräte und Beladungszustände nicht durchgängig kontrolliert sind, belegen diese Zeiten keinen isolierten Inferenzvorteil. Trainingszeiten, statische Energieauswertungen, Widomversuche und GCMC-Schritte werden getrennt berichtet; ihre Mittelwerte sind nicht austauschbar.

## Ergänzende Potentialvergleiche

Der GoldDAC-Vergleich umfasst vier Potential-/Dispersionsvarianten auf denselben 156 CO₂-Konfigurationen. Er prüft Interaktionsenergien gegen DFT und ergänzt damit die Fehler vollständiger beladener Systeme. Die Ergebnisse zeigen, dass Potential und Dispersionskorrektur gemeinsam beurteilt werden müssen. Die Einheiten bleiben hier eV pro Interaktionskonfiguration; sie werden nicht direkt mit meV/Atom des Fine-Tunings verglichen.

{table([r for r in data['gold'] if r['region']=='all'], [('model','Potential'), ('energy_mae_eV','Interaktions-MAE (eV)'), ('energy_bias_eV','Bias (eV)'), ('mean_runtime_seconds','Zeit (s/Konfiguration)')])}

![DFT-Vergleich mit und ohne Dispersionskorrektur](05_golddac_dispersion.png)

Der ZIF-8-Widom-Vergleich bei 273 K enthält für MACE-MP-0a, MACE-DAC und NequIP jeweils drei Seeds mit 100000 Einfügungsversuchen. Die Konfidenzintervalle werden über Seedmittel berechnet. Das archivierte NequIP-Seed-Präzisionsziel wird nicht erreicht; eine ausreichende gepoolte Blockpräzision hebt diesen Befund nicht auf. Die zusätzlichen 298,15-K-Daten bleiben getrennt. Referenzwärmen bei endlichem Niederdruck sind keine exakten Nullbeladungsgrenzwerte.

![Henry-Koeffizienten und Adsorptionswärmen](04_widom_potential_classes.png)

## Schlussfolgerung und noch offene Nachweise

Die vorhandenen Daten belegen keine allgemeine Verbesserung durch Fine-Tuning. Die beobachteten Änderungen hängen von Material, Messgröße und Referenz ab. Insbesondere liefert die bessere Kraftvalidierung des POLAR-Folgelaufs keinen zusätzlichen Isothermenvorteil im untersuchten ZIF-8-Piloten. Wegen unterschiedlicher Validierungs-/Adsorptionssysteme, begrenzten Samplings und jeweils eines Trainingsseeds bleibt diese Aussage auf den vorhandenen Bestand beschränkt.

Für eine belastbare quantitative Adsorptionsrangfolge fehlen unabhängige Wiederholungen und ausreichend stabile Ketten an den relevanten Druckpunkten. Das bereits vorbereitete Clusterpaket adressiert diese Lücke; seine geplanten Rechnungen zählen hier nicht als Ergebnisse. Zusätzliche DFT-Diagnostik könnte systematische Bindungsenergiefehler erklären, ist aber keine Voraussetzung dafür, die bereits beobachteten Grenzen transparent zu berichten. Mischungs- und Gerüstflexibilitätsvalidierung sind durch diesen Modul-C-Datensatz nicht erbracht.

## Datenprüfung und Quellen

Die Energie- und Kraft-MAE/RMSE wurden aus den archivierten Einzelvorhersagen neu berechnet und mit CSV-/JSON-Berichten abgeglichen. Referenzlabels, Testdateihashes und ausgewählte Modellgewichte wurden geprüft. Die MH-Matrixübersicht enthält veraltete `prepared`-Statuswerte; ausschlaggebend sind die geprüften Einzelläufe. Die Quellübersicht wurde nicht überschrieben. Details: [Datenprüfung](DATENPRUEFUNG.md), [Prüfergebnisse](audit.json), [Quellenprüfsummen](provenance.json).
'''
    (output / 'ERGEBNISSE_UND_DISKUSSION.md').write_text(result)
    protocols = [{**r['protocol'], 'run':r['run'], 'training_frames':r['training_frames'],
                  'fraction_percent':100*r['actual_training_fraction']} for r in data['runs']]
    method = f'''# Methodik des Modul-C-Potentialbenchmarks

## Ziel und Vergleichsdesign

Untersucht wird, wie Potentialklasse und Fine-Tuning die Vorhersage von Energien, Kräften und Adsorptionsgrößen beeinflussen. Der Vergleich umfasst drei getrennte Ebenen: Fehler vollständiger beladener Konfigurationen, DFT-Interaktionsenergien und Adsorptionssimulationen. Ein unverändertes Foundation-Modell dient jeweils als profilinterne Baseline. Die MH/MOF-5- und POLAR/ZIF-8-Adsorptionsstudien verwenden unterschiedliche Materialien und erlauben daher keinen isolierten Vergleich der Modellarchitekturen.

## Datenaufteilung und Fine-Tuning

Die Fine-Tuning-Daten stammen aus den veröffentlichten uMOF-Konfigurationen mit r2SCAN-D4-Energie- und Kraftlabels. Verwendet werden reine CO₂-, N₂- und CH₄-Beladungen. Die Trainingsmaterialien sind CAU-10, IRMOF-3, MOF-5, die beiden MOF-808-Varianten, Al-MIL-120, ALF und CALF-20-beta entsprechend den archivierten Bezeichnungen. HKUST-1 und MIL-53-Al bilden den Validierungssatz; MOF-303 und MgMOF-74 den Testsatz.

Der Vergleich enthält MH mit nominell 10 % und 25 % sowie POLAR mit nominell 25 % Trainingsdaten. Die stratifizierte Auswahl führt zu den folgenden tatsächlichen Anteilen und Protokollen:

{table(protocols, [('run','Lauf'), ('training_frames','Trainingskonfigurationen'), ('fraction_percent','Tatsächlicher Anteil (%)'), ('max_epochs','Max. Epochen'), ('replay_samples','Replay-Konfigurationen'), ('device','Archiviertes Gerät'), ('dtype','Datentyp')])}

Der Lernratenstart beträgt jeweils 1e-4 bei Batchgröße 1. Die archivierten Protokolle enthalten Replay-Gewicht 0,25 und den Scheduler ReduceLROnPlateau. Unterschiede in Replay, Epochenbudget, numerischer Präzision und Rechenhardware werden als Vergleichsgrenzen behandelt. Für POLAR wurden vorhandene Ladungs-/Spinmetadaten verwendet; fehlende Werte wurden mit Ladung 0 und Multiplizität 1 belegt. Diese Defaults sind keine abgeleiteten Spinlabels. Stress und Virial wurden nicht trainiert.

Die Modellselektion verwendet den Validierungs-Kraft-RMSE, danach den Validierungs-Energie-RMSE und bei Gleichstand Stage 1. Der Testsplit wird nicht zur Checkpointwahl verwendet. Alle hier ausgewerteten ursprünglichen Fine-Tuning-Läufe wählen Stage 1. Der zusätzliche POLAR-Folgelauf wird getrennt als Validierungsanalyse geführt. Ein Trainingsseed erlaubt keine Schätzung der Streuung zwischen unabhängigen Trainingsläufen.

MH und POLAR mit 25 % verwenden dieselben Geometrien und Referenzlabels in gleicher Reihenfolge: 211 Trainings-, 46 Validierungs- und 38 Testeinträge. Die Testeinträge entsprechen 36 unterschiedlichen Frame-IDs. Überschneidungen der IDs zwischen den drei Splits wurden geprüft; eine Prüfung auf Foundation-Pretraining-Überlappung liegt nicht vor. Alle aggregierten Metriken behalten die ursprüngliche Gewichtung nach Einträgen bei.

## Energie- und Kraftfehler

Für jede Konfiguration i wird die Energiedifferenz pro Atom als Δeᵢ = (Eᵢᴹᵒᵈᵉˡˡ − Eᵢᴿᵉᶠ)/Nᵢ berechnet. Der Energie-MAE ist der Mittelwert von |Δeᵢ|; der Energie-RMSE die Quadratwurzel des Mittelwerts von Δeᵢ². Jede Konfiguration hat dabei gleiches Gewicht. Kraftfehler werden über sämtliche kartesischen Kraftkomponenten aller Atome gepoolt; größere Konfigurationen liefern damit mehr Kraftkomponenten. Die Ausgabe verwendet meV/Atom bzw. meV/Å (Umrechnung aus eV mit Faktor 1000).

Die nach MOF oder Gas gruppierten Werte beziehen sich jeweils auf die vollständigen Konfigurationen. Energiefehler pro Atom sind nicht mit Host-Gast-Interaktionsenergien gleichzusetzen. Letztere werden im separaten GoldDAC-Vergleich in eV bewertet. Positive Residuen bedeuten Modell minus Referenz; ein negativer Interaktionsenergiebias entspricht einer zu günstigen vorhergesagten Wechselwirkung.

## Adsorption und Referenzvergleich

Der MH/MOF-5-Pilot umfasst CO₂ bei 298 K und 0,1, 1 und 10 bar. Der POLAR/ZIF-8-Pilot umfasst 0,1, 0,5, 1 und 5 bar bei 298 K mit 1000 Equilibrierungs- und 2000 Produktionsschritten je Punkt. Historische Adapter- und Samplingprotokolle werden beibehalten; die Studien werden nicht als einheitliche kontrollierte Architekturanalyse zusammengefasst.

Absolute Beladungen werden gegen die passenden CRAFTED-Isothermen innerhalb des gemeinsamen Druckbereichs verglichen. Die Einheit mol/kg ist zahlenmäßig identisch zu mmol/g. RMSE, MAE, vorzeichenbehafteter Bias und mittlere absolute relative Abweichung werden über die verglichenen Druckpunkte berichtet. Die Referenzwahl UFF/DDEC oder DREIDING/DDEC bleibt sichtbar. Eine klassische Simulationsreferenz wird nicht als experimenteller Genauigkeitsnachweis bezeichnet; unbekannte Absolute-/Excess-Basen werden nicht gleichgesetzt.

## Sampling, Unsicherheit und Laufzeit

Ein-Seed-Isothermen erhalten keine Konfidenzintervalle aus vermeintlich unabhängigen MC-Schritten. Mittelwerte der Produktionshälften, Blockdiagnostik und Autokorrelation beschreiben die Stabilität der jeweiligen Kette. Sie ersetzen keine unabhängigen Wiederholungen und beweisen kein Gleichgewicht. Im Widom-Vergleich bei 273 K liegen drei Seeds mit je 100000 Versuchen vor; 95-%-Student-t-Intervalle beziehen sich auf die Seedmittel. Ergebnisse anderer Temperaturen werden nicht gepoolt.

Laufzeiten werden nach Trainingslauf, statischer Konfiguration, Widomversuch und GCMC-Schritt getrennt ausgewiesen. Unterschiede in Gerät, Präzision und Beladung begrenzen die Vergleichbarkeit. Eine Vollkostenrechnung einschließlich Datenbeschaffung oder DFT-Labelerzeugung ist nicht Bestandteil der archivierten Zeiten.

## Reproduzierbarkeit der lokalen Auswertung

Stand: {as_of}. Diese Auswertung startet weder Inferenz noch Training oder Simulation. Einzelvorhersagen, Testlabels, Checkpoint-Hashes und die Auswahlregel werden geprüft. Die veraltete zentrale MH-Statusübersicht wird separat gekennzeichnet. Alle gelesenen Dateien und die Auswertungsskripte sind in `provenance.json` mit SHA-256 erfasst; nach dem Lauf werden die Quellen auf unveränderte Inhalte geprüft.

{followup_note}

Aufruf aus dem Projektverzeichnis:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mof-c-review-mpl \\
python -m analysis.module_c_local_review --as-of {as_of} --output /tmp/module_c_review_reproduced
```

Literatur- und Softwarezitate für die endgültige Arbeit sind aus der Projektbibliografie zu übernehmen. Dieser Abschnitt dokumentiert das tatsächlich archivierte Versuchsdesign und erfindet keine fehlenden Produktions- oder Validierungsergebnisse.
'''
    (output / 'METHODIK.md').write_text(method)
    (output / 'DATENPRUEFUNG.md').write_text(f'''# Prüfung des lokal verfügbaren Datenbestands

Stand: {as_of}. Keine neuen Modellvorhersagen, Trainings- oder Simulationsläufe.

- {len(audit['prediction_checks'])} neu berechnete MAE/RMSE-Werte stimmen innerhalb der numerischen Toleranz mit den archivierten Einzelberichten überein (relativ {METRIC_RTOL:g}, absolut {METRIC_ATOL:g} in eV-Einheiten). Die Toleranz berücksichtigt die begrenzte Dezimalpräzision der archivierten Float32-Vorhersagen; Einzelabweichungen stehen in `audit.json`.
- Energie- und Kraftreferenzen der Paritätstabellen stimmen mit den Testlabels überein.
- Testdateien und ausgewählte Checkpoints der drei ursprünglichen Fine-Tuning-Läufe stimmen mit ihren archivierten Hashes überein.
- Keine Überlappung der verwendeten Frame-IDs zwischen Training, Validierung und Test; Foundation-Pretraining bleibt ungeprüft.
- MH/POLAR-25-%-Geometrien und Labels sind identisch; 38 Testeinträge enthalten 36 unterschiedliche Frame-IDs.

## Inkonsistente Statusübersichten

{table(audit['status_checks'], [('run','Lauf'), ('matrix_status','Matrixübersicht'), ('run_status','Einzellauf'), ('consistent','Konsistent')])}

Die Auswertung verwendet die verifizierten Einzelläufe. Die ursprünglichen Kampagnen und die veralteten Übersichten werden nicht verändert.

## Aussagegrenzen

{followup_note}

- Archivierter Folgemodell-Hash: `{audit['followup_checkpoint_identity']['recorded_sha256']}`
- Lokaler Folgemodell-Hash: `{audit['followup_checkpoint_identity']['local_sha256']}`

Die Kennzahlenprüfung beweist keine Unabhängigkeit vom Pretraining, keine physikalische Genauigkeit und keine GCMC-Konvergenz. Die Isothermen beruhen auf jeweils einem Seed. Der ursprüngliche 50-%-MH-Lauf ist unterbrochen; ein ausgewerteter 100-%-Lauf liegt nicht vor.
''')
    write_csv(output / 'fine_tuning_changes.csv', ratios)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', type=date.fromisoformat, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / f'reports/module_c_local_review_{args.as_of}'
    output.mkdir(parents=True, exist_ok=False)
    sources = Sources(ROOT)
    data = collect(sources)
    audit = audit_predictions(sources, data)
    figures = plot_all(data, sources, output)
    figures.append(followup_figure(data, output))
    write_sections(data, audit, sources, output, str(args.as_of))
    for name, key in [('fine_tuning_metrics_meV', 'fine_tuning'), ('gold_metrics', 'gold'),
                      ('polar_isotherm_metrics', 'polar_isotherm_metrics'), ('polar_validation_metrics', 'polar_validation'),
                      ('runtime_polar', 'polar_runtime')]:
        write_csv(output / f'{name}.csv', data[key])
    dump(output / 'results.json', data)
    dump(output / 'audit.json', audit)
    for path, expected in sources.files.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Source changed during review: {path}')
    import ase, scipy, matplotlib
    dump(output / 'provenance.json', {'as_of': str(args.as_of), 'source_sha256': sources.files,
         'scripts_sha256': {str(Path(__file__).relative_to(ROOT)): digest(__file__),
                            'analysis/module_c_closure.py': digest(ROOT / 'analysis/module_c_closure.py')},
         'source_files_unchanged': True, 'new_simulations': False, 'new_predictions': False,
         'versions': {'ase':ase.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__, 'matplotlib':matplotlib.__version__},
         'figures': figures})
    print(json.dumps({'output':str(output), 'figures':len(figures),
                      'prediction_checks_passed':len(audit['prediction_checks']),
                      'stale_status_rows':sum(not r['consistent'] for r in audit['status_checks'])}, indent=2))


if __name__ == '__main__':
    main()
