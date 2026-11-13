"""Audit returned uMOF cluster traces and write a separate local update; no inference.

python -m analysis.module_c_return_review --as-of 2026-11-10
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
from ase.io import read

from analysis.module_c_closure import ROOT, Sources, digest, dump, error_metrics, seed_interval, table, write_csv
from engines.mlip_mc_runner import _read_gcmc_transition_log, _reconstruct_gcmc_samples
from modules.module_c_mlips.closure_campaign import job_list
from modules.module_c_mlips.umof_isotherm_stability import assess

BUNDLE = 'reports/module_c_closure/cluster_bundle'
CRITERIA = ('at_least_20000_samples', 'at_least_20_effective_samples',
            'precision_within_tolerance', 'half_drift_within_tolerance')


def close(actual, expected, context):
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f'{context}: recomputed {actual}, archived {expected}')


def collect_returns(sources):
    manifest_path = sources.path(BUNDLE + '/bundle_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest['file_sha256'].items():
        if digest(sources.path(BUNDLE + '/' + path)) != expected:
            raise ValueError(f'Changed bundle input: {path}')
    review = sources.json('outputs/umof_polar_followup_evaluation/training_review.json')
    result = {'systems': {}, 'bundle_files_verified': len(manifest['file_sha256']),
              'scope': 'Returned scientific runs only; smoke runs excluded; no new simulation or inference.'}
    for system in ('mh', 'polar'):
        config_path = sources.path(f'{BUNDLE}/{system}.json')
        config = json.loads(config_path.read_text())
        jobs = job_list(config)
        if config.get('technical_smoke') or len(config['seeds']) != 3:
            raise ValueError('Expected three-seed scientific campaign')
        if system == 'polar':
            for model, archived in zip(config['models'], ('baseline', 'parent_stage1')):
                candidate = next(r for r in review['candidates'] if r['label'] == archived)
                if candidate['sha256'] != model['sha256']:
                    raise ValueError('POLAR model mapping differs from archived Stage 1 comparison')
        host = read(sources.path(BUNDLE + '/' + config['framework']['path']))
        factor = 1000 / host.get_masses().sum()
        g = config['gcmc']
        total = g['equilibration_steps'] + g['production_steps']
        folder = f'{BUNDLE}/results/{system}'
        archived = sources.json(folder + '/report/assessment.json')
        archived_seeds = sources.csv(folder + '/report/seed_results.csv')
        rows, screens, blocks, environments = [], [], [], set()
        for job in jobs:
            job_folder = folder + '/' + job['id']
            marker = sources.json(job_folder + '/completed.json')
            identity = marker['identity']
            if identity['bundle_sha256'] != digest(manifest_path) or identity['config_sha256'] != digest(config_path):
                raise ValueError('Result input identity differs from bundle/config')
            environments.add(json.dumps(identity['runtime_versions'], sort_keys=True))
            path = sources.path(job_folder + '/' + marker['result_path'])
            if digest(path) != marker['result_sha256']:
                raise ValueError(f'Result checksum mismatch: {job["id"]}')
            raw = json.loads(path.read_text())
            if raw['status'] != 'completed' or raw['model'] != job['model'] or raw['seed'] != job['seed']:
                raise ValueError('Result model/seed/status mismatch')
            if len(raw['pressure_points']) != 1:
                raise ValueError('Expected an isolated pressure point')
            point = raw['pressure_points'][0]
            for key in ('pressure_bar', 'seed'):
                close(point[key], job[key], key)
            for key in ('equilibration_steps', 'production_steps'):
                close(point[key], g[key], key)
            close(raw['temperature_K'], config['temperature_K'], 'temperature')
            matches = list(path.parent.rglob(Path(point['transition_log_path']).name))
            if len(matches) != 1:
                raise ValueError('Expected one transition log per job')
            trace_path = sources.path(str(matches[0].relative_to(ROOT)))
            transitions = _read_gcmc_transition_log(trace_path)
            trace = _reconstruct_gcmc_samples(transitions, total_steps=total)
            values = np.array([r['uptake'] for r in trace[g['equilibration_steps']:]], float) * factor
            if len(values) != g['production_steps'] or np.any(values < 0):
                raise ValueError('Invalid production trace')
            restarts = list(trace_path.parent.glob('restart/restart_*.json'))
            if len(restarts) != 1:
                raise ValueError('Expected one final restart')
            restart = sources.json(str(restarts[0].relative_to(ROOT)))
            close(restart['n_equil_completed'], g['equilibration_steps'], 'restart equilibration')
            close(restart['n_prod_completed'], g['production_steps'], 'restart production')
            close(values.mean(), point['loading_mol_per_kg'], 'raw loading')
            close(point['production_sample_count'], len(values), 'sample count')
            screen = assess(values)
            old = next(r for r in archived_seeds if r['model'] == job['model']
                       and float(r['pressure_bar']) == job['pressure_bar'] and int(r['seed']) == job['seed'])
            for actual, key in ((values.mean(), 'loading_mmol_g'),
                                (screen['effective_sample_count_estimate'], 'effective_samples'),
                                (screen['half_difference_mmol_g'], 'drift_mmol_g')):
                close(actual, float(old[key]), 'seed report ' + key)
            if screen['status'] != old['chain_screen']:
                raise ValueError('Archived chain screen does not reproduce')
            rows.append({'system': system, **job, 'loading_mmol_g': float(values.mean()),
                         'chain_screen': screen['status'], 'effective_samples': screen['effective_sample_count_estimate'],
                         'drift_mmol_g': screen['half_difference_mmol_g'], 'runtime_seconds': raw['runtime_seconds'],
                         **screen['criteria']})
            screens.append({'job': job['id'], **screen})
            for i, chunk in enumerate(np.array_split(values, 20)):
                blocks.append({'system': system, **job, 'block': i + 1, 'loading_mmol_g': float(chunk.mean())})
        if len(environments) != 1 or len(rows) != archived['completed_jobs'] or len(jobs) != archived['expected_jobs']:
            raise ValueError('Runtime environments or completion counts disagree')
        points = []
        for model in ('pretrained', 'finetuned'):
            for pressure in config['pressures_bar']:
                selected = [r for r in rows if r['model'] == model and r['pressure_bar'] == pressure]
                stats = seed_interval([r['loading_mmol_g'] for r in selected])
                target = max(config['precision']['absolute_floor_mmol_g'],
                             config['precision']['relative_seed_ci95_target'] * abs(stats['mean']))
                passed = all(r['chain_screen'] == 'passes_operational_screen' for r in selected)
                seed_pass = stats['ci95_half_width'] <= target
                point = {'system': system, 'model': model, 'pressure_bar': pressure, **stats,
                         'chains_passed': sum(r['chain_screen'] == 'passes_operational_screen' for r in selected),
                         'seed_precision_passed': seed_pass, 'all_screens_passed': passed and seed_pass}
                old = next(r for r in archived['points'] if r['model'] == model and r['pressure_bar'] == pressure)
                for key in ('mean', 'ci95_half_width', 'relative_ci95_half_width', 'replicates'):
                    close(point[key], old[key], 'point report ' + key)
                if point['all_screens_passed'] != old['passes_precision_and_chain_screens']:
                    raise ValueError('Point sampling screen differs')
                points.append(point)
        references, metrics = [], []
        for ref in config['references']:
            xy = np.loadtxt(sources.path(BUNDLE + '/' + ref['path']), delimiter=',')
            references.append({'label': ref['label'], 'pressure_bar': (xy[:, 0] / 1e5).tolist(), 'loading_mmol_g': xy[:, 1].tolist()})
            for model in ('pretrained', 'finetuned'):
                selected = [r for r in points if r['model'] == model]
                p = [r['pressure_bar'] for r in selected]
                if min(p) < xy[:, 0].min()/1e5 or max(p) > xy[:, 0].max()/1e5:
                    raise ValueError('Reference interpolation would extrapolate')
                errors = error_metrics([r['mean'] for r in selected], np.interp(p, xy[:, 0]/1e5, xy[:, 1]))
                old = next(r for r in archived['reference_metrics'] if r['model'] == model and r['reference'] == ref['label'])
                for key, value in errors.items(): close(value, old[key], 'reference ' + key)
                metrics.append({'system': system, 'model': model, 'reference': ref['label'], **errors})
        result['systems'][system] = {'config': config, 'seeds': rows, 'screens': screens, 'blocks': blocks,
                                    'points': points, 'references': references, 'reference_metrics': metrics,
                                    'runtime_versions': json.loads(next(iter(environments))),
                                    'archived_report_reproduced': True}
    return result


def figures(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'pretrained': '#2676a6', 'finetuned': '#d27031'}
    names = {'pretrained': 'Foundation', 'finetuned': 'Fine-Tuning (Stage 1)'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (system, d) in zip(axes, data['systems'].items()):
        for ref in d['references']:
            ax.plot(ref['pressure_bar'], ref['loading_mmol_g'], '--', lw=1.3,
                    label='CRAFTED ' + ('DREIDING' if 'DREIDING' in ref['label'] else 'UFF'), alpha=.8)
        for model in names:
            rows = [r for r in d['points'] if r['model'] == model]
            ax.errorbar([r['pressure_bar'] for r in rows], [r['mean'] for r in rows],
                        yerr=[r['ci95_half_width'] for r in rows], fmt='o-', capsize=4,
                        color=colors[model], label=names[model])
            seeds = [r for r in d['seeds'] if r['model'] == model]
            ax.scatter([r['pressure_bar'] for r in seeds], [r['loading_mmol_g'] for r in seeds],
                       marker='x', s=25, color=colors[model], alpha=.7)
        p = d['config']['pressures_bar']
        ax.set(xscale='log', xlim=(min(p)*.7, max(p)*1.4), xlabel='Pressure (bar)', ylabel='Absolute loading (mmol/g)',
               title=f'{system.upper()} / {d["config"]["material"]} · {len(d["seeds"])} runs')
        ax.set_xticks(p)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('uMOF cluster returns: all runs completed, sampling remains unresolved', fontsize=15, weight='bold')
    fig.text(.06, .025, 'CO₂ · 298 K · 20,000 equilibration + 40,000 production steps per seed. Crosses: three seed means.\nBars: 95% Student-t intervals across seeds, not equilibrium errors. Negative interval limits are statistical artifacts.\nNone of the 14 model-pressure points passes all checks. CRAFTED: classical simulation reference.', fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.24, wspace=.3)
    for ext in ('png', 'svg'): fig.savefig(output / f'01_returned_isotherms.{ext}', dpi=180)
    plt.close(fig)

    all_points = [p for d in data['systems'].values() for p in d['points']]
    failures = []
    labels = []
    for p in all_points:
        rows = [r for r in data['systems'][p['system']]['seeds'] if r['model'] == p['model'] and r['pressure_bar'] == p['pressure_bar']]
        failures.append([sum(not r[k] for r in rows) for k in CRITERIA])
        labels.append(f'{p["system"].upper()} · {"Fine-tuned" if p["model"] == "finetuned" else "Foundation"} · {p["pressure_bar"]:g} bar')
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(failures, cmap='Oranges', vmin=0, vmax=3, aspect='auto')
    for i, row in enumerate(failures):
        for j, val in enumerate(row): ax.text(j, i, f'{val}/3', ha='center', va='center', color='white' if val >= 2 else '#222')
    ax.set(yticks=range(len(labels)), yticklabels=labels, xticks=range(4),
            xticklabels=['≥ 20,000\nproduction steps', '≥ 20 effective\nsamples', 'Block interval\nwithin tolerance', 'Half-chain drift\nwithin tolerance'])
    ax.set_title('Number of seeds failing each chain criterion', pad=18, weight='bold')
    fig.text(.06, .025, 'Chain screen: at least 20 effective samples, ≥ 8 blocks for the interval; tolerance max(0.02 mmol/g, 15%).\nAdditional seed check: 95% interval half-width ≤ max(0.02 mmol/g, 20%); only 3 of 14 points pass.\nPassing an individual criterion does not establish equilibrium.', fontsize=9)
    fig.subplots_adjust(left=.31, right=.97, top=.90, bottom=.20)
    for ext in ('png', 'svg'): fig.savefig(output / f'02_sampling_failures.{ext}', dpi=180)
    plt.close(fig)


def write_report(data, output, as_of):
    points = [r for d in data['systems'].values() for r in d['points']]
    seeds = [r for d in data['systems'].values() for r in d['seeds']]
    metrics = [r for d in data['systems'].values() for r in d['reference_metrics']]
    summaries = [{'system': s.upper(), 'jobs': len(d['seeds']),
                  'chain_passes': sum(r['chain_screen'] == 'passes_operational_screen' for r in d['seeds']),
                  'point_passes': sum(r['all_screens_passed'] for r in d['points']),
                  'points': len(d['points'])} for s, d in data['systems'].items()]
    text = f'''# uMOF: Auswertung der zurückgekehrten Clusterrechnungen

Stand: {as_of}. Die Rückläufe liegen unter `reports/module_c_closure/cluster_bundle/results/`. Alle 18 MH- und 24 POLAR-Produktionsläufe sind vorhanden. Die frühere Aussage, dass diese Rechnungen noch ausstehen, ist damit überholt. Die lokale Auswertung vom 9. November hatte diesen Unterordner nicht berücksichtigt.

## Ergebnis

Die technische Ausführung ist vollständig, die wissenschaftliche Absicherung der Isothermen bleibt offen. Von 42 einzelnen Ketten bestehen nur {sum(r['chain_screen']=='passes_operational_screen' for r in seeds)} den vorab implementierten Stabilitätsscreen. Keiner der 14 Modell-Druckpunkte besteht gemeinsam alle Ketten- und Seedpräzisionsprüfungen. Die RMSE-Werte sind deshalb deskriptive Abweichungen von klassischen Referenzsimulationen, keine belastbare Genauigkeitsrangfolge.

{table(summaries, [('system','Profil'), ('jobs','Abgeschlossene Läufe'), ('chain_passes','Bestandene Kettenscreens'), ('point_passes','Vollständig bestandene Modell-Druckpunkte'), ('points','Geprüfte Modell-Druckpunkte')])}

![Neue Isothermen mit unabhängigen Seedintervallen](01_returned_isotherms.png)

## Was sich gegenüber der bisherigen Auswertung ändert

- Es liegen jetzt drei unabhängige MC-Seeds je Modell und Druck vor. „Nur ein Seed“ gilt für die historischen Piloten, nicht für diese Rückläufe. Trainingsseed-Streuung wurde dadurch nicht untersucht.
- Die drei Seeds und die längeren Ketten schließen die Sampling-Frage nicht: Große Streuung, Autokorrelation und/oder zu breite Blockintervalle bleiben bestehen.
- Im POLAR/ZIF-8-Rücklauf ist der UFF-Isothermen-RMSE der Foundation mit 0,617 mmol/g deskriptiv kleiner als der von Stage 1 mit 0,673 mmol/g. Der deutliche Stage-1-Vorteil des früheren kurzen Ein-Seed-Piloten wird hier somit nicht bestätigt. Wegen offener Sampling-Prüfungen folgt daraus keine gesicherte Umkehr der Rangfolge.
- Der POLAR-Clusterrücklauf verwendet **parent_stage1**, nicht `followup_epoch_1`. Die vier Modellgewichte im Clusterpaket stimmen mit den eingefrorenen Prüfsummen überein. Die separat festgestellte Prüfsummenabweichung der lokalen Folgemodelldatei betrifft diese Rückläufe nicht; sie bleibt für den historischen Folgelauf offen.
- Die Energie-/Kraft-Testkennzahlen aus dem bisherigen Fine-Tuning-Bericht werden durch diese Adsorptionsrechnungen nicht ersetzt. Die historischen und neuen Isothermen werden nicht zu einer gemeinsamen Stichprobe zusammengefasst.

## Quantitative Referenzabweichungen

Alle Fehler in mmol/g; berechnet aus den Mittelwerten über drei Seeds an den vorgesehenen Druckpunkten. CRAFTED UFF/DDEC und DREIDING/DDEC sind klassische Simulationsreferenzen, keine Experimente. MH/MOF-5 und POLAR/ZIF-8 erlauben keinen direkten Architekturvergleich.

{table(metrics, [('system','Profil'), ('model','Modell'), ('reference','Referenz'), ('mae','MAE'), ('rmse','RMSE'), ('bias','Bias')])}

Für MH steigt der deskriptive UFF-RMSE von 5,391 auf 6,537 mmol/g. Bei 10 bar liegen die Mittel bei 12,578 bzw. 20,798 mmol/g. Der Fine-Tuning-Punkt hat hier ein relativ enges Seedintervall, aber keine seiner drei Ketten besteht den Kettenscreen. Übereinstimmende Seedmittel allein sichern die Schätzung daher nicht ab.

## Samplingdiagnostik

{table(points, [('system','Profil'), ('model','Modell'), ('pressure_bar','p (bar)'), ('mean','Beladung (mmol/g)'), ('ci95_half_width','95-%-Halbbreite'), ('chains_passed','Bestandene Ketten / 3'), ('seed_precision_passed','Seedpräzision erfüllt')])}

![Fehlgeschlagene Kriterien pro Modell-Druckpunkt](02_sampling_failures.png)

Die Toleranzen werden unverändert aus dem vorhandenen Prüfverfahren übernommen. Ein Kettenscreen verlangt mindestens 20000 Produktionswerte, mindestens 20 geschätzte effektive Stichproben, ein hinreichend enges Blockmittelintervall aus mindestens acht Blöcken und eine begrenzte Änderung zwischen Produktionshälften. Für Intervall und Drift gilt max(0,02 mmol/g, 15 % des mittleren Betrags). Die unabhängige Seedprüfung verwendet max(0,02 mmol/g, 20 %). Das Bestehen wäre ein operationaler Stabilitätshinweis, kein Gleichgewichtsbeweis.

## Methodik und Datenprüfung

CO₂ bei 298 K; starre Gerüste. MH: MOF-5 bei 0,1/1/10 bar, float64. POLAR: ZIF-8 bei 0,1/0,5/1/5 bar, float32. Beide vergleichen die Foundation mit dem 25-%-Stage-1-Modell. Seeds: 32452843, 32452867 und 32452889. Pro Lauf: 20000 Equilibrierungs- und 40000 Produktionsschritte, zusammen 2,52 Millionen MC-Schritte. Technische Smoke-Tests zählen nicht als wissenschaftliche Wiederholungen.

Die archivierte Ausführung verwendet den expliziten MLIP-MC-Adapter mit einer Überlappungsgrenze von 0,8 mal der Summe der ASE-Kovalenzradien, modellspezifischer Vakuum-Gasreferenz, neutraler Singulett-Konditionierung und ohne externe Dispersionskorrektur. Insbesondere die MH-Überlappungsregel unterscheidet sich vom historischen Standardadapter. Unterschiede zwischen alten und neuen MH-Werten lassen sich daher nicht ausschließlich der Kettenlänge zuschreiben. Es erfolgte kein erneutes Training.

Für alle 42 Jobs wurden Bundle-/Konfigurationsidentität, Ergebnisprüfsummen und Abschluss-Checkpoints geprüft. Aus den binären Übergangslogs wurden alle MC-Zustände einschließlich verworfener Schritte rekonstruiert. Die 40000 Produktionswerte je Lauf ergeben die archivierten Mittelwerte und Kettenscreens erneut. Seedintervalle sowie Referenz-MAE/RMSE/Bias reproduzieren die mitgelieferten Berichte. Die Modellzuordnung für POLAR wurde zusätzlich mit dem ursprünglichen Auswahlbericht abgeglichen. Pro Profil ist genau eine archivierte Laufzeitumgebung enthalten; ihre Versionen stehen in `results.json`.

Die Übergangslogs werden bei dieser lokalen Prüfung neu mit SHA-256 erfasst. Die ursprünglichen Abschlussmarker enthalten einen Hash des Ergebnis-JSON, aber keinen ursprünglichen Transferhash der Übergangslogs. Die Rekonstruktion bestätigt deren Konsistenz mit den archivierten Kennzahlen; sie ist kein zusätzlicher Nachweis ihrer historischen Unverändertheit.

## Schlussfolgerung für die Arbeit

Die erweiterten uMOF-Adsorptionsrechnungen liefern keinen belastbaren Nachweis einer verbesserten Isothermenvorhersage durch das untersuchte Fine-Tuning. Sie zeigen zugleich, dass vollständige Rechnungen und mehrere Seeds die Prüfung der Kettendynamik nicht ersetzen. Die offene Aufgabe ist jetzt die gezielte Klärung der Sampling-Probleme und der Sensitivität gegenüber dem Adapterprotokoll, nicht das erstmalige Ausführen des bereits abgeschlossenen 42-Job-Pakets. Vor weiteren Produktionsläufen sollten insbesondere die MH-Zeitreihen und Akzeptanzstatistiken geprüft und belastbare Fortsetzungs-/Initialisierungsvergleiche geplant werden.

## Dateien und Reproduktion

- `seed_results.csv`: alle 42 Ketten einschließlich Einzelkriterien und Laufzeiten.
- `point_summary.csv`: 14 Modell-Druckpunkte mit Seedintervallen.
- `reference_metrics.csv`: Referenzabweichungen; `production_blocks.csv`: 20 zusammenhängende Produktionsblockmittel je Kette, keine unabhängigen Replikate.
- `results.json`: Konfigurationen, vollständige Screens und archivierte Softwareversionen.
- `provenance.json`: geprüfte Eingabedateien und Auswertungscode mit SHA-256.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mof-c-return-mpl \\
python -m analysis.module_c_return_review --as-of {as_of} --output /tmp/module_c_return_reproduced
```

Der Ausgabeordner muss neu sein. Die Auswertung führt keine Simulation oder Inferenz aus und verändert die zurückgekehrten Quelldaten nicht.
'''
    (output / 'REPORT.md').write_text(text)
    for name, rows in [('seed_results', seeds), ('point_summary', points), ('reference_metrics', metrics),
                       ('production_blocks', [r for d in data['systems'].values() for r in d['blocks']])]:
        write_csv(output / f'{name}.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', type=date.fromisoformat, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f'reports/module_c_cluster_return_{args.as_of}'
    if output.exists(): raise FileExistsError(output)
    sources = Sources(ROOT)
    data = collect_returns(sources)
    output.mkdir(parents=True)
    figures(data, output)
    write_report(data, output, str(args.as_of))
    dump(output / 'results.json', data)
    for path, expected in sources.files.items():
        if digest(ROOT / path) != expected: raise RuntimeError(f'Source changed during audit: {path}')
    scripts = [Path(__file__).relative_to(ROOT), Path('analysis/module_c_closure.py'),
               Path('engines/mlip_mc_runner.py'), Path('modules/module_c_mlips/closure_campaign.py'),
               Path('modules/module_c_mlips/umof_isotherm_stability.py')]
    dump(output / 'provenance.json', {'as_of': str(args.as_of), 'source_sha256': sources.files,
         'scripts_sha256': {str(p): digest(ROOT / p) for p in scripts},
         'source_files_unchanged': True, 'new_simulations': False, 'new_predictions': False})
    print(json.dumps({'output': str(output), 'jobs_reconstructed': sum(len(d['seeds']) for d in data['systems'].values()),
                      'all_source_reports_reproduced': True, 'source_files': len(sources.files)}, indent=2))


if __name__ == '__main__':
    main()
