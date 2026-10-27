"""Plot existing Module-C GCMC campaign results without running simulations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_crafted_reference(path, material, component, temperature):
    """Read the CRAFTED filename contract and convert Pa to bar; retain raw errors."""
    path = Path(path)
    prefix, forcefield, gas, reference_temperature = path.stem.rsplit('_', 3)
    charges, reference_material = prefix.split('_', 1)
    if (reference_material.casefold() != material.casefold() or gas != component
            or not np.isclose(float(reference_temperature), temperature, rtol=0, atol=.01)):
        raise ValueError('CRAFTED reference material, gas or temperature does not match campaign')
    values = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    if (values.shape[1] != 3 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(values[:, 0] <= 0) or np.any(values[:, 1:] < 0)):
        raise ValueError('Invalid CRAFTED pressure/loading/error columns')
    values = values[np.argsort(values[:, 0])]
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError('Reference pressures must be distinct')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'label': f'CRAFTED {forcefield}/{charges} (simulation)',
            'reference_class': 'classical_simulation', 'adsorption_basis': 'absolute',
            'material': reference_material, 'component': gas, 'temperature_K': float(reference_temperature),
            'doi': '10.1038/s41597-023-02116-z',
            'pressure_conversion': 'Pa / 100000 -> bar', 'loading_units': 'mol/kg = mmol/g',
            'points': [{'pressure_bar': float(p/1e5), 'loading_mol_per_kg': float(q),
                        'source_reported_error_mol_per_kg': float(e)} for p, q, e in values]}


def compare_reference(rows, reference):
    """Compare seed means at matching pressures; never extrapolate reference data."""
    points = reference['points']
    pressures = np.asarray([p['pressure_bar'] for p in points])
    loadings = np.asarray([p['loading_mol_per_kg'] for p in points])
    comparisons, metrics = [], []
    for row in rows:
        pressure = row['pressure_bar']
        if not pressures[0] <= pressure <= pressures[-1]:
            continue
        matches = np.flatnonzero(np.isclose(pressures, pressure, rtol=1e-9, atol=1e-12))
        matched = float(loadings[matches[0]]) if len(matches) else float(np.interp(pressure, pressures, loadings))
        difference = row['mean_loading_mol_per_kg'] - matched
        comparisons.append({**row, 'reference_loading_mol_per_kg': matched,
                            'reference_match': 'exact' if len(matches) else 'linear_interpolation_in_pressure',
                            'signed_deviation_mol_per_kg': difference,
                            'relative_deviation_percent': 100*difference/matched if matched else None})
    if not comparisons:
        raise ValueError('No pressure overlap with reference')
    for model in sorted({r['model'] for r in rows}):
        selected = [r for r in comparisons if r['model'] == model]
        if not selected:
            raise ValueError(f'No reference matches for {model}')
        errors = np.asarray([r['signed_deviation_mol_per_kg'] for r in selected])
        relative = [abs(r['relative_deviation_percent']) for r in selected if r['relative_deviation_percent'] is not None]
        metrics.append({'model': model, 'matched_pressure_count': len(selected),
                        'unmatched_pressure_count': sum(r['model'] == model for r in rows)-len(selected),
                        'mae_mol_per_kg': float(np.mean(np.abs(errors))),
                        'rmse_mol_per_kg': float(np.sqrt(np.mean(errors**2))),
                        'mean_signed_deviation_mol_per_kg': float(np.mean(errors)),
                        'mean_absolute_relative_deviation_percent': float(np.mean(relative)) if relative else None})
    return comparisons, sorted(metrics, key=lambda r: r['rmse_mol_per_kg'])


def create_report(run_directory, output_directory, stage='gcmc_pilot', reference_csv=None):
    root, output = Path(run_directory), Path(output_directory)
    groups, metadata, sources = defaultdict(list), set(), {}
    seen = set()
    for path in sorted(root.glob(f'*/results/{stage}.json')):
        result = json.loads(path.read_text())
        if result.get('status') != 'completed' or result.get('method', '').lower() != 'gcmc':
            raise ValueError(f'Expected a completed GCMC result: {path}')
        system = (result['material'], result['component'], float(result['temperature_K']))
        metadata.add(system)
        model, seed = result['model'], int(result['seed'])
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for point in result['pressure_points']:
            pressure, loading = float(point['pressure_bar']), float(point['loading_mol_per_kg'])
            if not np.isfinite([pressure, loading]).all() or pressure <= 0:
                raise ValueError(f'Invalid pressure/loading in {path}')
            identity = (model, seed, pressure)
            if identity in seen:
                raise ValueError(f'Duplicate model/seed/pressure: {identity}')
            seen.add(identity)
            groups[(model, pressure)].append({
                'model': model, 'seed': seed, 'pressure_bar': pressure,
                'loading_mol_per_kg': loading,
                'within_run_loading_std_mol_per_kg': point.get('loading_std_mol_per_kg'),
                'production_steps': point.get('production_steps'),
                'production_sample_count': point.get('production_sample_count'),
                'sampling_status': result.get('sampling_status', 'unassessed'),
            })
    if not groups or len(metadata) != 1:
        raise ValueError('Report requires results for exactly one material, component and temperature')
    material, component, temperature = next(iter(metadata))
    reference = load_crafted_reference(reference_csv, material, component, temperature) if reference_csv else None
    models = sorted({model for model, _ in groups})
    grids = [{pressure for model, pressure in groups if model == name} for name in models]
    if any(grid != grids[0] for grid in grids):
        raise ValueError('Models have different pressure grids; compare matched conditions')
    for model in models:
        seeds_by_pressure = [{r['seed'] for r in groups[(model, p)]} for p in sorted(grids[0])]
        if any(seeds != seeds_by_pressure[0] for seeds in seeds_by_pressure):
            raise ValueError(f'Incomplete seed/pressure grid for {model}')
    rows = []
    for (model, pressure), replicas in sorted(groups.items()):
        values = [r['loading_mol_per_kg'] for r in replicas]
        rows.append({'model': model, 'pressure_bar': pressure, 'replicates': len(values),
                     'mean_loading_mol_per_kg': float(np.mean(values)),
                     'between_seed_sd_mol_per_kg': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    output.mkdir(parents=True, exist_ok=True)
    def write_csv(name, records):
        with (output/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write_csv('isotherm_summary.csv', rows)
    write_csv('isotherm_seeds.csv', [r for group in groups.values() for r in group])
    comparisons, metrics = compare_reference(rows, reference) if reference else ([], [])
    if reference:
        write_csv('reference_points.csv', reference['points'])
        write_csv('reference_deviations.csv', comparisons)
        write_csv('reference_metrics.csv', metrics)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(zip(models, ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']))
    colors = {name: colors.get(name, f'C{i % 10}') for i, name in enumerate(models)}
    def save(fig, name):
        for extension in ('png', 'svg'):
            fig.savefig(output/f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    title = f'{material} / {component} at {temperature:g} K — {stage.replace("_", " ")}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for model in models:
        selected = [r for r in rows if r['model'] == model]
        x = [r['pressure_bar'] for r in selected]
        y = [r['mean_loading_mol_per_kg'] for r in selected]
        sd = [np.nan if r['between_seed_sd_mol_per_kg'] is None else r['between_seed_sd_mol_per_kg'] for r in selected]
        for ax in axes:
            ax.errorbar(x, y, yerr=sd, fmt='o-', capsize=3, color=colors[model], label=model, lw=1.7)
    axes[1].set_xscale('log')
    if reference:
        visible_reference = [p for p in reference['points'] if min(grids[0]) <= p['pressure_bar'] <= max(grids[0])]
        for ax in axes:
            ax.plot([p['pressure_bar'] for p in visible_reference],
                    [p['loading_mol_per_kg'] for p in visible_reference],
                    'D--', color='#222222', lw=2, ms=4, label=reference['label'], zorder=5)
    for ax, heading in zip(axes, ('Full pressure range', 'Logarithmic pressure axis')):
        ax.set(xlabel='Pressure (bar)', ylabel='Absolute loading (mol/kg = mmol/g)', title=heading)
        ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(title+'\nMean ± sample SD across seeds; convergence not established', fontsize=12)
    save(fig, 'isotherm_comparison')

    if reference:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for model in models:
            selected = [r for r in comparisons if r['model'] == model]
            x = [r['pressure_bar'] for r in selected]
            for ax, key in zip(axes, ('signed_deviation_mol_per_kg', 'relative_deviation_percent')):
                ax.plot(x, [r[key] for r in selected], 'o-', color=colors[model], label=model)
        for ax, ylabel in zip(axes, ('Model − reference (mol/kg)', '100 × (model − reference) / reference (%)')):
            ax.axhline(0, color='#222222', ls='--', lw=1)
            ax.set(xlabel='Pressure (bar)', ylabel=ylabel, xscale='log')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(title+'\nDeviations from '+reference['label'], fontsize=12)
        save(fig, 'reference_deviations')

    ncols = 2
    fig, axes = plt.subplots((len(models)+1)//2, ncols, figsize=(12, 3.6*((len(models)+1)//2)), squeeze=False, layout='constrained')
    for ax, model in zip(axes.flat, models):
        seeds = sorted({r['seed'] for (name, _), group in groups.items() if name == model for r in group})
        for seed in seeds:
            series = sorted((r for (name, _), group in groups.items() if name == model for r in group if r['seed'] == seed), key=lambda r: r['pressure_bar'])
            ax.plot([r['pressure_bar'] for r in series], [r['loading_mol_per_kg'] for r in series], 'o-', label=f'Seed {seed}', alpha=.8)
        ax.set(title=model, xlabel='Pressure (bar)', ylabel='Absolute loading (mol/kg)', xscale='log', ylim=(0, None))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle(title+'\nIndividual seeds; independent y-axis scales', fontsize=12)
    save(fig, 'isotherm_seed_comparison')
    report = {'material': material, 'component': component, 'temperature_K': temperature,
              'stage': stage, 'source_sha256': sources, 'summary': rows,
              'error_bars': 'Sample standard deviation of replicate means, not within-run fluctuations or a confidence interval',
              'convergence': 'Not established by this report; see source sampling_status',
              'reference_comparison': {'reference': reference, 'metrics': metrics,
                                       'matched_points': comparisons,
                                       'interpretation': 'Agreement with this classical simulation baseline, not experimental accuracy',
                                       'metric_weighting': 'Equal weight per matched pressure; errors of seed means; no extrapolation',
                                       'reference_uncertainty': 'Source error column archived; not plotted or used as statistical weights'} if reference else
                                      'No reference overlaid; model differences do not establish accuracy',
              'runtime_comparison': 'Omitted: recorded execution times may include reuse of completed simulations'}
    (output/'isotherm_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'README.md').write_text(
        '# Module C isotherm comparison\n\n'
        f'{material}/{component}, {temperature:g} K; {len(models)} models.\n\n'
        '`isotherm_comparison.png` and `.svg`: model means ± sample standard deviation across independent seed runs, '
        'with linear and logarithmic pressure axes. `isotherm_seed_comparison.png` and `.svg`: individual seed curves; '
        'each panel has its own loading scale. CSV files retain aggregate and per-seed observations.\n\n'
        'Within-run loading fluctuations are not used as error bars on a mean. Neither seed agreement nor completion '
        'proves equilibration or convergence. Runtime ranking is omitted because result-file runtimes may reflect reused runs.\n' +
        ('\nThe dashed black curve is '+reference['label']+' at matching conditions. '
         'It is a classical simulation baseline, not experimental ground truth. '
         'See https://www.nature.com/articles/s41597-023-02116-z . '
         'Reference pressures are converted from Pa to bar; loading is absolute in mol/kg. '
         'Metrics compare seed means with equal weight per matched pressure, use linear interpolation only where needed, '
         'and never extrapolate. Matching methods and SHA-256 provenance are archived. '
         'The source error column is retained but not plotted or treated as a confidence interval.\n'
         if reference else '\nNo reference is overlaid, so these figures do not rank model accuracy.\n'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', default='gcmc_pilot')
    parser.add_argument('--reference-csv', type=Path, help='CRAFTED Q_MOF_FF_GAS_T.csv; matching material/gas/temperature required')
    args = parser.parse_args()
    report = create_report(args.run_directory, args.output_dir, args.stage, args.reference_csv)
    print(f"Created plots and CSV/JSON for {len(report['summary'])} model/pressure groups in {args.output_dir}")


if __name__ == '__main__':
    main()
