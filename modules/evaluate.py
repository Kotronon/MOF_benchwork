"""Evaluate static CP2K force/energy predictions; this is not an isotherm test.

Run from the project root: python modules/evaluate.py --device cuda:0
Use --committee to additionally report the mean of all three fine-tuned models.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator

DEFAULT_ROOT = Path('outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/'
                    'zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning')


def get_configuration_type(candidate_id):
    for token, label in [('multiple_adsorbates', 'Multiple CO₂'),
                         ('single_adsorbate', 'Single CO₂'), ('framework', 'Framework')]:
        if token in candidate_id:
            return label
    return 'Unknown'


def error_row(atoms, energy, forces, model, kind):
    candidate_id = str(atoms.info['candidate_id'])
    n_framework = int(atoms.info['framework_atom_count'])
    if not 0 < n_framework <= len(atoms):
        raise ValueError(f'Invalid framework_atom_count: {candidate_id}')
    guest_symbols = atoms.get_chemical_symbols()[n_framework:]
    n_guest = len(guest_symbols)
    if n_guest % 3 or guest_symbols.count('C') != n_guest // 3 or guest_symbols.count('O') != 2 * n_guest // 3:
        raise ValueError(f'Expected physical CO2 guest atoms after framework: {candidate_id}')
    forces = np.asarray(forces, dtype=float)
    reference = np.asarray(atoms.arrays['REF_forces'], dtype=float)
    if forces.shape != (len(atoms), 3) or reference.shape != forces.shape:
        raise ValueError(f'Invalid force shape: {candidate_id}')
    if (not np.isfinite(forces).all() or not np.isfinite(reference).all()
            or not np.isfinite(energy) or not np.isfinite(float(atoms.info['REF_energy']))):
        raise ValueError(f'Non-finite predictions/references: {candidate_id}')
    norms = np.linalg.norm(forces - reference, axis=1)
    signed = float(energy) - float(atoms.info['REF_energy'])
    row = dict(model=model, model_kind=kind, candidate_id=candidate_id,
               configuration_type=get_configuration_type(candidate_id), n_atoms=len(atoms),
               n_framework=n_framework, n_guest=n_guest, n_co2=n_guest // 3,
               raw_total_energy_signed_error_ev=signed,
               raw_total_energy_absolute_error_ev=abs(signed),
               force_mae_ev_A=float(norms.mean()), force_error_sum=float(norms.sum()),
               maximum_force_error_ev_A=float(norms.max()))
    for key, errors in [('framework', norms[:n_framework]), ('co2', norms[n_framework:])]:
        row[f'{key}_force_error_sum'] = float(errors.sum())
        row[f'{key}_force_mae'] = float(errors.mean()) if len(errors) else np.nan
    return row


def summarize(df, groups):
    rows = []
    for key, group in df.groupby(groups, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row.update(configurations=len(group),
                   force_mae_ev_A=float(group.force_error_sum.sum() / group.n_atoms.sum()),
                   configuration_weighted_force_mae_ev_A=float(group.force_mae_ev_A.mean()),
                   raw_total_energy_mae_ev=float(group.raw_total_energy_absolute_error_ev.mean()),
                   raw_total_energy_bias_ev=float(group.raw_total_energy_signed_error_ev.mean()),
                   maximum_force_error_ev_A=float(group.maximum_force_error_ev_A.max()))
        for prefix, count in [('framework', 'n_framework'), ('co2', 'n_guest')]:
            denominator = group[count].sum()
            row[f'{prefix}_force_mae'] = (float(group[f'{prefix}_force_error_sum'].sum() / denominator)
                                         if denominator else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--iteration', type=int, default=4)
    parser.add_argument('--member', type=int, choices=range(3), default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--committee', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    import torch
    from ase.io import read
    torch.set_num_threads(args.threads)
    structures = read(args.root / f'iteration_{args.iteration:02d}/datasets/test.extxyz', index=':')
    ids = [str(a.info['candidate_id']) for a in structures]
    if len(set(ids)) != len(ids) or not structures:
        raise ValueError('Test configurations must be nonempty and unique')
    state = json.loads((args.root / 'state.json').read_text())
    if set(ids) != set(state['test_ids']):
        raise ValueError('Dataset does not match the held-out campaign test IDs')
    if set(ids) & (set(state['training_ids']) | set(state['validation_ids'])):
        raise ValueError('Test set overlaps training/validation IDs')
    specs = [('MACE-MP small', 'pretrained', {'model': 'small'})]
    members = range(3) if args.committee else [args.member]
    member_names = []
    for member in members:
        name = f'MACE CP2K fine-tuned (member {member}, seed {[12345,23456,34567][member]})'
        member_names.append(name)
        stem = f'mace_adsorption_committee_{member}'
        specs.append((name, 'finetuned_single', {'model': str(args.root / f'iteration_{args.iteration:02d}/training/{stem}/models/{stem}_compiled.model')}))
    specs.extend([
        ('uMOF-MH', 'pretrained', {'name':'umof_mh', 'model':'external/models/umof-mh.model', 'head':'r2scan+d4'}),
        ('uMOF-Polar', 'pretrained', {'name':'umof_polar', 'model':'external/models/umof-polar.model',
                                    'head':'r2scan+d4', 'model_type':'PolarMACE'}),
    ])
    rows, predictions = [], {}
    for name, kind, spec in specs:
        print(f'Evaluating {name}', flush=True)
        calculator = build_ase_calculator({'backend':'mace-torch', 'device':args.device,
                                          'default_dtype':'float64', 'dispersion':False, **spec})
        model_predictions = []
        for index, atoms in enumerate(structures):
            current = atoms.copy()
            current.calc = calculator
            energy, forces = float(current.get_potential_energy()), current.get_forces()
            rows.append(error_row(atoms, energy, forces, name, kind))
            if kind == 'finetuned_single': model_predictions.append((energy, forces.copy()))
            print(f'  {index+1}/{len(structures)}', flush=True)
        if model_predictions: predictions[name] = model_predictions
        del current, calculator
        gc.collect()
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    if args.committee:
        for index, atoms in enumerate(structures):
            energy = np.mean([predictions[n][index][0] for n in member_names])
            forces = np.mean([predictions[n][index][1] for n in member_names], axis=0)
            rows.append(error_row(atoms, energy, forces, 'MACE CP2K committee (3 members)', 'finetuned_committee'))
    df = pd.DataFrame(rows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / 'cp2k_model_comparison.csv', index=False)
    summary = summarize(df, ['model', 'model_kind'])
    summary.to_csv(output / 'cp2k_model_summary.csv', index=False)
    summarize(df, ['model', 'model_kind', 'configuration_type']).to_csv(output / 'cp2k_model_comparison_by_type.csv', index=False)
    metadata = {'reference':'CP2K PBE-D3(BJ)', 'umof_reference':'r2SCAN-D4',
                'force_metric':'mean atomic force-error vector norm; summary weighted by atom count',
                'energy_warning':'Raw total energies have different zeros; not adsorption-energy errors or a fair model ranking.',
                'scope':'Static test, not adsorption/isotherm validation. Boxplots show variation across configurations, not seed uncertainty.',
                'device':args.device, 'iteration':args.iteration, 'models':specs, 'test_ids':ids}
    (output / 'cp2k_evaluation_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
