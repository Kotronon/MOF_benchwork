"""Compare fixed-geometry MOF-5/CO2 energies without further MC or training."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from ase.io import read, write
from ase.geometry import get_distances

from .umof_isotherm_comparison import configure, preflight, digest, STAGE
from .umof_workflow import atomic_json, campaign_lock
from .datasets.common import load_host_guest_system
from .potential_backends.calculators import build_ase_calculator
from pipeline.planning import build_run_plan

OUTPUT = Path('outputs/module_C_umof_energy_diagnostic_025pct_mof5')
EV_TO_KJ_MOL = 96.48533212331002


def interaction_energy(combined, host, gas):
    return combined - host - gas


def prepare(output):
    config = configure(ten_bar=True)
    tasks, run_root, _, provenance = preflight(config)
    plan = build_run_plan(tasks[0]['config'])
    ff = plan['resources']['forcefield']
    host, molecule, _ = load_host_guest_system(
        plan['resources']['cif_path'], ff['adsorbates']['CO2'], ff['files']['pseudo_atoms'],
        cell_representation=plan['simulation']['cell_representation'],
        unit_cells=plan['simulation']['unit_cells'], cutoff_A=plan['simulation']['cutoff_A'],
        minimum_image_policy=plan['simulation']['minimum_image_policy'])
    molecule = molecule[np.asarray(molecule.get_masses()) >= .01]
    if molecule.get_chemical_formula() != 'CO2':
        raise ValueError('Expected a physical CO2 molecule without virtual sites')
    gas = molecule.copy()
    gas.positions -= gas.get_center_of_mass()
    gas.translate(host.cell.cartesian_positions([.5, .5, .5]))
    gas.set_cell(host.cell); gas.set_pbc(True)
    rng = np.random.default_rng(20261029)
    combined, guests, samples = [], [], []
    for task in tasks:
        root = run_root/task['task_id']/'engine/mlip_mc'/STAGE/'10bar'/'checkpoints_10.0000bar'
        for step in range(12000, 30001, 2000):
            path = root/f'checkpoint_{step}'/'traj.xyz'
            atoms = read(path)
            if len(atoms) <= len(host) or (len(atoms)-len(host)) % len(gas):
                raise ValueError(f'Invalid host/molecule partition in {path}')
            if (not np.array_equal(atoms.numbers[:len(host)], host.numbers)
                    or not np.allclose(atoms.positions[:len(host)], host.positions, rtol=0, atol=1e-7)
                    or not np.allclose(atoms.cell, host.cell, rtol=0, atol=1e-7)):
                raise ValueError(f'Framework differs from comparison structure in {path}')
            count = (len(atoms)-len(host))//len(gas)
            index = int(rng.integers(count))
            guest = atoms[len(host)+index*len(gas):len(host)+(index+1)*len(gas)]
            if not np.array_equal(guest.numbers, gas.numbers):
                raise ValueError(f'Guest atom ordering differs in {path}')
            pair = host + guest
            pair.set_cell(host.cell); pair.set_pbc(True)
            _, distances = get_distances(guest.positions, host.positions, cell=host.cell, pbc=True)
            sample = {'sample_id': len(samples), 'source_model': task['model'],
                      'source_step': step, 'molecule_index': index,
                      'source_path': str(path), 'source_sha256': digest(path),
                      'minimum_host_guest_distance_A': float(distances.min())}
            pair.info.update(sample)
            guest.info.update(sample)
            combined.append(pair); guests.append(guest); samples.append(sample)
    models = config['benchmark']['potential_benchmark']['campaign']['models']
    manifest = {'source_protocol': provenance, 'samples': samples, 'models': models,
                'selection': 'One random molecule from each of 10 production checkpoints per model; balanced diagnostic sample, not an unbiased adsorption ensemble',
                'definition': 'E_ads = E_host+guest - E_host - E_guest; identical fixed geometries and model-specific heads for every term',
                'scope': 'Single-guest interaction only; guest-guest and finite-loading effects not included'}
    archived = output/'inputs.json'
    if archived.exists() and json.loads(archived.read_text()) != manifest:
        raise ValueError('Diagnostic inputs changed; use a new output directory')
    write(output/'host.extxyz', host, columns=['symbols', 'positions'])
    write(output/'gas_reference.extxyz', gas, columns=['symbols', 'positions'])
    write(output/'combined.extxyz', combined, columns=['symbols', 'positions'])
    write(output/'guests.extxyz', guests, columns=['symbols', 'positions'])
    atomic_json(archived, manifest)
    return manifest


def energy(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    result = float(atoms.get_potential_energy())
    if not np.isfinite(result):
        raise ValueError('Nonfinite model energy')
    return result


def atomic_reference(calc, specification, atoms):
    model = calc.models[0]
    head = list(model.heads).index(specification['head'])
    values = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    values = values[head] if values.ndim == 2 else values
    numbers = model.atomic_numbers.detach().cpu().numpy().tolist()
    mapping = {int(z): float(v) for z, v in zip(numbers, values)}
    return float(sum(mapping[int(z)] for z in atoms.numbers))


def worker(output, index):
    manifest = json.loads((output/'inputs.json').read_text())
    spec = manifest['models'][index]
    if digest(spec['model']) != spec['sha256']:
        raise ValueError('Model checksum mismatch')
    calc = build_ase_calculator(spec)
    host = read(output/'host.extxyz')
    gas = read(output/'gas_reference.extxyz')
    host_e, gas_e = energy(calc, host), energy(calc, gas)
    rows = []
    pairs, guests = read(output/'combined.extxyz', ':'), read(output/'guests.extxyz', ':')
    for sample, pair, guest in zip(manifest['samples'], pairs, guests):
        total_e, guest_e = energy(calc, pair), energy(calc, guest)
        rows.append({**sample, 'total_energy_ev': total_e, 'host_energy_ev': host_e,
                     'gas_energy_ev': guest_e,
                     'interaction_energy_ev': interaction_energy(total_e, host_e, guest_e)})
        print(f'{spec["name"]}: pose {sample["sample_id"]+1}/{len(pairs)}', flush=True)
    # Numerical disconnected-fragment control, not a physical MOF adsorption geometry.
    cluster_host = host.copy(); cluster_host.set_pbc(False)
    cluster_host.set_cell([200., 200., 200.])
    distant_gas = gas.copy(); distant_gas.set_pbc(False); distant_gas.set_cell(cluster_host.cell)
    distant_gas.positions -= distant_gas.get_center_of_mass()
    distant_gas.translate(cluster_host.positions.max(axis=0) + 30.)
    residual = interaction_energy(energy(calc, cluster_host+distant_gas),
                                  energy(calc, cluster_host), energy(calc, distant_gas))
    result = {'model': spec, 'rows': rows, 'host_energy_ev': host_e,
              'gas_reference_energy_ev': gas_e,
              'host_atomic_reference_sum_ev': atomic_reference(calc, spec, host),
              'gas_atomic_reference_sum_ev': atomic_reference(calc, spec, gas),
              'disconnected_fragment_additivity_residual_ev': residual,
              'isolated_guest_energy_range_ev': float(np.ptp([r['gas_energy_ev'] for r in rows])),
              'isolated_guest_reference_max_deviation_ev': float(max(abs(r['gas_energy_ev']-gas_e) for r in rows))}
    atomic_json(output/f'model_{index}.json', result)


def report(output):
    manifest = json.loads((output/'inputs.json').read_text())
    base, fine = [json.loads((output/f'model_{i}.json').read_text()) for i in range(2)]
    rows = []
    for a, b in zip(base['rows'], fine['rows']):
        if a['sample_id'] != b['sample_id'] or a['source_sha256'] != b['source_sha256']:
            raise ValueError('Unpaired geometries')
        change = b['interaction_energy_ev'] - a['interaction_energy_ev']
        rows.append({'sample_id': a['sample_id'], 'source_model': a['source_model'],
                     'source_step': a['source_step'], 'minimum_host_guest_distance_A': a['minimum_host_guest_distance_A'],
                     'pretrained_adsorption_energy_ev': a['interaction_energy_ev'],
                     'finetuned_adsorption_energy_ev': b['interaction_energy_ev'],
                     'pretrained_adsorption_energy_kj_mol': a['interaction_energy_ev']*EV_TO_KJ_MOL,
                     'finetuned_adsorption_energy_kj_mol': b['interaction_energy_ev']*EV_TO_KJ_MOL,
                     'finetuned_minus_pretrained_ev': change,
                     'finetuned_minus_pretrained_kj_mol': change*EV_TO_KJ_MOL,
                     'delta_total_energy_ev': b['total_energy_ev']-a['total_energy_ev'],
                     'delta_host_energy_ev': b['host_energy_ev']-a['host_energy_ev'],
                     'delta_gas_energy_ev': b['gas_energy_ev']-a['gas_energy_ev']})
    changes = np.array([r['finetuned_minus_pretrained_ev'] for r in rows])
    tolerance = 1e-6  # eV; exclude numerical cancellation noise.
    summary = {'sample_count': len(rows), 'fine_more_attractive_count': int((changes < -tolerance).sum()),
               'fine_less_attractive_count': int((changes > tolerance).sum()),
               'unchanged_count': int((np.abs(changes) <= tolerance).sum()),
               'classification_tolerance_ev': tolerance,
               'delta_adsorption_energy_ev': {'mean': float(changes.mean()), 'median': float(np.median(changes)),
                    'min': float(changes.min()), 'max': float(changes.max()), 'std_across_selected_poses': float(changes.std())},
               'mean_delta_guest_addition_energy_before_gas_subtraction_ev': float(np.mean([r['delta_total_energy_ev']-r['delta_host_energy_ev'] for r in rows])),
               'delta_host_energy_ev': fine['host_energy_ev']-base['host_energy_ev'],
               'delta_gas_reference_energy_ev': fine['gas_reference_energy_ev']-base['gas_reference_energy_ev'],
               'delta_host_atomic_reference_sum_ev': fine['host_atomic_reference_sum_ev']-base['host_atomic_reference_sum_ev'],
               'delta_gas_atomic_reference_sum_ev': fine['gas_atomic_reference_sum_ev']-base['gas_atomic_reference_sum_ev'],
               'model_checks': [{k:v for k,v in m.items() if k != 'rows'} for m in (base,fine)],
               'limitations': ['No DFT or experimental accuracy ranking from model-to-model differences.',
                              'Raw isolated-system energy differences alone do not establish incorrect energy references.',
                              'Balanced but correlated, model-selected single-guest poses; no Henry coefficient or isotherm estimator.',
                              'Additional loading-dependent guest-guest effects are outside this diagnostic.']}
    for r in rows:
        if not np.isclose(r['finetuned_minus_pretrained_ev'], r['delta_total_energy_ev']-r['delta_host_energy_ev']-r['delta_gas_energy_ev'], atol=1e-9):
            raise ValueError('Energy subtraction inconsistency')
    with (output/'paired_energies.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(output/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    x = np.arange(1, len(rows)+1)
    axes[0].plot(x, [r['pretrained_adsorption_energy_kj_mol'] for r in rows], 'o-', label='Pretrained UMOF-MH')
    axes[0].plot(x, [r['finetuned_adsorption_energy_kj_mol'] for r in rows], 's-', label='25% fine-tuning')
    axes[0].set(ylabel='Single-guest interaction energy (kJ/mol)', title='MOF-5 / CO2: identical fixed geometries; model comparison, no DFT reference')
    axes[0].legend()
    axes[1].bar(x, changes*EV_TO_KJ_MOL, color=['#0072B2' if d < 0 else '#D55E00' for d in changes])
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(xlabel='Pose (1-10 from pretrained MC; 11-20 from fine-tuned MC)',
                ylabel='Fine-tuned minus pretrained (kJ/mol)', title='Negative: fine-tuning predicts stronger binding at this geometry')
    for ax in axes:
        ax.grid(alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(output/f'adsorption_energies.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    text = (f'# MOF-5 / CO2 energy diagnostic\n\nCompared {len(rows)} identical single-guest poses. '
            f'Fine-tuning is more attractive at {summary["fine_more_attractive_count"]} poses, '
            f'unchanged at {summary["unchanged_count"]}, less attractive at {summary["fine_less_attractive_count"]} '
            f'(tolerance {tolerance:g} eV). '
            f'Mean change: {changes.mean()*EV_TO_KJ_MOL:.3f} kJ/mol; '
            f'range: {changes.min()*EV_TO_KJ_MOL:.3f} to {changes.max()*EV_TO_KJ_MOL:.3f} kJ/mol.\n\n'
            'See summary.json for separate host, gas, atomic-reference and additivity checks; '
            'paired_energies.csv for every subtraction term; model_0.json and model_1.json for raw model outputs. '
            'Geometries are archived in extxyz files. inputs.json records model and snapshot checksums.\n\n'
            + '\n'.join('- '+item for item in summary['limitations'])+'\n')
    (output/'REPORT.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k != 'model_checks'}, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--worker', type=int, choices=(0,1))
    parser.add_argument('--gpus', default='0,3')
    args = parser.parse_args(argv)
    output = args.output_dir
    if args.worker is not None:
        worker(output, args.worker)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with campaign_lock(output):
        prepare(output)
        devices = args.gpus.split(',')
        if len(devices) != 2 or any(not d.isdigit() for d in devices) or devices[0] == devices[1]:
            raise ValueError('Choose two distinct physical GPU indices')
        running = []
        try:
            for index, gpu in enumerate(devices):
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                    env.setdefault(key, '1')
                log = (output/f'model_{index}.log').open('w')
                command = [sys.executable, '-m', 'modules.module_c_mlips.umof_energy_diagnostic', '--worker', str(index), '--output-dir', str(output)]
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append((process, log))
            for index, (process, log) in enumerate(running):
                if process.wait():
                    raise RuntimeError(f'Diagnostic failed; inspect {output}/model_{index}.log')
        finally:
            for process, log in running:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                log.close()
        report(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
