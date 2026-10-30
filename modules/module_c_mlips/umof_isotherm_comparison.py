"""Run the frozen pretrained/25%-fine-tuned MOF-5 CO2 isotherm comparison."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.module_c_isotherm_report import create_report, load_crafted_reference
from modules.module_c_mlips.campaign import build_campaign_tasks, _run_campaign_task
from modules.module_c_mlips.umof_workflow import atomic_json
from pipeline.config import normalize_config
from pipeline.planning import build_run_plan

CONFIG = Path('input_json_files/benchmark_mof5_co2_umof_025pct_isotherm.json')
REFERENCE = Path('CRAFTED-2.0.0/ISOTHERM_FILES/DDEC_IRMOF-1_UFF_CO2_298.csv')
STAGE = 'gcmc_pilot'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def configure(smoke=False, quick=False, single_seed=False, ten_bar=False):
    config = json.loads(CONFIG.read_text())
    if ten_bar:
        if quick:
            raise ValueError('--ten-bar cannot be combined with --quick')
        campaign = config['benchmark']['potential_benchmark']['campaign']
        campaign['id'] += '_10bar'
        config['conditions']['pressures_bar'] = [10.]
        # Preserve the actual pressure-point seed from the three-pressure pilot.
        config['simulation']['seeds'] = [config['simulation']['seeds'][0] + 2]
        single_seed = True
    if quick:
        campaign = config['benchmark']['potential_benchmark']['campaign']
        campaign['id'] += '_quick'
        campaign['stages'][STAGE].update(equilibration_steps=1000, production_steps=2000)
        config['simulation'].update(initialization_cycles=1000, cycles=2000)
    if single_seed:
        campaign = config['benchmark']['potential_benchmark']['campaign']
        campaign['id'] += '_single_seed'
        config['simulation']['seeds'] = config['simulation']['seeds'][:1]
    if smoke:
        campaign = config['benchmark']['potential_benchmark']['campaign']
        campaign['id'] += '_technical_smoke'
        campaign['stages'][STAGE].update(equilibration_steps=2, production_steps=4)
        config['conditions']['pressures_bar'] = [10.] if ten_bar else [.1]
        config['simulation']['seeds'] = config['simulation']['seeds'][:1]
    return config


def preflight(config):
    normalized = normalize_config(config)
    campaign = config['benchmark']['potential_benchmark']['campaign']
    for model in campaign['models']:
        if digest(model['model']) != model['sha256']:
            raise ValueError(f"Model checksum mismatch: {model['name']}")
    ref = load_crafted_reference(REFERENCE, 'IRMOF-1', 'CO2', 298.)
    pressures = config['conditions']['pressures_bar']
    if min(pressures) < min(p['pressure_bar'] for p in ref['points']) or max(pressures) > max(p['pressure_bar'] for p in ref['points']):
        raise ValueError('Comparison pressures exceed reference range')
    tasks = build_campaign_tasks(normalized, stage=STAGE)
    plan = build_run_plan(tasks[0]['config'])
    if not plan['benchmark']['applicability']['can_attempt_simulation']:
        raise ValueError(plan['benchmark']['applicability']['reason'])
    forcefield = plan['resources']['forcefield']
    files = [plan['resources']['cif_path'], forcefield['files']['pseudo_atoms'],
             forcefield['adsorbates']['CO2'], str(REFERENCE)]
    provenance = {'config': config, 'input_sha256': {str(p): digest(p) for p in files}}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = Path(config['output']['directory'])
    run_root = output/'runs'/campaign['id']
    report_root = output/'campaigns'/campaign['id']
    return tasks, run_root, report_root, {'identity': identity, **provenance}


def summarize_pilot(report, output):
    """Assess a paired-seed trend; do not infer equilibrium or experimental accuracy."""
    import numpy as np
    from scipy.stats import t
    groups = {}
    for row in report['reference_comparison']['matched_points']:
        groups[(row['model'], row['pressure_bar'])] = row['reference_loading_mol_per_kg']
    seed_errors = {}
    for path in Path(report['run_directory']).glob(f'*/results/{STAGE}.json'):
        result = json.loads(path.read_text())
        errors = [(p['loading_mol_per_kg']-groups[(result['model'], p['pressure_bar'])])**2
                  for p in result['pressure_points']]
        seed_errors[(result['model'], result['seed'])] = float(np.sqrt(np.mean(errors)))
    names = ('umof_mh_pretrained', 'umof_mh_finetuned_025pct')
    seeds = sorted(seed for model, seed in seed_errors if model == names[0])
    if not seeds or set(seeds) != {seed for model, seed in seed_errors if model == names[1]}:
        raise ValueError('Pilot comparison requires matching sets of seeds')
    differences = np.array([seed_errors[(names[1], seed)]-seed_errors[(names[0], seed)] for seed in seeds])
    mean = float(differences.mean())
    half = float(t.ppf(.975, len(seeds)-1)*differences.std(ddof=1)/np.sqrt(len(seeds))) if len(seeds) >= 3 else None
    trend = 'descriptive_only_single_seed' if len(seeds) == 1 else 'inconclusive'
    if half is not None:
        if mean+half < 0:
            trend = 'lower_error_to_classical_reference'
        elif mean-half > 0:
            trend = 'higher_error_to_classical_reference'
    atomic_json(output/'pilot_assessment.json', {
        'trend': trend, 'paired_seed_rmse_difference_mol_per_kg': mean,
        'ci95': [mean-half, mean+half] if half is not None else None, 'seeds': seeds,
        'negative_difference_favors': names[1],
        'method': ('Student-t interval of paired per-seed isotherm RMSE differences; equal weight per pressure'
                   if half is not None else 'Descriptive RMSE difference only; insufficient seeds for an uncertainty estimate'),
        'reference': 'CRAFTED UFF/DDEC classical simulation, not experiment or DFT',
        'limitations': ['Pilot trend only; equilibration and convergence are not established.',
                       'Reference uncertainty and model systematic errors are not included.',
                       'The selected pressure points do not validate the full isotherm.']})
    print(f'Pilot trend against CRAFTED: {trend}', flush=True)


def write_ten_bar_history(run_root, output):
    """Plot loading histories and archive descriptive production-block means."""
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from engines.mlip_mc_runner import _read_gcmc_transition_log, _reconstruct_gcmc_samples
    reference = load_crafted_reference(REFERENCE, 'IRMOF-1', 'CO2', 298.)
    ref = next(p['loading_mol_per_kg'] for p in reference['points'] if p['pressure_bar'] == 10.)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    records, summary = [], []
    for path in sorted(Path(run_root).glob(f'*/results/{STAGE}.json')):
        result = json.loads(path.read_text())
        point = next(p for p in result['pressure_points'] if p['pressure_bar'] == 10.)
        equil = point['equilibration_steps']
        total = equil + point['production_steps']
        trace = _reconstruct_gcmc_samples(_read_gcmc_transition_log(Path(point['transition_log_path'])), total_steps=total)
        if len(trace) != total:
            raise ValueError('Incomplete GCMC history')
        loading = np.array([row['uptake'] for row in trace])*1000./result['framework_mass_amu']
        production = loading[equil:]
        cumulative = np.cumsum(production)/np.arange(1, len(production)+1)
        bin_size = max(1, min(500, total//20))
        bins = [loading[i:i+bin_size] for i in range(0, total, bin_size)]
        x = [i+(len(b)+1)/2 for i, b in zip(range(0, total, bin_size), bins)]
        axes[0].plot(x, [b.mean() for b in bins], label=result['model'])
        axes[1].plot(np.arange(equil+1, total+1), cumulative, label=result['model'])
        axes[0].axvline(equil, color='gray', linestyle=':', alpha=.5)
        for step, value in enumerate(loading, start=1):
            records.append({'model': result['model'], 'step': step,
                            'phase': 'equilibration' if step <= equil else 'production',
                            'loading_mol_per_kg': float(value)})
        blocks = np.array_split(production, 4)
        summary.append({'model': result['model'], 'seed': result['seed'],
                        'production_block_means_mol_per_kg': [float(b.mean()) for b in blocks],
                        'production_mean_mol_per_kg': float(production.mean()),
                        'production_first_half_mean_mol_per_kg': float(production[:len(production)//2].mean()),
                        'production_second_half_mean_mol_per_kg': float(production[len(production)//2:].mean())})
    for ax in axes:
        ax.axhline(ref, color='black', linestyle='--', label='CRAFTED UFF/DDEC reference')
        ax.set(xlabel='Monte Carlo step', ylabel='Absolute loading (mmol/g)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title('MOF-5 / CO2, 298 K, 10 bar: binned loading; dotted line ends equilibration')
    axes[1].set_title('Cumulative mean of production samples only')
    for extension in ('png', 'svg'):
        fig.savefig(output/f'10bar_history.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    with (output/'10bar_history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    atomic_json(output/'10bar_sampling_diagnostics.json', {
        'models': summary, 'interpretation': 'Descriptive history diagnostics only; no proof of equilibration or convergence.',
        'production_blocks': 'Four consecutive blocks; their means are correlated and are not independent seed replicates.'})


def parse_gpus(value):
    devices = [item.strip() for item in value.split(',')]
    if not devices or any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError('Use distinct physical GPU indices, e.g. 0,1,3')
    return devices


def run_tasks(tasks, run_root, report_root, *, smoke=False, quick=False, single_seed=False, ten_bar=False, gpus=None):
    """Assign one isolated model/seed process per GPU; reap children on errors."""
    devices = deque(gpus if gpus is not None else [None])
    pending = deque()
    for index, task in enumerate(tasks):
        result_path = run_root/task['task_id']/'results'/f'{STAGE}.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'completed':
            print(f'[{index+1}/{len(tasks)}] Reuse {task["task_id"]}', flush=True)
        else:
            pending.append((index, task))
    running = []
    try:
        while pending or running:
            while pending and devices:
                index, task = pending.popleft()
                device = devices.popleft()
                log_path = report_root/f'{task["task_id"]}.log'
                env = os.environ.copy()
                if device is not None:
                    env['CUDA_VISIBLE_DEVICES'] = device
                command = [sys.executable, '-m', 'modules.module_c_mlips.umof_isotherm_comparison', '--task', str(index)]
                if smoke:
                    command.append('--smoke')
                if quick:
                    command.append('--quick')
                if single_seed:
                    command.append('--single-seed')
                if ten_bar:
                    command.append('--ten-bar')
                print(f'[{index+1}/{len(tasks)}] Run {task["task_id"]} on GPU '
                      f'{env.get("CUDA_VISIBLE_DEVICES", "default")}; log: {log_path}', flush=True)
                log = log_path.open('a')
                try:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                running.append((process, log, device, log_path))
            for entry in list(running):
                process, log, device, log_path = entry
                code = process.poll()
                if code is None:
                    continue
                running.remove(entry)
                log.close()
                if code:
                    raise RuntimeError(f'Simulation failed; inspect {log_path}')
                devices.append(device)
            if running:
                time.sleep(.2)
    finally:
        for process, log, _, _ in running:
            if process.poll() is None:
                process.terminate()
        for process, log, _, _ in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate inputs without running simulations')
    parser.add_argument('--smoke', action='store_true', help='Separate six-move technical test; no scientific assessment')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Separate short pilot: 1000 equilibration + 2000 production moves per pressure')
    parser.add_argument('--single-seed', action='store_true', help='Only seed 104729; descriptive comparison without a seed confidence interval')
    parser.add_argument('--ten-bar', action='store_true', help='Only 10 bar and one seed: 10000 equilibration + 20000 production moves per model')
    parser.add_argument('--gpus', type=parse_gpus, help='Physical GPU indices; one simultaneous model/seed process per GPU, e.g. 0,1,3')
    parser.add_argument('--task', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.ten_bar and args.quick:
        parser.error('--ten-bar cannot be combined with --quick')
    config = configure(args.smoke, args.quick, args.single_seed, args.ten_bar)
    tasks, run_root, report_root, provenance = preflight(config)
    if args.check:
        print(f'Inputs OK: {len(tasks)} model/seed runs, pressures {config["conditions"]["pressures_bar"]} bar, T=298 K')
        print(f'GPUs={args.gpus or [os.environ.get("CUDA_VISIBLE_DEVICES", "not set")]}')
        settings = config['benchmark']['potential_benchmark']['campaign']['stages'][STAGE]
        count = len(tasks)*len(config['conditions']['pressures_bar'])*(settings['equilibration_steps']+settings['production_steps'])
        print(f'Total Monte Carlo moves: {count:,}')
        print(f'Reference: {REFERENCE} (classical simulation)')
        print(f'Report: {report_root}/comparison')
        return 0
    if args.task is not None:
        result = _run_campaign_task(tasks[args.task], submit=False, train=False, resume=True)
        if result.get('status') != 'completed':
            raise RuntimeError(f'Task failed: {result}')
        return 0
    report_root.mkdir(parents=True, exist_ok=True)
    # Serialize invocations and forbid reuse after inputs/model weights change.
    from modules.module_c_mlips.umof_workflow import campaign_lock
    with campaign_lock(report_root):
        manifest = report_root/'comparison_inputs.json'
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError('Inputs changed since this comparison started; use a new campaign ID')
        atomic_json(manifest, provenance)
        if not args.analyze_only:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this Python environment')
            stage_settings = config['benchmark']['potential_benchmark']['campaign']['stages'][STAGE]
            print(f"Starting {len(tasks)} model/seed runs at {config['conditions']['pressures_bar']} bar; "
                  f"{stage_settings['equilibration_steps']} equilibration + "
                  f"{stage_settings['production_steps']} production moves per pressure; "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
            run_tasks(tasks, run_root, report_root, smoke=args.smoke, quick=args.quick, single_seed=args.single_seed, ten_bar=args.ten_bar, gpus=args.gpus)
        missing = [task['task_id'] for task in tasks
                   if not (run_root/task['task_id']/'results'/f'{STAGE}.json').exists()]
        if missing:
            raise RuntimeError(f'Missing results: {missing}')
        output = report_root/'comparison'
        report = create_report(run_root, output, STAGE, REFERENCE)
        report['run_directory'] = str(run_root)
        if args.ten_bar:
            write_ten_bar_history(run_root, output)
        if not args.smoke:
            summarize_pilot(report, output)
            if args.quick:
                print('Short pilot only: inspect equilibration and seed spread before deciding on longer runs.', flush=True)
        print(f'Finished. Plots, reference errors and seed results: {output}', flush=True)
        if args.smoke:
            print('Technical smoke only; these six moves do not measure adsorption accuracy.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
