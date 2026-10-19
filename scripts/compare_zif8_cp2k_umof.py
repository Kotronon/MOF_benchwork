"""Compare pretrained and fine-tuned models on the frozen ZIF-8 CP2K test set."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator

ROOT = Path('outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import torch
    from ase.io import read
    torch.set_num_threads(args.threads)
    state = json.loads((ROOT / 'state.json').read_text())
    ids = state['test_ids']
    labels = {}
    for path in ROOT.glob('iteration_*/cp2k/*/label.json'):
        label = json.loads(path.read_text())
        if label['candidate_id'] in ids:
            labels[label['candidate_id']] = label
    if set(labels) != set(ids):
        raise ValueError('Incomplete fixed test set')
    out = ROOT / 'evaluation/umof_comparison'
    out.mkdir(parents=True, exist_ok=True)
    specs = [('foundation_mace', {'model': 'small'})]
    for member in range(3):
        name = f'mace_adsorption_committee_{member}'
        model = ROOT / f'iteration_04/training/{name}/models/{name}_compiled.model'
        specs.append((f'finetuned_{member}', {'model': str(model)}))
    for name, model in [('umof_mh','external/models/umof-mh.model'), ('umof_polar','external/models/umof-polar.model')]:
        specs.append((name, {'name': name, 'model_family': 'umof', 'model': model, 'head': 'r2scan+d4'}))
    all_predictions = {}
    errors = {}
    for name, spec in specs:
        destination = out / f'{name}_predictions.json'
        predictions = json.loads(destination.read_text()) if destination.exists() else {}
        print('MODEL', name, 'cached', len(predictions), flush=True)
        calculator = None
        try:
            for candidate_id in ids:
                if candidate_id in predictions:
                    continue
                if calculator is None:
                    calculator = build_ase_calculator({'backend':'mace-torch', 'device':args.device,
                                                       'default_dtype':'float64','dispersion':False, **spec})
                atoms = read(labels[candidate_id]['structure_path'])
                atoms.calc = calculator
                if args.device.startswith('cuda'): torch.cuda.synchronize()
                start = time.perf_counter()
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()
                if args.device.startswith('cuda'): torch.cuda.synchronize()
                predictions[candidate_id] = {'energy_ev':energy, 'forces_ev_per_A':forces.tolist(),
                                            'runtime_seconds':time.perf_counter()-start}
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(predictions))
                temporary.replace(destination)
                print(name,candidate_id,len(predictions),'/',len(ids),flush=True)
            all_predictions[name] = predictions
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors[name] = repr(exc)
        finally:
            atoms = None
            calculator = None
            gc.collect()
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
    members = [f'finetuned_{i}' for i in range(3)]
    if all(name in all_predictions for name in members):
        all_predictions['finetuned_committee'] = {
            candidate_id: {
                'energy_ev':float(np.mean([all_predictions[n][candidate_id]['energy_ev'] for n in members])),
                'forces_ev_per_A':np.mean([all_predictions[n][candidate_id]['forces_ev_per_A'] for n in members],axis=0).tolist(),
                'runtime_seconds':sum(all_predictions[n][candidate_id]['runtime_seconds'] for n in members),
            } for candidate_id in ids
        }
    rows = []
    per_config = []
    for name,predictions in all_predictions.items():
        energy_errors, force_errors, runtimes = [], [], []
        for candidate_id in ids:
            pred, ref = predictions[candidate_id], labels[candidate_id]
            signed = pred['energy_ev']-ref['energy_ev']
            norms = np.linalg.norm(np.asarray(pred['forces_ev_per_A'])-np.asarray(ref['forces_ev_per_A']),axis=1)
            energy_errors.append(signed);force_errors.extend(norms);runtimes.append(pred['runtime_seconds'])
            per_config.append({'model':name,'candidate_id':candidate_id,'energy_error_ev':signed,
                               'force_vector_mae_ev_per_A':float(norms.mean()),'maximum_force_error_ev_per_A':float(norms.max())})
        rows.append({'model':name,'test_count':len(ids),'energy_mae_ev':float(np.mean(np.abs(energy_errors))),
                     'energy_bias_ev':float(np.mean(energy_errors)),
                     'force_vector_mae_ev_per_A':float(np.mean(force_errors)),
                     'maximum_force_error_ev_per_A':float(np.max(force_errors)),
                     'mean_runtime_seconds':float(np.mean(runtimes)),
                     'median_runtime_seconds':float(np.median(runtimes))})
    provenance = []
    for name,spec in specs:
        model = Path(spec['model'])
        provenance.append({'name':name,'spec':spec,'sha256':hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None})
    report = {'status':'completed' if not errors else 'partial','device':args.device,'threads':args.threads,
              'reference':'CP2K PBE-D3(BJ), cutoff 1200 Ry','umof_level':'r2SCAN-D4',
              'test_ids':ids,'models':provenance,'metrics':rows,'per_configuration':per_config,'errors':errors,
              'caveats':['Total energy zeros/reference methods differ; raw total-energy MAE is not an adsorption-energy ranking.',
                         'Runtime includes first evaluation warmup; model loading excluded.',
                         'Fine-tuned model selection used this test set in prior reporting; confirm on fresh data.']}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    import csv
    if rows:
        with (out/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps({'metrics':rows,'errors':errors},indent=2),flush=True)

if __name__ == '__main__': main()
