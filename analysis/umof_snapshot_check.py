"""Prepare and run a portable, bounded energy check of returned uMOF states.

Preparation is CPU-only geometry/log analysis. Evaluation performs single points,
never MD/GCMC, training, downloads or scheduler submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def final_transition(path):
    fmt = 'iiddi'
    size = struct.calcsize(fmt)
    if Path(path).stat().st_size % size:
        raise ValueError('Truncated transition log')
    with Path(path).open('rb') as stream:
        stream.seek(-size, 2)
        step, n, interaction, total, atoms = struct.unpack(fmt, stream.read(size))
    return {'step': step, 'guest_count': n, 'interaction_eV': interaction, 'total_eV': total, 'atom_count': atoms}


def geometry(atoms, host, guest):
    nh, ng = len(host), len(guest)
    if (len(atoms) - nh) % ng or not np.array_equal(atoms.numbers[:nh], host.numbers):
        raise ValueError('Snapshot atom order differs from host/guest convention')
    if not np.allclose(atoms.positions[:nh], host.positions, atol=2e-7, rtol=0):
        raise ValueError('Framework moved or changed order')
    if not np.allclose(atoms.cell, host.cell, atol=1e-9, rtol=0):
        raise ValueError('Cell mismatch')
    n = (len(atoms) - nh) // ng
    if not np.array_equal(atoms.numbers[nh:], np.tile(guest.numbers, n)):
        raise ValueError('Guest atom order differs from rigid molecule')
    for i in range(n):
        molecule = atoms[nh+i*ng:nh+(i+1)*ng]
        if not np.allclose(molecule.get_all_distances(mic=True), guest.get_all_distances(), atol=2e-6, rtol=0):
            raise ValueError('Nonrigid guest in snapshot')
    if not n:
        return {'guest_count': 0, 'minimum_contact_A': None, 'minimum_contact_covalent_ratio': None,
                'contact_pair': None, 'contact_kind': None, 'remove_guest_index': None}
    groups = np.r_[np.zeros(nh, int), np.repeat(np.arange(n)+1, ng)]
    distances = atoms.get_all_distances(mic=True)
    mask = (groups[:, None] != groups[None, :])
    radii = covalent_radii[atoms.numbers]
    ratios = np.where(mask, distances / (radii[:, None] + radii[None, :]), np.inf)
    i, j = np.unravel_index(np.argmin(ratios), ratios.shape)
    guest_group = int(groups[i] if groups[i] else groups[j])
    return {'guest_count': n, 'minimum_contact_A': float(distances[i, j]),
            'minimum_contact_covalent_ratio': float(ratios[i, j]),
            'contact_pair': atoms[i].symbol + '-' + atoms[j].symbol,
            'contact_kind': 'host-guest' if min(groups[i], groups[j]) == 0 else 'guest-guest',
            'remove_guest_index': guest_group - 1}


def verify_bundle(bundle):
    manifest = json.loads((bundle / 'bundle_manifest.json').read_text())
    for path, expected in manifest['file_sha256'].items():
        if sha(bundle / path) != expected:
            raise ValueError(f'Bundle input changed: {path}')


def prepare(bundle, output):
    verify_bundle(bundle)
    rows, sources, configs = [], {}, {}
    def track(path):
        sources[str(path.relative_to(bundle))] = sha(path)
        return path
    track(bundle / 'bundle_manifest.json')
    for profile in ('mh', 'polar'):
        config_path = track(bundle / f'{profile}.json')
        config = json.loads(config_path.read_text()); configs[profile] = config
        host = read(track(bundle / config['framework']['path']))
        guest = read(track(bundle / config['adsorbate']['path']))
        for pressure in config['pressures_bar']:
            for seed in config['seeds']:
                for model in config['models']:
                    job = f'{model["label"]}_p{pressure:g}_s{seed}'
                    folder = bundle / 'results' / profile / job
                    marker = json.loads(track(folder / 'completed.json').read_text())
                    if (marker['identity']['bundle_sha256'] != sha(bundle/'bundle_manifest.json')
                            or marker['identity']['config_sha256'] != sha(config_path)):
                        raise ValueError('Completed job identity differs from frozen inputs')
                    path = track(folder / marker['result_path'])
                    if sha(path) != marker['result_sha256']: raise ValueError('Result hash mismatch')
                    result = json.loads(path.read_text()); point = result['pressure_points'][0]
                    snapshots = list(path.parent.glob('**/restart/restart_*.xyz'))
                    traces = list(path.parent.rglob(Path(point['transition_log_path']).name))
                    if len(snapshots) != 1 or len(traces) != 1: raise ValueError('Ambiguous snapshot/log')
                    snapshot = track(snapshots[0]); trace = track(traces[0])
                    atoms = read(snapshot); state = geometry(atoms, host, guest); final = final_transition(trace)
                    if final['guest_count'] != state['guest_count'] or final['atom_count'] != len(atoms):
                        raise ValueError('Final trace and snapshot particle counts differ')
                    checkpoints = list(path.parent.glob('**/checkpoint_60000/traj.xyz'))
                    if len(checkpoints) != 1: raise ValueError('Expected final 60000-step snapshot')
                    checkpoint = read(track(checkpoints[0]))
                    if not np.array_equal(checkpoint.numbers, atoms.numbers) or not np.allclose(checkpoint.positions, atoms.positions, atol=2e-7, rtol=0):
                        raise ValueError('Restart snapshot differs from final checkpoint')
                    row = {'profile': profile, 'model': model['label'], 'job': job, 'pressure_bar': pressure, 'seed': seed,
                           **state, 'logged_final_interaction_eV': final['interaction_eV'],
                           'logged_final_total_eV': final['total_eV'], 'logged_gas_eV': result['gas_reference_eV'],
                           'final_interaction_per_guest_eV': final['interaction_eV']/state['guest_count'] if state['guest_count'] else None,
                           'mean_interaction_eV': point['mean_interaction_energy_ev'],
                           'runtime_hours': result['runtime_seconds']/3600,
                           'snapshot_source': str(snapshot.relative_to(bundle)), 'snapshot_sha256': sha(snapshot),
                           'runtime_versions': marker['identity']['runtime_versions']}
                    for move, counts in point['move_statistics'].items():
                        row[f'{move}_acceptance_percent'] = 100*counts['accepted']/counts['attempted'] if counts['attempted'] else None
                    rows.append(row)
    # Three deterministic examples per frozen model: most negative per guest,
    # median, and least negative nonempty endpoint. This is diagnostic selection.
    selected = []
    for profile in ('mh', 'polar'):
        for model in ('pretrained', 'finetuned'):
            candidates = sorted([r for r in rows if r['profile']==profile and r['model']==model and r['guest_count']],
                                key=lambda r: (r['final_interaction_per_guest_eV'], r['job']))
            for why, index in (('most_negative', 0), ('median', len(candidates)//2), ('least_negative', -1)):
                selected.append({**candidates[index], 'selection': why})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'snapshots').mkdir()
    for i, row in enumerate(selected):
        row['case'] = i; row['snapshot'] = f'snapshots/{i:02d}.xyz'
        shutil.copy2(bundle / row['snapshot_source'], output / row['snapshot'])
    shutil.copy2(Path(__file__), output / 'umof_snapshot_check.py')
    write_csv(output/'all_endpoints.csv', [{k:v for k,v in r.items() if k!='runtime_versions'} for r in rows])
    dump(output/'manifest.json', {'bundle_sha256': sha(bundle/'bundle_manifest.json'), 'configs': configs,
         'cases': selected, 'source_sha256': sources, 'script_sha256': sha(output/'umof_snapshot_check.py'),
         'selection_scope': '12 deliberately selected nonempty endpoints, not an unbiased adsorption benchmark'})
    print(json.dumps({'packet':str(output),'endpoints':len(rows),'single_point_cases':len(selected)},indent=2))


def check(bundle, packet):
    verify_bundle(bundle)
    manifest = json.loads((packet/'manifest.json').read_text())
    if sha(bundle/'bundle_manifest.json') != manifest['bundle_sha256']:
        raise ValueError('Wrong original cluster bundle')
    if sha(packet/'umof_snapshot_check.py') != manifest['script_sha256']:
        raise ValueError('Diagnostic script changed')
    for profile in manifest['configs']:
        if json.loads((bundle/f'{profile}.json').read_text()) != manifest['configs'][profile]:
            raise ValueError('Configuration changed')
    for row in manifest['cases']:
        if sha(packet/row['snapshot']) != row['snapshot_sha256']:
            raise ValueError('Diagnostic snapshot changed')
    return manifest


def evaluate(bundle, packet, output, profile, device, limit):
    manifest = check(bundle, packet)
    cases = [r for r in manifest['cases'] if profile is None or r['profile']==profile]
    if limit is not None: cases = cases[:limit]
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(bundle))
    from modules.module_c_mlips.umof_profiles import calculator, polar_conditioning
    import torch
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    versions = {name: importlib.metadata.version(name) for name in ('ase','numpy','torch','mace-torch','mlip-mc')}
    versions.update(device=device, cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    dump(output/'run_identity.json', {'manifest_sha256':sha(packet/'manifest.json'),'versions':versions,
         'case_ids':[r['case'] for r in cases], 'purpose':'single-point audit, no sampling'})
    results = []
    for key in dict.fromkeys((r['profile'],r['model']) for r in cases):
        cfg = manifest['configs'][key[0]]
        spec = next(r for r in cfg['models'] if r['label']==key[1])
        base = calculator(bundle/spec['path'], model=key[0], head=spec['head'], device=device, dtype=cfg['dtype'])
        if base.head != spec['head']: raise ValueError('Unexpected calculator head')
        def energy(atoms):
            clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            polar_conditioning(clean); clean.calc = base
            value = float(clean.get_potential_energy())
            if not np.isfinite(value): raise ValueError('Nonfinite energy')
            return value
        host = read(bundle/cfg['framework']['path']); guest = read(bundle/cfg['adsorbate']['path'])
        gas = guest.copy(); gas.set_cell([30,30,30]); gas.set_pbc(False); gas.center()
        eh, eg = energy(host), energy(gas)
        for row in [r for r in cases if (r['profile'],r['model']) == key]:
            atoms = read(packet/row['snapshot']); state = geometry(atoms,host,guest)
            et = energy(atoms); interaction = et-eh-state['guest_count']*eg
            start = len(host)+len(guest)*row['remove_guest_index']
            reduced = atoms.copy(); del reduced[start:start+len(guest)]
            removal = et-energy(reduced)-eg
            record = {'case':row['case'], 'profile':key[0], 'model':key[1], 'job':row['job'],
                      'selection':row['selection'], **state, 'fresh_total_eV':et, 'fresh_host_eV':eh,
                      'fresh_gas_eV':eg, 'fresh_interaction_eV':interaction,
                      'logged_interaction_eV':row['logged_final_interaction_eV'],
                      'interaction_difference_eV':interaction-row['logged_final_interaction_eV'],
                      'total_difference_eV':et-row['logged_final_total_eV'],
                      'gas_difference_eV':eg-row['logged_gas_eV'],
                      'contact_guest_binding_eV':removal}
            results.append(record); dump(output/f'case_{row["case"]:02d}.json',record)
            print(json.dumps(record),flush=True)
        del base
        if device=='cuda': torch.cuda.empty_cache()
    write_csv(output/'single_points.csv',results)
    dump(output/'completed.json', {'cases':len(results),'manifest_sha256':sha(packet/'manifest.json'),
                                  'result_sha256':sha(output/'single_points.csv'), 'no_simulations':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','check','evaluate'])
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--packet',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=['mh','polar'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--limit',type=int)
    args=p.parse_args(); bundle=args.bundle.resolve(); packet=args.packet.resolve()
    if args.limit is not None and args.limit<=0: p.error('--limit must be positive')
    if args.action=='check':
        m=check(bundle,packet);print(f'Checked original bundle and {len(m["cases"])} diagnostic snapshots.');return
    if args.output is None:p.error('--output required')
    if args.action=='prepare':prepare(bundle,args.output.resolve())
    else:evaluate(bundle,packet,args.output.resolve(),args.profile,args.device,args.limit)


if __name__=='__main__':main()
