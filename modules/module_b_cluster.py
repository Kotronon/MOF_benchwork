"""Prepared CO2/N2 campaigns for MOF-5, ZIF-8 and Cu-BTC (repository-relative bundle)."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from modules import module_b_campaign as al
from modules.module_b_active_learning import _save
from modules.module_b_profiles import point, simplex_candidates
from modules.module_b_runtime import campaign_lock, asset_fingerprints
from pipeline.planning import build_run_plan
from pipeline.prepare import prepare_benchmark
from pipeline.materialization import materialize_benchmark

DEFAULT_ROOT = Path('outputs/module_B_active_learning')
PILOT = Path('outputs/module_B_mixture_selectivity/runs/mof5_co2_n2_validation_pilot_2_longer')
MATERIALS = ('mof5', 'zif8', 'cubtc')


def read(path):
    return json.loads(Path(path).read_text())


def base_config(name, folder, smoke=False):
    cfg = read(f'input_json_files/benchmark_module_b_al_{name}.json')
    cfg['benchmark']['reference'] = {'source': 'none'}
    cfg['benchmark']['paper'] = {'target_protocol': 'Technical AL readiness; external validation open.'}
    cfg['output'].update(directory=str(folder / 'simulation'), run_id='prepared_model', resume=True,
                         save_dumps=True, save_restarts=True)
    cfg['conditions']['pressures_bar'] = [.1, 10.0]
    cfg['simulation']['seeds'] = [12345, 23456, 34567]
    cfg['simulation'].update(cycles=50000, initialization_cycles=5000)
    if name == 'zif8':
        cfg['material'].update(name='ZIF-8', charge_scheme='PACMOF')
        cfg['simulation'].update(unit_cells=[2,2,2], cell_representation='source')
        cfg['conditions']['temperature_K'] = 300.0
    elif name == 'cubtc':
        cfg['material'].update(name='Cu-BTC', charge_scheme='as_source',
                              cif_path='external/RASPA2/structures/mofs/cif/Cu-BTC.cif')
        cfg['conditions']['temperature_K'] = 300.0
    if smoke:
        cfg['simulation'].update(cycles=100, initialization_cycles=100, thermo_every_steps=1)
        cfg['output'].update(dump_every_steps=100, restart_every_steps=100)
    return cfg


def freeze_config(cfg, folder):
    """Resolve once through Module A, then archive and explicitly select every asset."""
    plan = prepare_benchmark(build_run_plan(cfg))
    sources = {}
    def archive(path):
        p = Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        dest = folder / 'model' / (digest[:16] + '_' + p.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != p.read_bytes():
            raise ValueError(f'Archived model collision: {dest}')
        shutil.copyfile(p, dest)
        sources[str(dest)] = {'source': str(p), 'sha256': digest}
        return str(dest)
    cfg = deepcopy(cfg)
    cfg['material'].update(name=plan['parameters']['material_id'],
                           cif_path=archive(plan['inputs']['framework_cif']))
    ff = cfg['simulation']['forcefield']
    ff['files'] = {k:archive(v) for k,v in plan['inputs']['forcefield_files'].items()}
    ff['definitions'] = {k:archive(v) for k,v in plan['inputs']['adsorbate_definitions'].items()}
    if plan['inputs']['adsorbate_parameter_files']:
        raise ValueError('Fixed CO2/N2 profiles unexpectedly resolved extra parameter bundles.')
    _save(folder / 'model_manifest.json', {
        'model_identity': al.identity(cfg), 'assets': sources,
        'parameters': plan['parameters'], 'units': 'LAMMPS real; pressure bar; loading molecules/cell',
        'source_model': 'UFF + existing CRAFTED CO2/N2 definitions; ZIF-8 is not the NIST reference model',
        'framework_flexible': False, 'physical_validation': 'open',
    })
    return cfg


def maximin(candidates, count, initial=()):
    import numpy as np
    selected = list(initial)
    domain = al.features(candidates, candidates)
    used = {p['candidate_id'] for p in selected}
    while len(selected) < count:
        if selected:
            observed = al.features(selected, candidates)
            distance = np.linalg.norm(domain[:,None,:] - observed[None,:,:], axis=2).min(axis=1)
        else:
            distance = np.linalg.norm(domain - .5, axis=1)
        for i,p in enumerate(candidates):
            if p['candidate_id'] in used:
                distance[i] = -1
        index = int(np.argmax(distance))
        if distance[index] < 0:
            raise ValueError('Insufficient distinct initial candidates.')
        selected.append(candidates[index]); used.add(candidates[index]['candidate_id'])
    return selected


def check_dependencies():
    import ase, numpy, scipy, sklearn, matplotlib  # noqa: F401
    executable = shutil.which('lmp')
    if not executable:
        raise RuntimeError('lmp is missing from PATH; activate the Python/LAMMPS environment.')
    result = subprocess.run([executable, '-help'], capture_output=True, text=True, timeout=30)
    if result.returncode or 'gcmc' not in result.stdout or 'pppm' not in result.stdout:
        raise RuntimeError('LAMMPS must provide GCMC and PPPM. ' + result.stderr)
    return {'lammps': executable, 'python': sys.executable,
            'lammps_help_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()}


def portable_copy(source, target):
    """Copy only pilot evidence, rewrite its own locations; never change the original."""
    if not target.exists():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('dumps', 'restarts', 'analysis', '*.png', '*.svg'))
        for p in target.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.in'):
                text = p.read_text().replace(str(Path.cwd()) + '/', '')
                text = text.replace(str(source) + '/', str(target) + '/')
                p.write_text(text)


def import_pilot(path, source=PILOT):
    """Only the explicitly selected longer pilot; no pooling of overlapping traces."""
    with campaign_lock(path):
        state = read(path)
        if state.get('campaign_kind') != 'production' or state.get('material_key') != 'mof5':
            raise ValueError('Pilot import is restricted to the production MOF-5 campaign.')
        source_summary = source / 'isotherm_summary.json'
        original_digest = hashlib.sha256(source_summary.read_bytes()).hexdigest()
        marker = state.get('pilot_import', {})
        if marker.get('summary_sha256') == original_digest and marker.get('completed'):
            return state
        if marker and marker.get('summary_sha256') != original_digest:
            raise ValueError('Different or overlapping pilot cannot be pooled.')
        if not marker and (state['results'] or state['pending_request']):
            raise ValueError('Import pilot only into an empty campaign; overlapping runs cannot be pooled.')
        prep = read(source / 'prepare_summary.json')['prepare_plan']
        expected = prepare_benchmark(build_run_plan(state['base_config']))
        parameters = prep['parameters']
        ignored = {'pressures_bar', 'mixture', 'save_dumps', 'save_restarts', 'dump_every_steps', 'restart_every_steps'}
        for key,value in expected['parameters'].items():
            if key not in ignored and parameters.get(key) != value:
                raise ValueError(f'Pilot context mismatch: {key}')
        if asset_fingerprints(prep['inputs']) != asset_fingerprints(expected['inputs']):
            raise ValueError('Pilot model assets do not match production.')
        summary = read(source_summary)
        if summary.get('status') != 'completed':
            raise ValueError('Pilot must be complete.')
        points = [point(p, parameters['temperature_K'], parameters['mixture']) for p in (.1,1,5,10)]
        if {r['pressure_bar'] for r in summary['results']} != {.1,1,5,10}:
            raise ValueError('Expected four longer-pilot states.')
        held = {p['candidate_id'] for p in state['holdout']}
        if any(p['candidate_id'] in held for p in points):
            raise ValueError('Pilot overlaps locked holdouts.')
        archive = Path(path).parent / 'pilot_evidence'
        portable_copy(source, archive)
        summary = read(archive / 'isotherm_summary.json')
        state['pilot_import'] = {'summary_sha256':original_digest, 'completed':False}
        _save(path,state)
        for p in points:
            if p['candidate_id'] in {r['candidate_id'] for r in state['results']}:
                continue
            filtered = deepcopy(summary)
            for key in ('results','replicate_results'):
                filtered[key] = [r for r in filtered[key] if r['pressure_bar'] == p['pressure_bar']]
            cfg = deepcopy(state['base_config'])
            cfg['conditions']['pressures_bar'] = [p['pressure_bar']]
            cfg['adsorbates']['mixture'] = p['mixture']
            cfgpath = archive / f"config_{p['candidate_id']}.json"
            _save(cfgpath,cfg)
            filtered_path = archive / f"summary_{p['candidate_id']}.json"
            _save(filtered_path,filtered)
            state['pending_request'] = {'candidate': p, 'action':'pilot_import', 'config_path':str(cfgpath)}
            _save(path,state)
            state = al.ingest(path,filtered_path)
        state['pilot_import'] = {'summary_sha256': original_digest, 'completed':True, 'source':str(source_summary),
                                 'status':'Pilot ausgewertet; quantitative Referenzvalidierung offen',
                                 'overlap_policy':'Only longer pilot; shorter prefix is not independent.'}
        _save(path,state)
        return state


def archive_prepared_manifest(folder, prepared):
    manifest = read(folder / 'model_manifest.json')
    work = Path(prepared['working_directory'])
    files = [p for sub in ('data', 'forcefield', 'molecules', 'inputs') for p in (work/sub).rglob('*') if p.is_file()]
    rendered = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if manifest.get('rendered_sha256') and manifest['rendered_sha256'] != rendered:
        raise ValueError('Previously prepared model inputs changed; inspect model_manifest.json.')
    manifest['rendered_sha256'] = rendered
    manifest['resolved_parameters'] = prepared['parameters']
    implementation = ['modules/module_b_campaign.py','modules/module_b_runtime.py',
                      'modules/module_b_cluster.py','pipeline/lammps_inputs.py','pipeline/eos.py',
                      'pipeline/materialization.py','converter/cif_to_lammps_data.py',
                      'converter/forcefield_to_lammps.py','analysis/module_b_convergence_report.py']
    manifest['implementation_sha256'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in implementation}
    _save(folder / 'model_manifest.json',manifest)


def prepare(name, root=DEFAULT_ROOT, smoke=False):
    folder = Path(root) / ('smoke' if smoke else 'production') / name
    path = folder / 'campaign.json'
    with campaign_lock(path):
        if path.exists():
            state = read(path)
            expected_kind = 'technical_smoke' if smoke else 'production'
            if state.get('campaign_kind') != expected_kind or state.get('material_key') != name:
                raise ValueError('Campaign material or smoke/production role mismatch.')
            if al.identity(state['base_config']) != state['model_identity']:
                raise ValueError('Prepared model has changed.')
            if not (folder / 'readiness.json').exists():
                materialize_benchmark(prepare_benchmark(build_run_plan(state['base_config'])))
                if name == 'mof5' and not smoke:
                    import_pilot(path)
                _save(folder / 'readiness.json', {'status':'prepared','production_started':False,
                      'physical_validation':'open','campaign_kind':state['campaign_kind']})
            model_run = folder / 'simulation/runs/prepared_model/prepare_summary.json'
            if model_run.exists():
                archive_prepared_manifest(folder, read(model_run)['result'])
            return path
        cfg = freeze_config(base_config(name,folder,smoke),folder)
        _save(folder / 'config.json', cfg)
        t = cfg['conditions']['temperature_K']
        make = lambda p,y: point(p,t,{'CO2':y,'N2':1-y})
        candidates = simplex_candidates(['CO2','N2'],[.1,10],[t])
        candidates += [make(p,y) for p in (.1,1,10) for y in (0.,1.)]
        pilot = [make(p,.15) for p in (.1,1,5,10)] if name == 'mof5' and not smoke else []
        if smoke:
            initial = [make(.1,.1),make(10,.1),make(1,.9)]
            holdout = [make(5,.5)]
            candidates = initial + holdout + [make(.5,.3),make(3,.7),make(.1,0),make(10,1)]
        else:
            candidates += pilot
            candidates = list({p['candidate_id']:p for p in candidates}.values())
            initial = maximin(candidates,12,pilot)
            import numpy as np
            eligible = [p for p in candidates if p['candidate_id'] not in {q['candidate_id'] for q in initial}]
            indices = np.random.default_rng(2026).choice(len(eligible),16,replace=False)
            holdout = [eligible[i] for i in indices]
        state = al.initialize(cfg,path,candidates=candidates,budget=4 if smoke else 24)
        state.update(initial_design=initial,holdout=holdout,
                     campaign_kind='technical_smoke' if smoke else 'production', material_key=name)
        state['options'].update(raw_log_uncertainty=True,exclude_holdout_from_stopping=True,
                                max_precision_retries=0 if smoke else 1)
        _save(path,state)
        # Materialize all fixed inputs with the same preparation path used by A.
        prepared = materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
        archive_prepared_manifest(folder, prepared)
        if pilot:
            import_pilot(path)
        _save(folder / 'readiness.json', {'status':'prepared', 'production_started':False,
              'physical_validation':'open', 'campaign_kind':state['campaign_kind']})
        return path


def smoke_test(name,root,jobs):
    path = prepare(name,root,smoke=True)
    with campaign_lock(path):
        state = read(path)
        if not (Path(path).parent / 'smoke_result.json').exists():
            # Resume outstanding initial requests; never execute the fourth adaptive point.
            remaining = 3-len(state['results'])
            if remaining:
                al.execute(path,remaining,jobs=jobs)
            state = read(path)
            if len(state['results']) != 3:
                raise RuntimeError('Smoke needs exactly three completed training states.')
            model = al.predict(state,state['candidates'])
            decision = al.recommend(state)
            if decision['status'] != 'request' or decision['action'] != 'sample':
                raise RuntimeError('Smoke GP failed to propose an adaptive state.')
            # Execute the locked holdout before storing the adaptive pending request.
            if not state['holdout_results']:
                al.execute(path,1,holdout=True,jobs=jobs)
            result = al.report(path)
            proposed = al.propose(path)
            resumed = al.propose(path)
            if proposed['request'] != resumed['request']:
                raise RuntimeError('Pending AL request changed on resume.')
            state = read(path)
            if len(state['results']) != 3 or len(state['holdout_results']) != 1:
                raise RuntimeError('Smoke observation counts changed on resume.')
            _save(path.parent / 'smoke_result.json', {
                'status':'passed', 'seed_runs':12, 'training_states':3,'holdout_states':1,
                'gp_kernels':model['kernels'], 'next_candidate':proposed['request']['candidate'],
                'resume':'same pending request, no duplicate observation',
                'scientific_precision_required':False, 'physical_validation':'open',
                'report':result,
            })
        state = read(path)
        for record in state['results'] + state['holdout_results']:
            for log in record['raw_diagnostics']['logs']:
                if hashlib.sha256(Path(log['log_file']).read_bytes()).hexdigest() != log['sha256']:
                    raise ValueError('Smoke evidence log changed: ' + log['log_file'])
        return read(path.parent / 'smoke_result.json')


def overview(root):
    with campaign_lock(Path(root)/"overview.json"):
        return _overview(root)


def _overview(root):
    rows=[]
    for kind in ('production','smoke'):
        for name in MATERIALS:
            p=Path(root)/kind/name/'campaign.json'
            if not p.exists(): continue
            s=read(p)
            allrows=s['results']+s['holdout_results']
            replaced=[h['replaced_observation'] for h in s['history'] if 'replaced_observation' in h]
            rows.append({'material':name,'kind':kind,'status':s['status'],
                         'training_points':len(s['results']), 'holdout_points':len(s['holdout_results']),
                         'precision_unmet':sum(not r['precision_met'] for r in allrows),
                         'seed_runs':sum(r['replicate_count'] for r in allrows+replaced),
                         'wall_seconds':sum(r.get('total_wall_time_seconds') or 0 for r in allrows+replaced),
                         'technical_test':read(p.parent/'smoke_result.json')['status'] if (p.parent/'smoke_result.json').exists() else 'not_run',
                         'last_error':s.get('last_error'), 'logs':s.get('last_error_logs'),
                         'external_validation':'open'})
    _save(Path(root)/'overview.json',rows)
    ready = all((Path(root)/'production'/name/'readiness.json').exists() and
                any(r['material']==name and r['kind']=='smoke' and r['technical_test']=='passed' for r in rows)
                for name in MATERIALS)
    _save(Path(root)/'readiness.json',{'status':'technically_ready' if ready else 'preparation_or_smoke_incomplete',
                                    'external_validation':'open', 'required_smoke_seed_runs':36})
    import csv
    if rows:
        with (Path(root)/'overview.csv').open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','start','resume','holdout','status','report'])
    parser.add_argument('material',choices=[*MATERIALS,'all'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--jobs',type=int,choices=[1,2,3],default=1)
    args=parser.parse_args(argv)
    if args.root.is_absolute():
        parser.error('--root must be repository-relative for portable campaigns.')
    if args.command in ('start','resume','holdout') and (args.max_requests is None or args.max_requests<1):
        parser.error('An explicit positive --max-requests is required (per MOF).')
    if args.command in ('prepare','smoke','start','resume','holdout'):
        environment=check_dependencies()
        _save(args.root/'environment.json',environment)
        if args.command=='prepare':
            shutil.copyfile('data/module_b/CLUSTER_AL.md',args.root/'START_HERE.md')
    for name in MATERIALS if args.material=='all' else [args.material]:
        path=args.root/'production'/name/'campaign.json'
        if args.command=='prepare': prepare(name,args.root)
        elif args.command=='smoke': smoke_test(name,args.root,args.jobs)
        elif args.command in ('start','resume','holdout'):
            if not path.exists(): raise FileNotFoundError(f'Run prepare first: {path}')
            state=read(path)
            if state.get('campaign_kind') != 'production' or state.get('material_key') != name:
                raise ValueError('Production command requires its matching production campaign.')
            al.execute(path,args.max_requests,holdout=args.command=='holdout',jobs=args.jobs)
            al.report(path)
        elif args.command=='report': al.report(path)
    print(json.dumps(overview(args.root),indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
