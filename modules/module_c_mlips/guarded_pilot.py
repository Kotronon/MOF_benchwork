"""Frozen MOF-5 three-model pilot or pressure screening with bounded execution.

No scheduler submission, downloading, automatic threshold fitting or extension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii
from ase.io import read, write

from analysis.module_c_closure import digest, dump, seed_interval, table, write_csv
from . import umof_adsorption_comparison as adapter
from .closure_campaign import completed_result, execution_versions, verify_bundle
from .umof_isotherm_stability import assess
from .umof_workflow import atomic_json, campaign_lock

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'input_json_files/benchmark_mof5_co2_umof_three_model_comparison.json'
ENTRY = 'modules.module_c_mlips.guarded_pilot'


def radii_array(config):
    radii = np.full(len(vdw_radii), np.nan)
    for symbol, value in config['overlap']['radii_A'].items():
        radii[atomic_numbers[symbol]] = value
    return radii


def allowed(atoms, host_atoms, radii):
    n = (len(atoms)-host_atoms)//3
    if len(atoms) != host_atoms+3*n: raise ValueError('Expected rigid three-site CO2')
    if not n: return True
    groups = np.r_[np.zeros(host_atoms, int), np.repeat(np.arange(n)+1, 3)]
    distances = atoms.get_all_distances(mic=True)
    limits = radii[atoms.numbers][:, None]+radii[atoms.numbers][None, :]
    if not np.isfinite(limits).all(): raise ValueError('Missing fixed overlap radius')
    return bool(np.all(distances[groups[:, None] != groups[None, :]] >= limits[groups[:, None] != groups[None, :]]))


def engines(config):
    _, parent = adapter.engine_classes(.8)
    radii = radii_array(config)
    class FixedWallGCMC(parent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.vdw = radii.copy()
    return FixedWallGCMC


def guarded_calculator(base, host, guest, floor):
    """Abort the run on a gross energy anomaly; never reject it as an MC move."""
    from ase.calculators.calculator import Calculator, all_changes
    reference = adapter.consistent_calculator(base, guest)
    empty = host.copy(); empty.calc = reference
    host_energy = float(empty.get_potential_energy())
    class Guard(Calculator):
        implemented_properties = ['energy']
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            state = atoms.copy(); state.calc = reference
            value = float(state.get_potential_energy())
            n = (len(state)-len(host))//len(guest)
            if n > 0:
                interaction = value-host_energy-n*reference.gas_energy_eV
                if interaction/n < floor:
                    raise RuntimeError(f'Energy guard: {interaction/n:.6g} eV/guest < {floor}; run aborted, not a rejected MC move')
            self.results = {'energy': value}
    result = Guard(); result.gas_energy_eV = reference.gas_energy_eV
    return result


def probes(host, guest, diagnostic):
    frames = []
    def add(atoms, kind, nh):
        a = adapter.clean_atoms(atoms); a.info.update(probe_kind=kind, host_atoms=nh)
        frames.append(a)
    for orientation in ('parallel', 'axial'):
        for distance in (1., 1.5, 2., 2.5, 3., 3.5, 4., 5.):
            a, b = guest.copy(), guest.copy()
            b.translate([distance, 0, 0] if orientation=='parallel' else [0,0,distance])
            pair = a+b; pair.set_cell([30,30,30]); pair.set_pbc(False); pair.center()
            add(pair, f'dimer_{orientation}_{distance:g}A', 0)
    manifest = json.loads((diagnostic/'manifest.json').read_text())
    for row in manifest['cases']:
        if digest(diagnostic/row['snapshot']) != row['snapshot_sha256']:
            raise ValueError('Diagnostic snapshot checksum mismatch')
    # Same real problematic host contact, isolated from other guests.
    row = next(r for r in manifest['cases'] if r['case']==1)
    snapshot = read(diagnostic/row['snapshot']); nh = len(host)
    if not np.array_equal(snapshot.numbers[:nh], host.numbers) or not np.allclose(snapshot.positions[:nh], host.positions, atol=2e-7, rtol=0):
        raise ValueError('Diagnostic snapshot uses a different framework')
    start = nh+3*row['remove_guest_index']; molecule = snapshot[start:start+3]
    combined = host+molecule
    distances = combined.get_all_distances(mic=True)[:nh, nh:]
    i,j = np.unravel_index(np.argmin(distances),distances.shape)
    vector = combined.get_distance(i,nh+j,mic=True,vector=True); vector /= np.linalg.norm(vector)
    for shift in (0.,.25,.5,.75,1.,1.5,2.,3.,4.):
        moved = molecule.copy(); moved.translate(shift*vector)
        add(host+moved, f'host_contact_shift_{shift:g}A',nh)
    rng = np.random.default_rng(7727)
    for index in range(32):
        molecule = guest.copy()
        q,_ = np.linalg.qr(rng.normal(size=(3,3)))
        molecule.positions = molecule.positions @ q
        molecule.translate(rng.random(3) @ host.cell.array)
        add(host+molecule,f'host_pose_{index:02d}',nh)
    for case in (0,2,3):
        row = next(r for r in manifest['cases'] if r['case']==case)
        add(read(diagnostic/row['snapshot']),f'archived_mh_case_{case}',nh)
    return frames


def pressure_points(config):
    return config['pressures_bar'] if config.get('schema_version', 1) == 2 else [config['pressure_bar']]


def validate_sampling(pressures, seeds, gcmc):
    if not pressures or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or p <= 0 for p in pressures):
        raise ValueError('Pressures must be finite positive numbers')
    if len(set(pressures)) != len(pressures):
        raise ValueError('Duplicate pressures')
    if not seeds or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct integer seeds in [0, 2**32) required')
    for key in ('equilibration_steps', 'production_steps', 'checkpoint_interval'):
        if type(gcmc[key]) is not int or gcmc[key] <= 0:
            raise ValueError(f'Positive integer {key} required')


def prepare(root, source_path=None):
    if root.exists(): raise FileExistsError(root)
    source_path = Path(source_path).resolve() if source_path else SOURCE
    source = json.loads(source_path.read_text())
    configurable = source_path != SOURCE
    pressures = source['conditions']['pressures_bar'] if configurable else [1.]
    seeds = source['simulation']['seeds'] if configurable else [49979687,49979693,49979713]
    gcmc = {key: source['benchmark']['potential_benchmark']['campaign']['stages']['gcmc_pilot'][key]
            for key in ('equilibration_steps','production_steps','checkpoint_interval')}
    validate_sampling(pressures, seeds, gcmc)
    if source['material']['name'] != 'MOF-5' or source['adsorbates']['components'] != ['CO2'] or source['conditions']['temperature_K'] != 298.:
        raise ValueError('This entrypoint supports only MOF-5 / CO2 at 298 K')
    model_specs = source['benchmark']['potential_benchmark']['campaign']['models']
    frozen_specs = json.loads(SOURCE.read_text())['benchmark']['potential_benchmark']['campaign']['models']
    if model_specs != frozen_specs:
        raise ValueError('Keep the frozen three-model selection, heads and float64 protocol')
    old = PROJECT/'reports/module_c_closure/cluster_bundle'
    verify_bundle(old)
    root.mkdir(parents=True); (root/'assets').mkdir()
    origins = {}
    def asset(path, name, expected=None):
        if expected and digest(path)!=expected: raise ValueError(f'Model hash mismatch: {path}')
        dest=root/'assets'/name; shutil.copy2(path,dest)
        origins[str(dest.relative_to(root))]={'path':str(path.relative_to(PROJECT)),'sha256':digest(path)}
        return {'path':str(dest.relative_to(root)),'sha256':digest(dest)}
    models=[]
    for spec in source['benchmark']['potential_benchmark']['campaign']['models']:
        label=spec['name']; profile='polar' if 'polar' in label else 'mh'
        models.append({'label':label,'profile':profile,'head':spec['head'],'dtype':'float64',
                       **asset(PROJECT/spec['model'],label+'.model',spec['sha256'])})
    host_asset=asset(old/'assets/mof5.extxyz','mof5.extxyz')
    gas_asset=asset(old/'assets/co2_mh.extxyz','co2.extxyz')
    reference=asset(old/'assets/DDEC_IRMOF-1_UFF_CO2_298.csv','reference.csv')
    host,guest=read(root/host_asset['path']),read(root/gas_asset['path'])
    symbols=sorted(set(host.get_chemical_symbols()+guest.get_chemical_symbols()))
    radii={s:float(vdw_radii[atomic_numbers[s]]-.35) for s in symbols}
    if any(not np.isfinite(r) or r<=0 for r in radii.values()):raise ValueError('Invalid ASE vdW radius')
    config={'schema_version':1,'material':'MOF-5','component':'CO2','temperature_K':298.,
        'models':models,'framework':host_asset,'adsorbate':gas_asset,'reference':reference,
        'pressure_bar':1.,'seeds':seeds,
        'gcmc':gcmc,
        'smoke':{'equilibration_steps':20,'production_steps':40,'checkpoint_interval':20},
        'overlap':{'rule':'sum of ASE vdW radii minus 0.35 A per atom','radii_A':radii,
                   'scope':'Predeclared legacy MLIP-MC exclusion approximation; not fitted to CRAFTED. New protocol, not a continuation.'},
        'guard':{'minimum_interaction_eV_per_guest':-2.,
                 'scope':'Gross anomaly stop, not a physical accuracy threshold; aborts entire run and does not truncate sampled energies.'},
        'precision':{'relative_seed_ci95_target':.2,'absolute_floor_mmol_g':.02},
        'scope':'Bounded three-model one-pressure diagnostic pilot; all models float64, same host, gas, reference, wall and log-space acceptance.'}
    if configurable:
        config.pop('pressure_bar')
        config.update(schema_version=2, pressures_bar=pressures,
                      scope='Bounded pressure screening; common float64, host, gas, reference, wall and log-space acceptance. One seed supplies no between-seed uncertainty.')
    dump(root/'pilot.json',config)
    diagnostic=PROJECT/'reports/umof_cluster_diagnostic_2026-11-10'
    frames=probes(host,guest,diagnostic)
    write(root/'assets/probes.extxyz',frames)
    shutil.copy2(source_path,root/'source_comparison.json')
    for directory in ('analysis','modules','engines','pipeline','converter','resolvers','parsers'):
        for path in (PROJECT/directory).rglob('*.py'):
            dest=root/path.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ('requirements.txt','requirements-mlip-mc.txt','requirements-module-c-umof.txt'):
        shutil.copy2(PROJECT/name,root/name)
    instructions='modules/module_c/umof_mof5_single_seed_isotherm.md' if configurable else 'modules/module_c/umof_guarded_pilot.md'
    shutil.copy2(PROJECT/'docs'/instructions,root/'START_HERE.md')
    (root/'run_gpu.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\npython -m modules.module_c_mlips.guarded_pilot start --directory .\n')
    dump(root/'bundle_manifest.json',{'origins':origins,'diagnostic_manifest_sha256':digest(diagnostic/'manifest.json'),
        'file_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},'new_simulations_started':False})
    check(root)
    archive=root.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as handle:handle.add(root,arcname=root.name)
    Path(str(archive)+'.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(f'Archive: {archive}',flush=True)


def check(root):
    verify_bundle(root)
    config=json.loads((root/'pilot.json').read_text())
    if config.get('schema_version', 1) not in (1, 2):raise ValueError('Unknown pilot schema')
    if len(config['models'])!=3 or len({m['label'] for m in config['models']})!=3:
        raise ValueError('Expected three distinct frozen models')
    validate_sampling(pressure_points(config), config['seeds'], config['gcmc'])
    if config.get('schema_version', 1)==1 and (len(config['seeds'])!=3 or config['pressure_bar']!=1.):
        raise ValueError('Legacy pilot requires three seeds at 1 bar')
    if config['temperature_K']!=298.:raise ValueError('Unexpected pilot temperature')
    for spec in [config['framework'],config['adsorbate'],config['reference'],*config['models']]:
        if digest(root/spec['path'])!=spec['sha256']:raise ValueError('Changed asset')
    for model in config['models']:
        if model['dtype']!='float64':raise ValueError('Common float64 protocol required')
    counts={stage:len(jobs(config,stage)) for stage in ('scan','smoke','pilot')}
    steps=counts['pilot']*(config['gcmc']['equilibration_steps']+config['gcmc']['production_steps'])
    print(f'Verified frozen inputs: 3 models; {len(pressure_points(config))} pressure(s); {len(config["seeds"])} seed(s); '
          f'{counts["scan"]} scans + {counts["smoke"]} smoke jobs + {counts["pilot"]} pilot jobs ({steps} MC steps).',flush=True)
    return config


def jobs(config,stage):
    seeds=config['seeds'] if stage=='pilot' else [config['seeds'][0]]
    if config.get('schema_version', 1)==1 or stage=='scan':
        return [{'id':f'{m["label"]}_s{s}','model':m['label'],'seed':s} for m in config['models'] for s in seeds]
    return [{'id':f'{m["label"]}_p{repr(float(p)).replace(".","p")}bar_s{s}',
             'model':m['label'],'seed':s,'pressure_bar':p}
            for m in config['models'] for p in pressure_points(config) for s in seeds]


def versions(device):
    if device=='cuda':return execution_versions()
    import importlib.metadata
    return {'device':'cpu','versions':{k:importlib.metadata.version(k) for k in ('torch','mace-torch','mlip-mc','ase','numpy','scipy')}}


def identity(root,stage,job,runtime):
    return {'bundle_sha256':digest(root/'bundle_manifest.json'),'stage':stage,'job':job,'runtime':runtime}


def gate(root,config,stage,runtime,local_model=None):
    for previous in ({'scan':[], 'smoke':['scan'],'pilot':['scan','smoke']}[stage]):
        for job in jobs(config,previous):
            if local_model and job['model'] != local_model:continue
            result=completed_result(root/'results'/previous/job['id'],identity(root,previous,job,runtime))
            if result is None or not result['gate_passed']:
                raise RuntimeError(f'Missing or failed {previous} gate: {job["id"]}')


def calculate_scan(root,config,model,base,host,guest,output):
    from .umof_profiles import polar_conditioning
    def energy(atoms):
        a=adapter.clean_atoms(atoms);polar_conditioning(a);a.calc=base
        value=float(a.get_potential_energy())
        if not np.isfinite(value):raise ValueError('Nonfinite scan energy')
        return value
    gas=guest.copy();gas.set_cell([30,30,30]);gas.set_pbc(False);gas.center()
    eg=energy(gas);eh=energy(host);rows=[];radii=radii_array(config)
    for a in read(root/'assets/probes.extxyz',index=':'):
        nh=int(a.info['host_atoms']);n=(len(a)-nh)//3
        value=energy(a)-(eh if nh else 0)-n*eg
        rows.append({'probe':a.info['probe_kind'],'host_atoms':nh,'guests':n,
                     'allowed_by_wall':allowed(a,nh,radii),'interaction_eV':value,'interaction_per_guest_eV':value/n})
    failures=[r for r in rows if r['allowed_by_wall'] and r['interaction_per_guest_eV']<config['guard']['minimum_interaction_eV_per_guest']]
    coverage={kind:sum(r['allowed_by_wall'] and r['probe'].startswith(kind) for r in rows) for kind in ('dimer','host')}
    write_csv(output/'scan.csv',rows)
    return {'stage':'scan','model':model['label'],'gate_passed':not failures and all(coverage.values()),
            'coverage':coverage,'failures':failures,'rows':rows,'gas_reference_eV':eg,
            'scope':'Finite diagnostic probes only; passing does not validate potential or production convergence.'}


def worker(root,stage,job,attempt,device):
    from .umof_profiles import calculator
    from engines.mlip_mc_runner import run_mlip_mc_gcmc, _read_gcmc_transition_log, _reconstruct_gcmc_samples
    config=check(root);model=next(m for m in config['models'] if m['label']==job['model'])
    if stage=='pilot' and device!='cuda':raise ValueError('Scientific pilot requires CUDA')
    gate(root,config,stage,versions(device),job['model'] if device=='cpu' else None)
    host,guest=read(root/config['framework']['path']),read(root/config['adsorbate']['path'])
    base=calculator(root/model['path'],model=model['profile'],head=model['head'],device=device,dtype=model['dtype'])
    if base.head!=model['head']:raise ValueError('Wrong head')
    heights=1/np.linalg.norm(np.linalg.inv(host.cell.array),axis=0)
    if heights.min()<2*base.r_max:raise ValueError('Cell below twice model cutoff')
    if stage=='scan':result=calculate_scan(root,config,model,base,host,guest,attempt)
    else:
        g=config['gcmc' if stage=='pilot' else 'smoke']
        calc=guarded_calculator(base,host,guest,config['guard']['minimum_interaction_eV_per_guest'])
        pressure=job.get('pressure_bar',config.get('pressure_bar'))
        raw=run_mlip_mc_gcmc(calc,host,guest,component='CO2',temperature_K=298.,pressures_bar=[pressure],
            equilibration_steps=g['equilibration_steps'],production_steps=g['production_steps'],seed=job['seed'],
            device=device,output_directory=attempt/'engine',checkpoint_interval=g['checkpoint_interval'],
            engine_class=engines(config))
        point=raw['pressure_points'][0]
        history=_reconstruct_gcmc_samples(_read_gcmc_transition_log(Path(point['transition_log_path'])),total_steps=g['equilibration_steps']+g['production_steps'])
        prod=history[g['equilibration_steps']:]
        values=np.array([r['uptake'] for r in prod])*1000/host.get_masses().sum()
        if len(values)!=g['production_steps']:raise ValueError('Incomplete production trace')
        diagnostics=assess(values)
        # Check every saved checkpoint geometry, not just the endpoint.
        snapshots=list((attempt/'engine').glob('**/checkpoints_*/checkpoint_*/traj.xyz'))
        if not snapshots:raise ValueError('No checkpoint structures')
        for snapshot in snapshots:
            if not allowed(read(snapshot),len(host),radii_array(config)):
                raise ValueError(f'Checkpoint violates frozen exclusion wall: {snapshot}')
        rows=[{'step':g['equilibration_steps']+i+1,'loading_mmol_g':float(v),'interaction_eV':float(r['interaction_energy_ev'])} for i,(r,v) in enumerate(zip(prod,values))]
        write_csv(attempt/'production.csv',rows)
        result={'stage':stage,'model':job['model'],'seed':job['seed'],'pressure_bar':pressure,'gate_passed':True,
            'loading_mmol_g':float(values.mean()),'screen':diagnostics,'raw':raw,'gas_reference_eV':calc.gas_energy_eV,
            'mean_interaction_eV_all_production_steps':float(np.mean([r['interaction_energy_ev'] for r in prod])),
            'checkpoint_structures_checked':len(snapshots),'scope':'gate_passed means technical execution, not convergence'}
    dump(attempt/'result.json',result)


def run(root,stage,device='cuda',model=None,max_jobs=1,index=None):
    config=check(root);runtime=versions(device)
    if stage=='pilot' and device!='cuda':raise ValueError('Scientific pilot requires CUDA')
    if device=='cpu' and not model:raise ValueError('CPU tests require an explicit --model')
    gate(root,config,stage,runtime,model if device=='cpu' else None)
    selected=jobs(config,stage)
    if index is not None:
        if index<0 or index>=len(selected):raise ValueError('Job index out of range')
        selected=[selected[index]]
    if model:selected=[j for j in selected if j['model']==model]
    if not selected:raise ValueError('No matching jobs')
    count=0
    for job in selected:
        folder=root/'results'/stage/job['id'];folder.mkdir(parents=True,exist_ok=True)
        ident=identity(root,stage,job,runtime)
        with campaign_lock(folder):
            previous=completed_result(folder,ident)
            if previous is not None:
                if not previous['gate_passed']:raise RuntimeError(f'Diagnostic gate failed: {folder}')
                print(f'Reuse {stage}/{job["id"]}',flush=True);continue
            if count>=max_jobs:break
            numbers=[int(p.name.split('-')[-1]) for p in folder.glob('attempt-*')]
            attempt=folder/f'attempt-{max(numbers,default=0)+1:03d}';attempt.mkdir()
            cmd=[sys.executable,'-m',ENTRY,'worker','--directory',str(root),'--stage',stage,'--job-index',str(jobs(config,stage).index(job)),'--attempt',str(attempt),'--device',device]
            print(f'Run {stage}/{job["id"]}; log: {attempt/"worker.log"}',flush=True)
            with (attempt/'worker.log').open('w') as log:
                process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                try:code=process.wait()
                except BaseException:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise
            if code:raise RuntimeError(f'Worker failed; inspect {attempt/"worker.log"}')
            result=attempt/'result.json'
            atomic_json(folder/'completed.json',{'identity':ident,'result_path':str(result.relative_to(folder)),'result_sha256':digest(result)})
            if not json.loads(result.read_text())['gate_passed']:raise RuntimeError(f'Energy scan gate failed: {result}')
            count+=1


def report(root):
    config=check(root);rows=[];missing=[];runtimes=set()
    for job in jobs(config,'pilot'):
        folder=root/'results/pilot'/job['id'];marker=folder/'completed.json'
        if not marker.exists():missing.append(job['id']);continue
        stored=json.loads(marker.read_text());ident=stored['identity']
        if ident!=identity(root,'pilot',job,ident['runtime']):raise ValueError('Result input identity changed')
        result=completed_result(folder,ident);runtimes.add(json.dumps(ident['runtime'],sort_keys=True))
        pressure=job.get('pressure_bar',config.get('pressure_bar'))
        if 'pressure_bar' in result and result['pressure_bar']!=pressure:raise ValueError('Result pressure mismatch')
        rows.append({'model':job['model'],'pressure_bar':pressure,'seed':job['seed'],'loading_mmol_g':result['loading_mmol_g'],
                     'chain_status':result['screen']['status'],'effective_samples':result['screen']['effective_sample_count_estimate'],
                     'drift_mmol_g':result['screen']['half_difference_mmol_g']})
    if len(runtimes)>1:raise ValueError('Mixed runtime environments')
    summaries=[];ref=np.loadtxt(root/config['reference']['path'],delimiter=',')
    for model in config['models']:
        for pressure in pressure_points(config):
            selected=[r for r in rows if r['model']==model['label'] and r['pressure_bar']==pressure]
            if not selected:continue
            reference_pressures=ref[:,0]/1e5
            if pressure<reference_pressures.min() or pressure>reference_pressures.max():
                raise ValueError('Reference extrapolation is not allowed')
            reference=float(np.interp(pressure,reference_pressures,ref[:,1]))
            stats=seed_interval([r['loading_mmol_g'] for r in selected])
            tolerance=max(config['precision']['absolute_floor_mmol_g'],config['precision']['relative_seed_ci95_target']*abs(stats['mean']))
            chain_passed=all(r['chain_status']=='passes_operational_screen' for r in selected)
            passed=len(selected)==len(config['seeds']) and stats['ci95_half_width'] is not None and stats['ci95_half_width']<=tolerance and chain_passed
            summaries.append({'model':model['label'],'pressure_bar':pressure,**stats,
                              'chain_screens_passed':chain_passed,'between_seed_uncertainty_available':stats['ci95_half_width'] is not None,
                              'all_screens_passed':passed,'reference_mmol_g':reference,'descriptive_bias_mmol_g':stats['mean']-reference})
    out=root/'report';out.mkdir(exist_ok=True)
    expected=len(jobs(config,'pilot'))
    dump(out/'assessment.json',{'completed_jobs':len(rows),'expected_jobs':expected,'missing':missing,'points':summaries,'equilibrium_established':False})
    if rows:write_csv(out/'seeds.csv',rows)
    if summaries:write_csv(out/'points.csv',summaries)
    (out/'REPORT.md').write_text('# MOF-5 / CO₂: begrenzter Drei-Modell-Vergleich\n\n'
        f'{len(rows)}/{expected} Läufe bei {pressure_points(config)} bar und 298 K; {len(config["seeds"])} Seed(s) je Modell/Druck. '
        'Festgeschriebene Ausschlussregel und gemeinsame float64-Präzision; getrennt von historischen Läufen.\n\n'
        +table(summaries,[('model','Modell'),('pressure_bar','bar'),('mean','mmol/g'),('ci95_half_width','95-%-Halbbreite'),('chain_screens_passed','Kettenscreens bestanden'),('all_screens_passed','Alle Screens bestanden')])
        +'\n\nBei einem Seed ist das ein Screening: keine Unsicherheit zwischen unabhängigen Seeds und keine statistisch abgesicherte Modellrangfolge. '
        '`None` bedeutet nicht bestimmbar, nicht null. Kettenscreens bleiben erhalten; sie ersetzen keine unabhängigen Wiederholungen. '
        'CRAFTED ist eine klassische Referenz. Keine automatische Verlängerung oder Gleichgewichtsfreigabe.\n')
    print(f'Report: {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','check','scan','smoke','pilot','start','worker','report'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='Source comparison JSON for prepare; freezes its pressures, seeds and MC budgets')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--model');parser.add_argument('--max-jobs',type=int,default=1)
    parser.add_argument('--job-index',type=int);parser.add_argument('--stage',choices=['scan','smoke','pilot'])
    parser.add_argument('--attempt',type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    if args.max_jobs<1:parser.error('--max-jobs must be positive')
    if args.config and args.action!='prepare':parser.error('--config is only supported with prepare')
    if args.action=='prepare':return prepare(root,args.config)
    if args.action=='check':return check(root)
    if args.action=='report':return report(root)
    if args.action=='worker':
        if not args.stage or args.job_index is None or not args.attempt:parser.error('worker needs stage, index, attempt')
        config=check(root);return worker(root,args.stage,jobs(config,args.stage)[args.job_index],args.attempt,args.device)
    if args.action=='start':
        if args.device!='cuda':parser.error('start requires CUDA; CPU is for explicitly selected local tests')
        config=check(root)
        for stage in ('scan','smoke','pilot'):run(root,stage,'cuda',max_jobs=len(jobs(config,stage)))
        return report(root)
    run(root,args.action,args.device,args.model,args.max_jobs,args.job_index)


if __name__=='__main__':main()
