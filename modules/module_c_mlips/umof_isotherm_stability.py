"""Bounded continuation of the one-seed MOF-5/CO2 CRAFTED comparison."""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import t
from ase.io import read

from .umof_isotherm_comparison import configure, preflight, digest, REFERENCE
from .umof_workflow import atomic_json, campaign_lock
from .datasets.common import load_host_guest_system
from .potential_backends.calculators import build_ase_calculator
from engines.mlip_mc_runner import run_mlip_mc_gcmc, _read_gcmc_transition_log, _reconstruct_gcmc_samples
from pipeline.planning import build_run_plan
from analysis.module_c_isotherm_report import create_report

ROOT = Path('outputs/module_C_potential_benchmark/campaigns/mof5_co2_umof_mh_025pct_stability_v1')
PRESSURES = [.1,1.,10.]
WINDOW = 20000
MAX_EXTRA = 20000
CHUNK = 10000


def assess(values):
    """Operational single-chain screening, not proof of equilibrium."""
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError('Invalid loading history')
    mean=float(x.mean()); centered=x-mean; var=float(np.var(x))
    if var<=1e-20:
        tau=float(len(x)); interval=None; batches=0; se=None
    else:
        nfft=1 << (2*len(x)-1).bit_length()
        spectrum=np.fft.rfft(centered,nfft)
        ac=np.fft.irfft(spectrum*np.conjugate(spectrum),nfft)[:len(x)]
        ac/=ac[0]
        total=0.;previous=float('inf')
        for i in range(1,len(ac)-1,2):
            pair=float(ac[i]+ac[i+1])
            if pair<=0: break
            pair=min(pair,previous);previous=pair;total+=pair
        tau=max(1.,1.+2.*total)
        block=max(500,int(np.ceil(2*tau)))
        batches=len(x)//block
        if batches>=8:
            means=x[-batches*block:].reshape(batches,block).mean(axis=1)
            se=float(means.std(ddof=1)/np.sqrt(batches))
            half=float(t.ppf(.975,batches-1)*se)
            interval=[mean-half,mean+half]
        else: interval=None;se=None
    half_drift=float(abs(x[len(x)//2:].mean()-x[:len(x)//2].mean()))
    tolerance=max(.02,.15*abs(mean))
    enough=len(x)>=WINDOW
    effective=float(len(x)/tau)
    precision=interval is not None and (interval[1]-interval[0])/2<=tolerance
    stable=bool(enough and effective>=20 and precision and half_drift<=tolerance)
    return {'status':'passes_operational_screen' if stable else 'not_resolved',
            'sample_count':len(x),'mean_mmol_g':mean,'std_mmol_g':float(x.std()),
            'first_half_mean_mmol_g':float(x[:len(x)//2].mean()),
            'second_half_mean_mmol_g':float(x[len(x)//2:].mean()),
            'half_difference_mmol_g':half_drift,'absolute_tolerance_mmol_g':tolerance,
            'statistical_inefficiency_estimate':tau,'effective_sample_count_estimate':effective,
            'batch_count':batches,'approximate_batch_mean_ci95':interval,
            'criteria':{'at_least_20000_samples':enough,'at_least_20_effective_samples':effective>=20,
                        'precision_within_tolerance':bool(precision),'half_drift_within_tolerance':half_drift<=tolerance},
            'interpretation':'Heuristic one-chain stability screen; does not establish equilibrium or independent-seed reproducibility.'}


def prepare(root):
    config=configure(single_seed=True)
    _,_,_,protocol=preflight(config)
    models=config['benchmark']['potential_benchmark']['campaign']['models']
    original=Path(config['output']['directory'])/'runs'
    jobs=[]
    # Prioritize the slow high-pressure points while the other GPUs cover low pressures.
    for pressure in reversed(PRESSURES):
        for index,model in enumerate(models):
            suffix='_10bar_single_seed' if pressure==10. else '_quick_single_seed'
            campaign=original/('mof5_co2_umof_mh_025pct_comparison_v1'+suffix)
            candidates=[]
            for source in campaign.glob('*/results/gcmc_pilot.json'):
                data=json.loads(source.read_text())
                if data['model']==model['name']:
                    candidates=[(source,data,p) for p in data['pressure_points'] if p['pressure_bar']==pressure]
            if len(candidates)!=1: raise ValueError('Expected exactly one completed source point')
            source,data,point=candidates[0]
            point_dir=Path(point['transition_log_path']).parent
            restart=point_dir/'restart'/f'restart_{pressure:.4f}bar.json'
            state=json.loads(restart.read_text())
            completed=state['n_equil_completed']+state['n_prod_completed']
            if completed!=point['equilibration_steps']+point['production_steps']:
                raise ValueError('Source checkpoint does not match completed result')
            name=f'{model["name"]}_{str(pressure).replace(".","p")}bar'
            jobs.append({'name':name,'model_index':index,'pressure_bar':pressure,'base_seed':104729+PRESSURES.index(pressure),
                'model':model,'source_result':str(source),'source_result_sha256':digest(source),
                'source_point_directory':str(point_dir),'restart_sha256':digest(restart),
                'trace_sha256':digest(point['transition_log_path']),
                'source_total_steps':completed,'source_runtime_seconds':point['runtime_seconds'],'equilibration_steps':point['equilibration_steps'],
                'max_total_steps':completed+MAX_EXTRA,'framework_mass_amu':data['framework_mass_amu']})
    manifest={'config':config,'protocol':protocol,'jobs':jobs,'window':WINDOW,'max_additional_steps_per_point':MAX_EXTRA,
              'screen':{'relative_precision_and_drift_tolerance':.15,'absolute_floor_mmol_g':.02,'min_effective_samples':20,'min_batches':8},
              'scope':'One seed, three pressures, CRAFTED UFF/DDEC reference. Bounded pilot; unresolved points prevent a definitive ranking.',
              'restart_rng':'MLIP-MC 0.1.3 does not save RNG state. Every continuation uses base_seed + completed_step_count; no bitwise uninterrupted-trajectory claim.'}
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest: raise ValueError('Stability campaign inputs changed')
    atomic_json(path,manifest)
    return manifest


def read_history(job,directory):
    label=Path(job['source_point_directory']).name
    point=directory/'engine'/label
    restart=point/'restart'/f'restart_{job["pressure_bar"]:.4f}bar.json'
    state=json.loads(restart.read_text());total=state['n_equil_completed']+state['n_prod_completed']
    transitions=_read_gcmc_transition_log(point/f'log_{job["pressure_bar"]:.4f}bar.bin')
    if transitions and transitions[-1]['step']>total: raise ValueError('Trace extends beyond saved checkpoint')
    values=np.array([r['uptake'] for r in _reconstruct_gcmc_samples(transitions,total_steps=total)])*1000./job['framework_mass_amu']
    n=min(WINDOW,total-job['equilibration_steps'])
    return values,assess(values[-n:]),total


def worker(root,index):
    manifest=json.loads((root/'manifest.json').read_text());job=manifest['jobs'][index]
    directory=root/'points'/job['name'];directory.mkdir(parents=True,exist_ok=True)
    engine=directory/'engine';engine.mkdir(exist_ok=True)
    destination=engine/Path(job['source_point_directory']).name
    if not destination.exists():
        if digest(job['source_result'])!=job['source_result_sha256']:raise ValueError('Source changed')
        if digest(Path(job['source_point_directory'])/f'log_{job["pressure_bar"]:.4f}bar.bin')!=job['trace_sha256']:raise ValueError('Source trace changed')
        shutil.copytree(job['source_point_directory'],destination)
    values,screen,total=read_history(job,directory)
    previous=directory/'assessment.json'
    continuations=json.loads(previous.read_text()).get('continuations',[]) if previous.exists() else []
    calc=None;runtime=sum(c.get('runtime_seconds',0.) for c in continuations)
    while screen['status']!='passes_operational_screen' and total<job['max_total_steps']:
        if calc is None:
            if digest(job['model']['model'])!=job['model']['sha256']:raise ValueError('Model changed')
            config=json.loads(json.dumps(manifest['config']))
            plan=build_run_plan(config);ff=plan['resources']['forcefield']
            host,gas,_=load_host_guest_system(plan['resources']['cif_path'],ff['adsorbates']['CO2'],ff['files']['pseudo_atoms'],
                cell_representation=plan['simulation']['cell_representation'],unit_cells=plan['simulation']['unit_cells'],
                cutoff_A=plan['simulation']['cutoff_A'],minimum_image_policy=plan['simulation']['minimum_image_policy'])
            gas=gas[np.asarray(gas.get_masses())>=.01]
            if not np.isclose(sum(host.get_masses()),job['framework_mass_amu']):raise ValueError('Framework mass changed')
            calc=build_ase_calculator(job['model'])
        target=min(total+CHUNK,job['max_total_steps'])
        atomic_json(directory/'status.json',{'status':'running','completed_steps':total,'target_steps':target,'max_steps':job['max_total_steps'],'screen':screen})
        start=time.monotonic()
        raw=run_mlip_mc_gcmc(calc,host,gas,component='CO2',temperature_K=298.,pressures_bar=[job['pressure_bar']],
            equilibration_steps=job['equilibration_steps'],production_steps=target-job['equilibration_steps'],
            seed=job['base_seed']+total,device='cuda',output_directory=engine,checkpoint_interval=1000)
        elapsed=time.monotonic()-start;runtime+=elapsed
        continuations.append({'from_step':total,'to_step':target,'continuation_seed':job['base_seed']+total,'runtime_seconds':elapsed})
        values,screen,total=read_history(job,directory)
        if total!=target:raise ValueError('Restart did not reach the requested target')
        atomic_json(directory/'assessment.json',{'screen':screen,'completed_steps':total,'continuations':continuations})
    n=screen['sample_count']
    record={'status':'completed','model':job['model']['name'],'pressure_bar':job['pressure_bar'],
            'base_seed':job['base_seed'],'loading_mol_per_kg':screen['mean_mmol_g'],
            'loading_std_mol_per_kg':screen['std_mmol_g'],'evaluation_window_start_step':total-n+1,
            'evaluation_window_end_step':total,'screen':screen,'continuations':continuations,
            'additional_runtime_seconds':runtime,'source_runtime_seconds':job['source_runtime_seconds'],'source_total_steps':job['source_total_steps'],
            'total_steps':total,'source_result':job['source_result'],'history_file':str(directory/'history.npz')}
    np.savez_compressed(directory/'history.npz',loading=values)
    atomic_json(directory/'point_result.json',record)
    atomic_json(directory/'status.json',{'status':'completed','screen_status':screen['status'],'completed_steps':total})


def make_report(root,manifest):
    records=[];missing=[]
    for job in manifest['jobs']:
        p=root/'points'/job['name']/'point_result.json'
        if p.exists():records.append(json.loads(p.read_text()))
        else:missing.append(job['name'])
    atomic_json(root/'progress.json',{'completed':len(records),'total':len(manifest['jobs']),'pending':missing})
    if missing:return None
    aggregated=root/'aggregate';aggregated.mkdir(exist_ok=True)
    for model in manifest['config']['benchmark']['potential_benchmark']['campaign']['models']:
        selected=sorted([r for r in records if r['model']==model['name']],key=lambda r:r['pressure_bar'])
        path=aggregated/model['name']/'results';path.mkdir(parents=True,exist_ok=True)
        atomic_json(path/'gcmc_pilot.json',{'status':'completed','method':'gcmc','material':'IRMOF-1','component':'CO2',
            'temperature_K':298.,'model':model['name'],'seed':104729,
            'sampling_status':'passes_operational_screen' if all(r['screen']['status']=='passes_operational_screen' for r in selected) else 'not_resolved',
            'pressure_points':[{'pressure_bar':r['pressure_bar'],'loading_mol_per_kg':r['loading_mol_per_kg'],
                'loading_std_mol_per_kg':r['loading_std_mol_per_kg'],'production_steps':r['screen']['sample_count'],
                'production_sample_count':r['screen']['sample_count']} for r in selected]})
    report=create_report(aggregated,root/'comparison','gcmc_pilot',REFERENCE)
    unresolved=[{'model':r['model'],'pressure_bar':r['pressure_bar'],'screen':r['screen']} for r in records if r['screen']['status']!='passes_operational_screen']
    metrics=report['reference_comparison']['metrics']
    ordered=sorted(metrics,key=lambda r:r['rmse_mol_per_kg'])
    assessment={'status':'pilot_screen_passed' if not unresolved else 'inconclusive_sampling',
                'descriptive_lowest_rmse_model':ordered[0]['model'],'reference_metrics':metrics,'unresolved_points':unresolved,
                'scope':manifest['scope'],'interpretation':'Descriptive CRAFTED comparison only. One seed and an operational screen do not establish reproducible accuracy or equilibrium.',
                'precision_note':'Within-chain batch estimates are archived per point; seed confidence intervals are unavailable.'}
    atomic_json(root/'comparison/final_assessment.json',assessment)
    atomic_json(root/'comparison/sampling_details.json',records)
    with (root/'comparison/runtime_details.csv').open('w',newline='') as f:
        timings=[{'model':r['model'],'pressure_bar':r['pressure_bar'],
                  'initial_mc_steps':r['source_total_steps'],'initial_seconds':r['source_runtime_seconds'],
                  'additional_mc_steps':r['total_steps']-r['source_total_steps'],
                  'additional_seconds':r['additional_runtime_seconds'],
                  'overall_seconds_per_mc_step':(r['source_runtime_seconds']+r['additional_runtime_seconds'])/r['total_steps']} for r in records]
        writer=csv.DictWriter(f,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
    comparisons=report['reference_comparison']['matched_points']
    lines=['# MOF-5 / CO2: UMOF versus 25% fine-tuning', '',
           '298 K; 0.1, 1 and 10 bar; one continued trajectory per model/pressure.', '',
           'Reference: CRAFTED UFF/DDEC, absolute loading. No DFT calculations are used.', '',
           'Assessment: '+assessment['status'], '',
           '| Model | MAE (mmol/g) | RMSE (mmol/g) |', '|---|---:|---:|']
    for row in metrics:
        lines.append(f"| {row['model']} | {row['mae_mol_per_kg']:.4f} | {row['rmse_mol_per_kg']:.4f} |")
    lines+=['', 'The lowest descriptive RMSE belongs to '+ordered[0]['model']+'.', '',
            ('Sampling remains unresolved at '+str(len(unresolved))+' point(s); a definitive ranking is withheld.'
             if unresolved else 'All points pass the operational screen; this remains a one-seed pilot result.'), '',
            'The screen uses a 20,000-step final evaluation window, correlation estimates, at least eight batch means, '
            'an effective-sample estimate of at least 20, and drift/precision tolerances of max(15% of loading, 0.02 mmol/g). '
            'It is a heuristic check, not proof of equilibrium. Full source and continuation histories are preserved.', '',
            'See sampling_details.json and loading_histories.png for drift and uncertainty diagnostics. '
            'runtime_details.csv reports observed trajectory cost, including source runs and additional work; '
            'it is not a controlled timing test on identical loaded configurations. Neither training cost nor '
            'DFT runtime is included in those timings.']
    (root/'comparison/RESULT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for row,pressure in enumerate(PRESSURES):
        for r in [x for x in records if x['pressure_bar']==pressure]:
            values=np.load(r['history_file'])['loading'];start=r['evaluation_window_start_step']-1
            bins=[values[i:i+500] for i in range(0,len(values),500)]
            axes[row,0].plot([(i*500)+(len(b)+1)/2 for i,b in enumerate(bins)],[b.mean() for b in bins],label=r['model'])
            tail=values[start:];axes[row,1].plot(np.arange(start+1,len(values)+1),np.cumsum(tail)/np.arange(1,len(tail)+1),label=r['model'])
            axes[row,0].axvline(start,color='gray',linestyle=':',alpha=.4)
        for col in (0,1):
            axes[row,col].set(title=f'{pressure:g} bar: '+('loading / 500-step bins' if col==0 else 'evaluation-window cumulative mean'),xlabel='MC step',ylabel='mmol/g')
            axes[row,col].legend(fontsize=7);axes[row,col].grid(alpha=.2)
    fig.suptitle('MOF-5 / CO2, 298 K — bounded one-seed stability comparison')
    for ext in ('png','svg'):fig.savefig(root/f'comparison/loading_histories.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    return assessment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status','report'])
    parser.add_argument('--worker',type=int)
    parser.add_argument('--gpus',default='0,3,1,2')
    args=parser.parse_args(argv);root=ROOT.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:worker(root,args.worker);return 0
    if args.action=='status':
        for p in [root/'run_status.json',root/'progress.json']:
            if p.exists(): print(p.name,p.read_text())
        for p in sorted((root/'points').glob('*/status.json')):print(p.parent.name,p.read_text())
        return 0
    with campaign_lock(root):
        manifest=prepare(root)
        if args.action=='prepare':print('Prepared bounded continuation of six points');return 0
        if args.action=='report':print(json.dumps(make_report(root,manifest),indent=2));return 0
        devices=deque(args.gpus.split(','))
        if len(set(devices))!=len(devices) or any(not d.isdigit() for d in devices):raise ValueError('Choose distinct physical GPUs')
        pending=deque((i,j) for i,j in enumerate(manifest['jobs']) if not (root/'points'/j['name']/'point_result.json').exists())
        running=[];atomic_json(root/'run_status.json',{'status':'running','pid':os.getpid(),'gpus':list(devices),'max_additional_mc_steps':len(manifest['jobs'])*MAX_EXTRA})
        try:
            while pending or running:
                while pending and devices:
                    index,job=pending.popleft();gpu=devices.popleft();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                    log=(root/(job['name']+'.log')).open('a')
                    command=[sys.executable,'-m','modules.module_c_mlips.umof_isotherm_stability','run','--worker',str(index)]
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running.append((process,log,gpu,job['name']));print('Started',job['name'],'GPU',gpu,flush=True)
                for item in list(running):
                    p,log,gpu,name=item;code=p.poll()
                    if code is None:continue
                    running.remove(item);log.close();devices.append(gpu)
                    if code:raise RuntimeError(f'{name} failed; inspect {root}/{name}.log')
                    print('Completed',name,flush=True);make_report(root,manifest)
                if running:time.sleep(1)
            assessment=make_report(root,manifest)
            atomic_json(root/'run_status.json',{'status':'completed','assessment_status':assessment['status']})
        except BaseException as exc:
            atomic_json(root/'run_status.json',{'status':'failed_or_interrupted','error':str(exc)})
            raise
        finally:
            for p,log,_,_ in running:
                if p.poll() is None:p.terminate()
            for p,log,_,_ in running:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                log.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
