"""Targeted periodic r2SCAN+D4 check of the fixed-geometry uMOF energy diagnostic.

CP2K evaluates r2SCAN; project-local dftd4 adds the periodic dispersion energy.
This is an independent CP2K reference, not an exact reproduction of VASP/PAW.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from ase.io import read, write

from .umof_energy_diagnostic import OUTPUT as DIAGNOSTIC, EV_TO_KJ_MOL
from .umof_isotherm_comparison import digest
from .umof_workflow import atomic_json, campaign_lock

OUTPUT = Path('outputs/module_C_umof_dft_reference_025pct_mof5')
PREFIX = Path('/home/ge64zuq/miniforge3/envs/MOF_sim')
EXECUTABLE = PREFIX/'bin/cp2k.psmp'
HARTREE_TO_EV = 27.211386245988
VALENCE = {'H': 1, 'C': 4, 'O': 6, 'Zn': 12}


def select_samples(rows):
    selected = []
    for name in ('umof_mh_pretrained', 'umof_mh_finetuned_025pct'):
        pool = [r for r in rows if r['source_model'] == name
                and float(r['pretrained_adsorption_energy_ev']) < -.02]
        pool.sort(key=lambda r: float(r['finetuned_minus_pretrained_ev']))
        if len(pool) < 2:
            raise ValueError('Need at least two bound poses per source model')
        selected.extend([int(pool[0]['sample_id']), int(pool[len(pool)//2]['sample_id'])])
    return selected


def render_input(atoms, *, project, cutoff, assets, real_indices=None):
    real = set(range(len(atoms))) if real_indices is None else set(real_indices)
    kinds = {}
    coordinates = []
    for i, (symbol, position) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
        name = symbol if i in real else symbol+'_ghost'
        kinds[name] = (symbol, i not in real)
        coordinates.append(f'      {name} '+' '.join(f'{value:.12f}' for value in position))
    kind_text = []
    for name, (symbol, ghost) in sorted(kinds.items()):
        q = VALENCE[symbol]
        kind_text.append(f'    &KIND {name}\n      ELEMENT {symbol}\n'
            f'      BASIS_SET TZV2P-MOLOPT-SCAN-GTH-q{q}\n'
            + ('      GHOST T\n' if ghost else f'      POTENTIAL GTH-SCAN-q{q}\n')+'    &END KIND')
    cell = '\n'.join(f'      {name} '+' '.join(f'{value:.12f}' for value in vector)
                     for name,vector in zip('ABC',atoms.cell.array))
    return f'''&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME {assets / 'BASIS_MOLOPT_UZH'}
    POTENTIAL_FILE_NAME {assets / 'POTENTIAL_UZH'}
    CHARGE 0
    MULTIPLICITY 1
    &QS
      EPS_DEFAULT 1.0E-12
    &END QS
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF 100
    &END MGRID
    &SCF
      EPS_SCF 1.0E-8
      MAX_SCF 200
      SCF_GUESS ATOMIC
      &OT
        MINIMIZER DIIS
        PRECONDITIONER FULL_SINGLE_INVERSE
      &END OT
      &OUTER_SCF
        EPS_SCF 1.0E-8
        MAX_SCF 5
      &END OUTER_SCF
    &END SCF
    &XC
      &XC_FUNCTIONAL
        &MGGA_X_R2SCAN
        &END MGGA_X_R2SCAN
        &MGGA_C_R2SCAN
        &END MGGA_C_R2SCAN
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
{cell}
      PERIODIC XYZ
    &END CELL
    &COORD
{chr(10).join(coordinates)}
    &END COORD
{chr(10).join(kind_text)}
  &END SUBSYS
&END FORCE_EVAL
'''


def parse_energy(text):
    if '[ABORT]' in text or 'PROGRAM ENDED AT' not in text:
        raise ValueError('CP2K calculation aborted or is incomplete')
    outcomes = re.findall(r'SCF run\s+(converged|NOT converged)', text, re.I)
    if not outcomes or outcomes[-1].lower() != 'converged':
        raise ValueError('SCF convergence is not established')
    values = re.findall(r'ENERGY\|\s+Total FORCE_EVAL.*?energy.*?:\s*([-+0-9.EeDd]+)', text)
    if not values:
        raise ValueError('Missing final CP2K energy')
    value = float(values[-1].replace('D','E').replace('d','e'))*HARTREE_TO_EV
    if not np.isfinite(value):
        raise ValueError('Nonfinite CP2K energy')
    return value


def d4_energy(atoms, output):
    sys.path.insert(0,str((output/'deps').resolve()))
    from dftd4.ase import DFTD4
    atoms = atoms.copy(); atoms.calc = DFTD4(method='r2scan', atm=True)
    return float(atoms.get_potential_energy())


def prepare(output):
    output = output.resolve()
    rows = list(csv.DictReader((DIAGNOSTIC/'paired_energies.csv').open()))
    selected = select_samples(rows)
    host = read(DIAGNOSTIC/'host.extxyz')
    pairs = read(DIAGNOSTIC/'combined.extxyz', ':')
    guests = read(DIAGNOSTIC/'guests.extxyz', ':')
    assets = output/'assets'; assets.mkdir(exist_ok=True)
    for name in ('BASIS_MOLOPT_UZH','POTENTIAL_UZH'):
        source = PREFIX/'share/cp2k/data'/name
        if not (assets/name).exists(): shutil.copy2(source, assets/name)
        if digest(source) != digest(assets/name): raise ValueError('CP2K data file changed')
    version = subprocess.run([str(EXECUTABLE),'--version'],capture_output=True,text=True,check=True).stdout.replace('\x00','')
    (output/'cp2k_version.txt').write_text(version)
    settings = {'functional': 'r2SCAN (LibXC)', 'dispersion': 'DFT-D4 3.7.0, r2scan parameters, ATM enabled, periodic',
        'basis': 'TZV2P-MOLOPT-SCAN-GTH', 'pseudopotential': 'GTH-SCAN (SCAN-optimized, not VASP PAW)',
        'primary_cutoff_Ry': 1200, 'check_cutoff_Ry': 1600, 'relative_cutoff_Ry': 100,
        'eps_scf': 1e-8, 'periodicity': 'XYZ', 'kpoints': 'Gamma', 'charge': 0, 'multiplicity': 1,
        'geometry': 'Frozen, identical to MLIP diagnostic; no relaxation',
        'numerical_energy_difference_target_kj_mol': .5,
        'selection': 'Per model trajectory: largest discrepancy and median discrepancy among baseline-bound single-guest poses; diagnostic, not a representative benchmark',
        'comparison_scope': 'Independent CP2K/GTH r2SCAN+D4 calculation; not exact reproduction of uMOF VASP/PAW training settings',
        'sources': ['https://arxiv.org/html/2608.28100v1#S2.SS1.SSS1',
                    'https://dftd4.readthedocs.io/en/latest/reference/python.html']}
    jobs = []
    def add(name, atoms, cutoff=1200, real=None):
        job_dir = output/'jobs'/name; job_dir.mkdir(parents=True,exist_ok=True)
        write(job_dir/'geometry.extxyz',atoms,columns=['symbols','positions'])
        text=render_input(atoms,project=name,cutoff=cutoff,assets=assets,real_indices=real)
        input_path=job_dir/'cp2k.inp'
        if input_path.exists() and input_path.read_text()!=text: raise ValueError('Input changed for existing job')
        input_path.write_text(text)
        physical = atoms if real is None else atoms[list(real)]
        jobs.append({'name':name,'directory':str(job_dir),'cutoff_Ry':cutoff,
                     'input_sha256':digest(input_path),'d4_energy_ev':d4_energy(physical,output),
                     'ghost_count':0 if real is None else len(atoms)-len(real)})
    add('host',host)
    for index in selected:
        pair, guest = pairs[index], guests[index]
        add(f'pose_{index:02d}_gas',guest)
        add(f'pose_{index:02d}_combined',pair)
        add(f'pose_{index:02d}_host_ghost',pair,real=list(range(len(host))))
        add(f'pose_{index:02d}_gas_ghost',pair,real=list(range(len(host),len(pair))))
    # Check grid convergence of an interaction-energy subtraction, not per-atom total energies.
    index=selected[0]
    add('host_1600',host,1600)
    add(f'pose_{index:02d}_gas_1600',guests[index],1600)
    add(f'pose_{index:02d}_combined_1600',pairs[index],1600)
    manifest={'settings':settings,'selected_sample_ids':selected,'jobs':jobs,
              'diagnostic_input_sha256':digest(DIAGNOSTIC/'inputs.json'),
              'paired_energies_sha256':digest(DIAGNOSTIC/'paired_energies.csv'),
              'assets_sha256':{name:digest(assets/name) for name in ('BASIS_MOLOPT_UZH','POTENTIAL_UZH')}}
    path=output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest: raise ValueError('DFT manifest changed')
    atomic_json(path,manifest)
    return manifest


def collect_job(job):
    path=Path(job['directory'])
    if digest(path/'cp2k.inp')!=job['input_sha256']: raise ValueError('CP2K input changed')
    energy=parse_energy((path/'cp2k.out').read_text())
    result={'name':job['name'],'r2scan_energy_ev':energy,'d4_energy_ev':job['d4_energy_ev'],
            'total_energy_ev':energy+job['d4_energy_ev'],'input_sha256':job['input_sha256']}
    atomic_json(path/'energy.json',result)
    return result


def execute_job(job, threads):
    path=Path(job['directory'])
    try:
        if (path/'cp2k.out').exists():
            return collect_job(job)
    except ValueError:
        # Preserve failed/incomplete output before a deliberate restart.
        shutil.copy2(path/'cp2k.out',path/f'previous_{time.time_ns()}.out')
    atomic_json(path/'status.json',{'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    env=os.environ.copy(); env.update(OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with (path/'launcher.log').open('w') as log:
        process=subprocess.run([str(EXECUTABLE),'-i','cp2k.inp','-o','cp2k.out'],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        atomic_json(path/'status.json',{'status':'failed','returncode':process.returncode})
        raise RuntimeError(f'CP2K exited {process.returncode}: {path}')
    result=collect_job(job)
    atomic_json(path/'status.json',{'status':'completed','seconds':time.monotonic()-start})
    return result


def collect(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if digest(DIAGNOSTIC/'paired_energies.csv') != manifest['paired_energies_sha256']:
        raise ValueError('Source model energies changed after DFT preparation')
    energies={}; missing=[]
    for job in manifest['jobs']:
        try: energies[job['name']]=collect_job(job)
        except (OSError,ValueError): missing.append(job['name'])
    atomic_json(output/'progress.json',{'completed':len(energies),'total':len(manifest['jobs']),'pending_or_failed':missing})
    if missing: return None
    source={int(r['sample_id']):r for r in csv.DictReader((DIAGNOSTIC/'paired_energies.csv').open())}
    rows=[]
    for index in manifest['selected_sample_ids']:
        tag=f'pose_{index:02d}'
        pair=energies[tag+'_combined']; host=energies['host']; gas=energies[tag+'_gas']
        plain=pair['total_energy_ev']-host['total_energy_ev']-gas['total_energy_ev']
        correction=(host['r2scan_energy_ev']-energies[tag+'_host_ghost']['r2scan_energy_ev']
                    +gas['r2scan_energy_ev']-energies[tag+'_gas_ghost']['r2scan_energy_ev'])
        cp=plain+correction
        row={'sample_id':index,'dft_uncorrected_adsorption_kj_mol':plain*EV_TO_KJ_MOL,
             'dft_counterpoise_adsorption_kj_mol':cp*EV_TO_KJ_MOL,'bsse_correction_kj_mol':correction*EV_TO_KJ_MOL,
             'pretrained_kj_mol':float(source[index]['pretrained_adsorption_energy_kj_mol']),
             'finetuned_kj_mol':float(source[index]['finetuned_adsorption_energy_kj_mol'])}
        rows.append(row)
    index=manifest['selected_sample_ids'][0];tag=f'pose_{index:02d}'
    high=energies[tag+'_combined_1600']['total_energy_ev']-energies['host_1600']['total_energy_ev']-energies[tag+'_gas_1600']['total_energy_ev']
    low=rows[0]['dft_uncorrected_adsorption_kj_mol']
    grid_delta=abs(high*EV_TO_KJ_MOL-low)
    stats={name:{kind:float(np.mean([abs(r[name+'_kj_mol']-r[reference]) for r in rows]))
                 for kind,reference in [('mae_to_uncorrected_kj_mol','dft_uncorrected_adsorption_kj_mol'),
                                        ('mae_to_counterpoise_kj_mol','dft_counterpoise_adsorption_kj_mol')]}
           for name in ('pretrained','finetuned')}
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'status':'completed','rows':rows,'model_errors':stats,'interaction_grid_delta_1200_vs_1600_kj_mol':grid_delta,
             'grid_check_passed':grid_delta<=.5,'settings':manifest['settings'],
             'interpretation':'Targeted model comparison only. Inspect grid convergence, BSSE and cross-code/reference differences before ranking. No validation of full isotherm.'}
    atomic_json(output/'comparison.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    x=np.arange(len(rows))
    for key,label in [('pretrained_kj_mol','Pretrained UMOF'),('finetuned_kj_mol','25% fine-tuning'),
                      ('dft_uncorrected_adsorption_kj_mol','CP2K r2SCAN+D4'),('dft_counterpoise_adsorption_kj_mol','CP2K r2SCAN+D4 + counterpoise')]:
        ax.plot(x,[r[key] for r in rows],'o-',label=label)
    ax.set(xticks=x,xticklabels=[str(r['sample_id']+1) for r in rows],xlabel='Pose number from energy diagnostic',ylabel='Adsorption energy (kJ/mol)',title='Fixed-geometry MOF-5 / CO2: targeted DFT comparison')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    for extension in ('png','svg'): fig.savefig(output/f'comparison.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (output/'REPORT.md').write_text('# Targeted r2SCAN+D4 check\n\n'+json.dumps(summary,indent=2)+'\n')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status','collect'])
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args(argv);output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.action=='status':
        manifest=json.loads((output/'manifest.json').read_text())
        result=[]
        for job in manifest['jobs']:
            path=Path(job['directory'])/'status.json'
            result.append({'name':job['name'],**(json.loads(path.read_text()) if path.exists() else {'status':'pending'})})
        print(json.dumps(result,indent=2));return 0
    if args.parallel<1 or args.threads<1: parser.error('Resources must be positive')
    with campaign_lock(output):
        if args.action=='collect':
            summary=collect(output);print(json.dumps(summary or json.loads((output/'progress.json').read_text()),indent=2));return 0
        if args.action=='run':
            atomic_json(output/'run_status.json',{'status':'preparing_d4','pid':os.getpid()})
        manifest=prepare(output)
        if args.action=='prepare':
            print(f'Prepared {len(manifest["jobs"])} CP2K single points for poses {manifest["selected_sample_ids"]}');return 0
        atomic_json(output/'run_status.json',{'status':'running','pid':os.getpid(),'parallel':args.parallel,'threads_per_job':args.threads})
        failures=[]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures={pool.submit(execute_job,job,args.threads):job for job in manifest['jobs']}
            for future in as_completed(futures):
                job=futures[future]
                try: future.result(); print('Completed',job['name'],flush=True)
                except Exception as exc: failures.append({'job':job['name'],'error':str(exc)});print('Failed',job['name'],str(exc),flush=True)
                collect(output)
        atomic_json(output/'run_status.json',{'status':'failed' if failures else 'completed','failures':failures})
        if failures: raise RuntimeError(f'{len(failures)} DFT jobs failed; inspect run_status.json')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
