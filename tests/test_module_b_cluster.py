from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from converter.cif_to_lammps_data import load_framework_structure, parse_cif_atom_site_charges
from modules import module_b_campaign as al
from modules.module_b_cluster import base_config, maximin, import_pilot, read
from modules.module_b_profiles import point, simplex_candidates
from modules.module_b_runtime import campaign_lock, add_raw_uncertainty

ROOT = Path('outputs/module_B_active_learning')

class SymmetryTests(unittest.TestCase):
    def test_cubtc_charge_expansion(self):
        s=load_framework_structure('external/RASPA2/structures/mofs/cif/Cu-BTC.cif',cell_representation='conventional',cutoff_A=12.8)
        self.assertEqual(s.atom_count,624)
        self.assertAlmostEqual(sum(s.charges),0,places=8)
        self.assertEqual(sum(v=='Cu' for v in s.symbols),48)
        self.assertEqual(sum(abs(q-1.248)<1e-10 for q in s.charges),48)
        self.assertEqual(len(set(tuple(round(v,7) for v in xyz) for xyz in s.scaled_positions)),624)

    def test_equivalent_duplicate_conflicting_charge(self):
        text='''data_test
_cell_length_a 30
_cell_length_b 30
_cell_length_c 30
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
C1 C 0 0 0 0.1
C2 C 1 0 0 CHARGE
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.cif';p.write_text(text.replace('CHARGE','0.1'))
            self.assertEqual(load_framework_structure(p).atom_count,1)
            p.write_text(text.replace('CHARGE','0.2'))
            with self.assertRaisesRegex(ValueError,'Conflicting charges'):
                load_framework_structure(p)

    def test_existing_charge_order_and_cell_parity(self):
        from ase.io import read as ase_read
        for path,rep in [('CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif',[1,1,1]),
                         ('CRAFTED-2.0.0/CIF_FILES/PACMOF/ZIF-8.cif',[2,2,2])]:
            atoms=ase_read(path)
            charges=[r.charge for r in parse_cif_atom_site_charges(path)]
            atoms.set_initial_charges(charges)
            atoms=atoms.repeat(rep)
            s=load_framework_structure(path,unit_cells=rep)
            np.testing.assert_allclose(s.charges,atoms.get_initial_charges())
            np.testing.assert_allclose(s.positions,atoms.positions)

class DesignTests(unittest.TestCase):
    def test_reproducible_maximin_and_pilot_anchors(self):
        candidates=simplex_candidates(['CO2','N2'],[.1,10],[298.15])
        pilot=[point(p,298.15,{'CO2':.15,'N2':.85}) for p in (.1,1,5,10)]
        a=maximin(candidates+pilot,12,pilot)
        self.assertEqual(a,maximin(candidates+pilot,12,pilot))
        self.assertEqual(a[:4],pilot)
        self.assertEqual(len({p['candidate_id'] for p in a}),12)
        self.assertGreater(max(p['mixture']['CO2'] for p in a),.9)

    def state(self):
        from tests.test_module_b_general import GeneralCampaignTests
        return GeneralCampaignTests().state()

    def test_budget_and_retry_limit(self):
        s=self.state();p=s['candidates'][0]
        s['results']=[dict(p,precision_met=False,precision_retries=0)]
        s['options']['max_training_points']=1
        self.assertEqual(al.recommend(s)['action'],'improve_precision')
        s['results'][0]['precision_retries']=1
        self.assertEqual(al.recommend(s)['status'],'budget_exhausted')

    def test_holdout_labels_and_uncertainty_do_not_control_stopping(self):
        s=self.state();held={p['candidate_id'] for p in s['holdout']}
        s['options']['exclude_holdout_from_stopping']=True
        s['results']=[dict(p,precision_met=True) for p in s['candidates'] if p['candidate_id'] not in held][:3]
        prediction={'kernels':{},'components':{c:[{'value':1.,'lower':.99,'upper':1e6 if p['candidate_id'] in held else 1.01}
                        for p in s['candidates']] for c in s['components']}}
        with patch.object(al,'predict',return_value=prediction):
            a=al.recommend(s)
            s['holdout_results']=[{'poison':'must not read'}]
            self.assertEqual(al.recommend(s),a)
            self.assertEqual(a['status'],'surrogate_uncertainty_target_met')

    def test_explicit_budget_and_parallel_limit(self):
        with tempfile.TemporaryDirectory() as d:
            for budget,jobs in [(0,1),(1,4)]:
                with self.assertRaises(ValueError): al.execute(Path(d)/'state.json',budget,jobs=jobs)

    def test_execution_contingent_and_error_status(self):
        from modules.module_b_active_learning import _save
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';s=self.state();_save(p,s)
            cfg=Path(d)/'config.json';_save(cfg,s['base_config'])
            request={'status':'request','request':{'action':'sample','config_path':str(cfg)}}
            with patch.object(al,'propose',return_value=request), \
                 patch('pipeline.planning.build_run_plan',return_value={}), \
                 patch('pipeline.prepare.prepare_benchmark',return_value={}), \
                 patch('pipeline.materialization.materialize_benchmark',return_value={}), \
                 patch('pipeline.runners.run_isotherm',return_value={'summary_file':'unused'}) as run, \
                 patch.object(al,'ingest'):
                result=al.execute(p,2,jobs=3)
                self.assertEqual(run.call_count,2)
                self.assertEqual(run.call_args.kwargs['jobs'],3)
                self.assertEqual(result['status'],'execution_budget_exhausted')
                run.side_effect=RuntimeError('controlled failure')
                with self.assertRaisesRegex(RuntimeError,'controlled failure'):al.execute(p,1)
                self.assertEqual(read(p)['status'],'technical_error')
                self.assertIn('logs',read(p)['last_error_logs'])

    def test_writer_lock_other_process_and_reentrant(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'campaign.json'
            with campaign_lock(p),campaign_lock(p):
                script='from modules.module_b_runtime import campaign_lock\nwith campaign_lock('+repr(str(p))+'): pass'
                r=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0)
                self.assertIn('active writer',r.stderr)
            with campaign_lock(p): pass

@unittest.skipUnless((ROOT/'smoke/mof5/smoke_result.json').exists(),'requires saved real smoke evidence')
class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.origin=Path.cwd()
        self.temp=tempfile.TemporaryDirectory()
        shutil.copytree(ROOT,Path(self.temp.name)/ROOT)
        os.chdir(self.temp.name)

    def tearDown(self):
        os.chdir(self.origin)
        self.temp.cleanup()

    def test_three_complete_smokes_and_separation(self):
        identities=[]
        for name in ('mof5','zif8','cubtc'):
            folder=ROOT/'smoke'/name;s=read(folder/'campaign.json')
            self.assertEqual(read(folder/'smoke_result.json')['status'],'passed')
            self.assertEqual(len(s['results']),3);self.assertEqual(len(s['holdout_results']),1)
            self.assertTrue({p['candidate_id'] for p in s['results']}.isdisjoint({p['candidate_id'] for p in s['holdout']}))
            self.assertEqual(s['campaign_kind'],'technical_smoke')
            for r in s['results']+s['holdout_results']:
                self.assertEqual(set(r['seeds']),{12345,23456,34567})
                self.assertEqual(len(r['raw_diagnostics']['logs']),3)
            identities.append(s['model_identity'])
        self.assertEqual(len(set(identities)),3)

    def test_pilot_import_idempotent_and_no_short_run(self):
        p=ROOT/'production/mof5/campaign.json'
        before=read(p);after=import_pilot(p,self.origin/'outputs/module_B_mixture_selectivity/runs/mof5_co2_n2_validation_pilot_2_longer')
        self.assertEqual(after['results'],before['results'])
        self.assertEqual(len(after['results']),4)
        for r in after['results']:
            self.assertEqual(r['replicate_count'],3)
        self.assertEqual([r['precision_met'] for r in after['results']],[False,False,True,True])
        row=after['results'][0]
        self.assertEqual(row['classic_seed_components']['N2']['standard_error_molecules'],0)
        self.assertGreater(row['components']['N2']['standard_error_molecules'],0)
        with self.assertRaisesRegex(ValueError,'restricted'):
            import_pilot(ROOT/'smoke/mof5/campaign.json')

    def test_framework_and_box_unchanged_in_all_smoke_dumps(self):
        dumps=list((ROOT/'smoke').glob('*/simulation/runs/al_*/dumps/*.lammpstrj'))
        self.assertEqual(len(dumps),36)
        for dump in dumps:
            lines=dump.read_text().splitlines();index=0;frames=[];boxes=[]
            while index < len(lines):
                self.assertEqual(lines[index],'ITEM: TIMESTEP')
                count=int(lines[index+3]);boxes.append(lines[index+5:index+8])
                columns=lines[index+8].split()[2:]
                atoms=[dict(zip(columns,row.split())) for row in lines[index+9:index+9+count]]
                frames.append({a['id']:tuple(a[c] for c in ('x','y','z')) for a in atoms})
                index+=9+count
            self.assertGreaterEqual(len(frames),3)
            for frame,box in zip(frames[1:],boxes[1:]):
                self.assertEqual(box,boxes[0])
                self.assertEqual({k:frame[k] for k in frames[0]},frames[0])

    def test_missing_noise_rejected_and_zero_boundary(self):
        s=read(ROOT/'smoke/mof5/campaign.json')
        zero=point(1,298.15,{'CO2':0,'N2':1})
        p=al.predict(s,[zero]);self.assertEqual(p['components']['CO2'][0]['value'],0)
        s['results'][0]['components']['CO2']['standard_error_molecules']=None
        with self.assertRaisesRegex(ValueError,'independent seeds'):al.predict(s,[zero])
        s['results'][0]['components']['CO2']['standard_error_molecules']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):al.predict(s,[zero])

    def test_smoke_rejected_as_production_observation(self):
        source=read(ROOT/'smoke/mof5/campaign.json')['results'][0]['source_summary']
        s=read(ROOT/'production/mof5/campaign.json')
        candidate=read(ROOT/'smoke/mof5/campaign.json')['results'][0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'campaign.json';cfg=deepcopy(s['base_config'])
            cfg['conditions'].update(pressures_bar=[candidate['pressure_bar']],temperature_K=candidate['temperature_K'])
            cfg['adsorbates']['mixture']=candidate['mixture']
            c=Path(d)/'request.json';c.write_text(json.dumps(cfg))
            s['pending_request']={'candidate':candidate,'action':'sample','config_path':str(c)}
            p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'sampling differs'):al.ingest(p,source)

    def test_pending_role_cannot_switch(self):
        with tempfile.TemporaryDirectory() as d:
            s=read(ROOT/'smoke/mof5/campaign.json');p=Path(d)/'campaign.json';p.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'different role'):al.execute(p,1,holdout=True)

    def test_precision_request_quadruples_steps_without_new_training_state(self):
        p=ROOT/'production/mof5/campaign.json';before=read(p)
        result=al.propose(p);cfg=read(result['request']['config_path'])
        self.assertEqual(result['action'],'improve_precision')
        self.assertEqual(cfg['simulation']['cycles'],200000)
        self.assertEqual(cfg['simulation']['initialization_cycles'],20000)
        self.assertEqual(read(p)['results'],before['results'])
        self.assertEqual(al.propose(p)['request'],result['request'])

    def test_actual_completed_seeds_reused_without_subprocess(self):
        from pipeline.materialization import materialize_benchmark
        from pipeline.prepare import prepare_benchmark
        from pipeline.planning import build_run_plan
        from pipeline.runners import run_isotherm
        for name in ('mof5','zif8','cubtc'):
            s=read(ROOT/'smoke'/name/'campaign.json')
            request=next(h['request'] for h in s['history'] if h.get('request',{}).get('action')=='holdout')
            cfg=read(request['config_path'])
            materialized=materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
            original_run=subprocess.run
            def no_lammps(command,*args,**kwargs):
                if Path(command[0]).name.startswith('lmp'):
                    raise AssertionError('Completed seed rerun')
                return original_run(command,*args,**kwargs)
            with patch('pipeline.runners.subprocess.run',side_effect=no_lammps):
                result=run_isotherm(materialized)
            summary=read(result['summary_file'])
            self.assertEqual(len(summary['replicate_results']),3)
            self.assertEqual(summary['status'],'completed')

    def test_relocation_and_checksum_guard(self):
        origin=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)
            # No CRAFTED tree: fixed campaign must use its archived assets only.
            shutil.copytree(ROOT/'smoke/mof5',target/ROOT/'smoke/mof5')
            os.chdir(target)
            try:
                p=ROOT/'smoke/mof5/campaign.json';s=read(p)
                self.assertEqual(al.identity(s['base_config']),s['model_identity'])
                self.assertEqual(al.propose(p)['request'],s['pending_request'])
                s['pending_request']=None;p.write_text(json.dumps(s))
                asset=Path(s['base_config']['material']['cif_path']);asset.write_text(asset.read_text()+'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'assets changed'):al.propose(p)
            finally: os.chdir(origin)

if __name__=='__main__':unittest.main()
