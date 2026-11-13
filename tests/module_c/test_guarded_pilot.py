"""Scientific guard and execution-gate tests; synthetic fixtures only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from analysis.module_c_closure import digest, dump
from modules.module_c_mlips.guarded_pilot import allowed, gate, guarded_calculator, identity, jobs, radii_array, report, validate_sampling
from modules.module_c_mlips.campaign import _run_adsorption


class ArtificialEnergy(Calculator):
    implemented_properties=['energy']
    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        self.results={'energy':-100. if len(atoms)>3 else 0.}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.config={'overlap':{'radii_A':{'Zn':1.04,'C':1.35,'O':1.17}},
                     'models':[{'label':x} for x in ('mh','polar','ft')], 'seeds':[17,23,31]}
        self.host=Atoms('Zn',positions=[[0,0,0]],cell=[20,20,20],pbc=True)
        self.guest=Atoms('OCO',positions=[[0,0,-1.16],[0,0,0],[0,0,1.16]])

    def test_intramolecular_distances_do_not_trigger_wall(self):
        gas=self.guest.copy();gas.translate([5,5,5])
        self.assertTrue(allowed(self.host+gas,1,radii_array(self.config)))

    def test_guest_guest_and_host_guest_clashes_are_rejected(self):
        self.assertFalse(allowed(self.host+self.guest,1,radii_array(self.config)))
        a=self.guest.copy();a.translate([5,5,5]);b=a.copy();b.translate([1.2,0,0])
        self.assertFalse(allowed(self.host+a+b,1,radii_array(self.config)))

    def test_interaction_guard_aborts_instead_of_modifying_energy(self):
        calc=guarded_calculator(ArtificialEnergy(),self.host,self.guest,-2.)
        state=self.host+self.guest;state.calc=calc
        with self.assertRaisesRegex(RuntimeError,'run aborted'):
            state.get_potential_energy()

    def test_finite_job_budget(self):
        self.assertEqual(len(jobs(self.config,'scan')),3)
        self.assertEqual(len(jobs(self.config,'smoke')),3)
        self.assertEqual(len(jobs(self.config,'pilot')),9)
        self.assertEqual(len({j['id'] for j in jobs(self.config,'pilot')}),9)

    def test_single_seed_pressure_jobs_are_distinct(self):
        config={**self.config,'schema_version':2,'pressures_bar':[.1,.5,1.,5.,10.],'seeds':[17]}
        self.assertEqual(len(jobs(config,'scan')),3)
        self.assertEqual(len(jobs(config,'smoke')),15)
        pilot=jobs(config,'pilot')
        self.assertEqual(len(pilot),15)
        self.assertEqual(len({job['id'] for job in pilot}),15)
        self.assertEqual({job['seed'] for job in pilot},{17})
        runtime={'fixture':'runtime'}
        with TemporaryDirectory() as temp:
            root=Path(temp);dump(root/'bundle_manifest.json',{'fixture':True})
            self.assertNotEqual(identity(root,'pilot',pilot[0],runtime),identity(root,'pilot',pilot[1],runtime))

    def test_invalid_pressure_seed_or_budget_is_rejected(self):
        budget={'equilibration_steps':20,'production_steps':40,'checkpoint_interval':10}
        validate_sampling([.1,1,10],[17],budget)
        for pressures,seeds,gcmc in [([1,1],[17],budget),([0],[17],budget),
                ([float('nan')],[17],budget),([1],[17,17],budget),([1],[],budget),
                ([1],[True],budget),([1],[17],{**budget,'production_steps':0})]:
            with self.assertRaises(ValueError):validate_sampling(pressures,seeds,gcmc)

    def test_single_seed_report_keeps_pressures_separate_without_seed_ci(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'ref.csv').write_text('10000,0.1\n100000,1.0\n')
            config={**self.config,'schema_version':2,'pressures_bar':[.1,1.],'seeds':[17],
                    'reference':{'path':'ref.csv'},'precision':{'absolute_floor_mmol_g':.02,'relative_seed_ci95_target':.2}}
            dump(root/'bundle_manifest.json',{'fixture':True})
            runtime={'fixture':'runtime'}
            for index,job in enumerate(jobs(config,'pilot')):
                folder=root/'results/pilot'/job['id'];folder.mkdir(parents=True)
                result={'loading_mmol_g':float(index+1),'pressure_bar':job['pressure_bar'],
                        'screen':{'status':'passes_operational_screen','effective_sample_count_estimate':100.,'half_difference_mmol_g':0.}}
                dump(folder/'result.json',result)
                dump(folder/'completed.json',{'identity':identity(root,'pilot',job,runtime),
                    'result_path':'result.json','result_sha256':digest(folder/'result.json')})
            with patch('modules.module_c_mlips.guarded_pilot.check',return_value=config):report(root)
            result=json.loads((root/'report/assessment.json').read_text())
            self.assertEqual(result['expected_jobs'],6)
            self.assertEqual(result['completed_jobs'],6)
            self.assertEqual(result['missing'],[])
            self.assertEqual(len(result['points']),6)
            for index,point in enumerate(result['points']):
                self.assertEqual(point['mean'],index+1)
                self.assertEqual(point['reference_mmol_g'],point['pressure_bar'])
                self.assertEqual(point['replicates'],1)
                self.assertIsNone(point['ci95_half_width'])
                self.assertFalse(point['between_seed_uncertainty_available'])
                self.assertTrue(point['chain_screens_passed'])
                self.assertFalse(point['all_screens_passed'])

    def test_missing_failed_and_changed_gate_results_block_pilot(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);dump(root/'bundle_manifest.json',{'fixture':True})
            runtime={'fixture':'runtime'}
            with self.assertRaisesRegex(RuntimeError,'Missing'):
                gate(root,self.config,'pilot',runtime)
            for stage in ('scan','smoke'):
                for job in jobs(self.config,stage):
                    folder=root/'results'/stage/job['id'];folder.mkdir(parents=True)
                    dump(folder/'result.json',{'gate_passed':True})
                    dump(folder/'completed.json',{'identity':identity(root,stage,job,runtime),
                         'result_path':'result.json','result_sha256':digest(folder/'result.json')})
            gate(root,self.config,'pilot',runtime)
            with self.assertRaises(ValueError):gate(root,self.config,'pilot',{'fixture':'other'})
            folder=root/'results/scan/mh_s17'
            dump(folder/'result.json',{'gate_passed':False})
            with self.assertRaisesRegex(ValueError,'checksum'):gate(root,self.config,'pilot',runtime)
            marker=json.loads((folder/'completed.json').read_text());marker['result_sha256']=digest(folder/'result.json')
            dump(folder/'completed.json',marker)
            with self.assertRaisesRegex(RuntimeError,'failed'):gate(root,self.config,'pilot',runtime)

    def test_generic_campaign_cannot_silently_replace_protocol(self):
        plan={'benchmark':{'potential_benchmark':{'execution_entrypoint':'modules.module_c_mlips.guarded_pilot'}}}
        with self.assertRaisesRegex(ValueError,'frozen guarded pilot'):_run_adsorption(plan,'gcmc_pilot')


if __name__=='__main__':unittest.main()
