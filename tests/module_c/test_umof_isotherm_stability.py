"""Operational stability checks must not turn short or frozen histories into evidence."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from modules.module_c_mlips.umof_isotherm_stability import assess, WINDOW, MAX_EXTRA, CHUNK, make_report


class StabilityScreenTests(unittest.TestCase):
    def test_short_or_frozen_trajectory_cannot_pass(self):
        for values in [np.ones(2000),np.ones(20000),np.zeros(20000)]:
            self.assertEqual(assess(values)['status'],'not_resolved')

    def test_stationary_independent_history_passes_operational_screen(self):
        values=np.random.default_rng(12345).normal(5.,.5,WINDOW)
        result=assess(values)
        self.assertEqual(result['status'],'passes_operational_screen')
        self.assertGreater(result['batch_count'],8)
        self.assertTrue(result['criteria']['half_drift_within_tolerance'])

    def test_drift_is_rejected_even_with_many_samples(self):
        values=np.random.default_rng(12).normal(0.,.01,WINDOW)+np.linspace(1.,4.,WINDOW)
        result=assess(values)
        self.assertEqual(result['status'],'not_resolved')
        self.assertFalse(result['criteria']['half_drift_within_tolerance'])

    def test_correlations_reduce_effective_sample_count(self):
        rng=np.random.default_rng(42)
        noise=rng.normal(size=WINDOW)
        values=np.empty(WINDOW);values[0]=0.
        for i in range(1,WINDOW):values[i]=.98*values[i-1]+noise[i]
        correlated=assess(values+100.)
        independent=assess(noise+100.)
        self.assertLess(correlated['effective_sample_count_estimate'], independent['effective_sample_count_estimate']/10)

    def test_extension_budget_is_finite(self):
        self.assertEqual(MAX_EXTRA,20000)
        self.assertEqual(MAX_EXTRA//CHUNK,2)

    def test_final_report_withholds_ranking_when_a_point_is_unresolved(self):
        # Synthetic reporting fixtures only; no physical adsorption claims.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=root/'DDEC_IRMOF-1_UFF_CO2_298.csv'
            reference.write_text('10000,1,.1\n100000,2,.1\n1000000,3,.1\n')
            models=[{'name':'original'},{'name':'finetuned'}]
            jobs=[]
            for model in models:
                for pressure,mean in zip([.1,1.,10.],[1.,2.,3.]):
                    name=model['name']+str(pressure)
                    jobs.append({'name':name})
                    directory=root/'points'/name;directory.mkdir(parents=True)
                    values=np.random.default_rng(42).normal(mean,.01,WINDOW)
                    screen=assess(values)
                    if name=='finetuned10.0':screen['status']='not_resolved'
                    np.savez_compressed(directory/'history.npz',loading=values)
                    record={'model':model['name'],'pressure_bar':pressure,'loading_mol_per_kg':float(values.mean()),
                            'loading_std_mol_per_kg':float(values.std()),'screen':screen,
                            'evaluation_window_start_step':1,'evaluation_window_end_step':WINDOW,
                            'total_steps':WINDOW,'source_total_steps':10000,'source_runtime_seconds':1.,
                            'additional_runtime_seconds':1.,'history_file':str(directory/'history.npz')}
                    (directory/'point_result.json').write_text(json.dumps(record))
            manifest={'jobs':jobs,'config':{'benchmark':{'potential_benchmark':{'campaign':{'models':models}}}},'scope':'test fixture'}
            with patch('modules.module_c_mlips.umof_isotherm_stability.REFERENCE',reference):
                result=make_report(root,manifest)
            self.assertEqual(result['status'],'inconclusive_sampling')
            self.assertEqual(len(result['unresolved_points']),1)
            for name in ['RESULT.md','isotherm_comparison.png','loading_histories.png','runtime_details.csv','final_assessment.json']:
                self.assertTrue((root/'comparison'/name).is_file(),name)
