"""Fixtures below test reporting logic only; they are not adsorption references."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.module_c_mlips import umof_isotherm_comparison as runner
from modules.module_c_mlips.campaign import build_campaign_tasks
from pipeline.config import normalize_config


class UmofIsothermComparisonTests(unittest.TestCase):
    def test_protocol_matches_models_and_covers_three_seeds_and_pressures(self):
        config = runner.configure()
        tasks = build_campaign_tasks(normalize_config(config), stage=runner.STAGE)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(config['conditions']['pressures_bar'], [.1, 1., 10.])
        models = config['benchmark']['potential_benchmark']['campaign']['models']
        self.assertEqual([m['head'] for m in models], ['r2scan+d4', 'Default'])
        for key in ['device', 'default_dtype', 'dispersion', 'model_type']:
            self.assertEqual(models[0][key], models[1][key])
        self.assertEqual(len({t['seed'] for t in tasks}), 3)

    def test_smoke_is_separate_and_does_not_change_full_protocol(self):
        full = runner.configure()
        smoke = runner.configure(True)
        self.assertNotEqual(full['benchmark']['potential_benchmark']['campaign']['id'],
                            smoke['benchmark']['potential_benchmark']['campaign']['id'])
        self.assertEqual(len(smoke['simulation']['seeds']), 1)
        self.assertEqual(len(smoke['conditions']['pressures_bar']), 1)
        self.assertEqual(runner.configure(), full)

    def test_quick_single_seed_has_18000_moves_and_distinct_outputs(self):
        config = runner.configure(quick=True, single_seed=True)
        campaign = config['benchmark']['potential_benchmark']['campaign']
        tasks = build_campaign_tasks(normalize_config(config), stage=runner.STAGE)
        settings = campaign['stages'][runner.STAGE]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(config['simulation']['seeds'], [104729])
        count = len(tasks)*len(config['conditions']['pressures_bar'])*(settings['equilibration_steps']+settings['production_steps'])
        self.assertEqual(count, 18000)
        self.assertTrue(campaign['id'].endswith('_quick_single_seed'))

    def test_ten_bar_protocol_is_longer_single_seed_and_separate(self):
        config = runner.configure(ten_bar=True)
        tasks = build_campaign_tasks(normalize_config(config), stage=runner.STAGE)
        campaign = config['benchmark']['potential_benchmark']['campaign']
        self.assertEqual(config['conditions']['pressures_bar'], [10.])
        self.assertEqual(config['simulation']['seeds'], [104731])
        self.assertEqual(len(tasks), 2)
        self.assertTrue(campaign['id'].endswith('_10bar_single_seed'))
        settings = campaign['stages'][runner.STAGE]
        self.assertEqual((settings['equilibration_steps'], settings['production_steps']), (10000, 20000))
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            runner.configure(quick=True, ten_bar=True)
        self.assertEqual(runner.configure(smoke=True, ten_bar=True)['conditions']['pressures_bar'], [10.])

    def test_ten_bar_flag_is_forwarded_to_both_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = MagicMock()
            process.poll.return_value = 0
            with patch.object(runner.subprocess, 'Popen', return_value=process) as start:
                runner.run_tasks([{'task_id': str(i)} for i in range(2)], root, root,
                                 ten_bar=True, gpus=['0', '3'])
            self.assertEqual(start.call_count, 2)
            self.assertTrue(all('--ten-bar' in call.args[0] for call in start.call_args_list))

    def test_parallel_workers_get_distinct_gpus_and_matching_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [{'task_id': str(i)} for i in range(2)]
            launched = []
            def start(command, **kwargs):
                launched.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                process = MagicMock()
                def finished():
                    self.assertEqual(len(launched), 2)
                    return 0
                process.poll.side_effect = finished
                return process
            with patch.object(runner.subprocess, 'Popen', side_effect=start):
                runner.run_tasks(tasks, root, root, quick=True, single_seed=True, gpus=['0', '3'])
            self.assertEqual([gpu for _, gpu in launched], ['0', '3'])
            self.assertTrue(all('--quick' in cmd and '--single-seed' in cmd for cmd, _ in launched))

    def test_gpu_parser_rejects_duplicate_or_empty_devices(self):
        import argparse
        self.assertEqual(runner.parse_gpus('0,3'), ['0', '3'])
        for value in ['0,0', '', '0,', '-1']:
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_gpus(value)

    def fixture_report(self, root, offsets):
        matched = []
        for model, delta in zip(('umof_mh_pretrained', 'umof_mh_finetuned_025pct'), offsets):
            for pressure in (.1, 1., 10.):
                matched.append({'model': model, 'pressure_bar': pressure,
                                'reference_loading_mol_per_kg': 1.})
            for seed in (1, 2, 3):
                folder = root/f'{model}_{seed}'/'results'
                folder.mkdir(parents=True)
                result = {'model': model, 'seed': seed,
                          'pressure_points': [{'pressure_bar': p, 'loading_mol_per_kg': 1.+delta}
                                              for p in (.1, 1., 10.)]}
                (folder/'gcmc_pilot.json').write_text(json.dumps(result))
        return {'run_directory': str(root), 'reference_comparison': {'matched_points': matched}}

    def test_pilot_trend_uses_negative_difference_to_favor_finetuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.summarize_pilot(self.fixture_report(root, [2., 1.]), root)
            result = json.loads((root/'pilot_assessment.json').read_text())
            self.assertEqual(result['trend'], 'lower_error_to_classical_reference')
            self.assertAlmostEqual(result['paired_seed_rmse_difference_mol_per_kg'], -1.)
            self.assertTrue(result['limitations'])

    def test_equal_models_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.summarize_pilot(self.fixture_report(root, [1., 1.]), root)
            result = json.loads((root/'pilot_assessment.json').read_text())
            self.assertEqual(result['trend'], 'inconclusive')

    def test_single_seed_report_has_no_confidence_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.fixture_report(root, [2., 1.])
            for path in root.glob('*/results/gcmc_pilot.json'):
                if json.loads(path.read_text())['seed'] != 1:
                    path.unlink()
            runner.summarize_pilot(report, root)
            result = json.loads((root/'pilot_assessment.json').read_text())
            self.assertEqual(result['trend'], 'descriptive_only_single_seed')
            self.assertIsNone(result['ci95'])

    def test_missing_seed_rejects_scientific_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.fixture_report(root, [2., 1.])
            (root/'umof_mh_finetuned_025pct_3/results/gcmc_pilot.json').unlink()
            with self.assertRaisesRegex(ValueError, 'matching sets'):
                runner.summarize_pilot(report, root)
