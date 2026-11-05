from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ase import Atoms
import benchmark
from analysis.module_d_flexibility import evaluate_flexibility, structural_metrics, summarize_cycles
from modules.module_d_flexibility import materialize_flexibility, prepare_flexibility, verify_reference_inputs
from modules.registry import module_by_id

CONFIG = Path('input_json_files/benchmark_module_d_nboffive_water.json')


def config():
    return json.loads(CONFIG.read_text())


def small_plan():
    data = config()
    data['conditions']['relative_humidity'] = [0.8]
    data['simulation']['seeds'] = [12345]
    data['benchmark']['flexibility'].update(sr_cycles=4, convergence_window=2)
    return benchmark.build_run_plan(data)


def measured_rows(plan, rigid=1.0, flexible=2.0):
    return [dict(case_id=case['case_id'], cycle=i, uptake_mol_kg=rigid if case['variant'] == 'rigid' else flexible,
                 framework_rmsd_A=0.1, cell_volume_A3=1000.0, potential_energy_eV=-30.0,
                 pore_volume_A3=None, linker_rotation_deg=None, runtime_s=3.0)
            for case in plan['cases'] for i in range(1, case['expected_cycles'] + 1)]


class ModuleDPlanningTests(unittest.TestCase):
    def test_paper_plan_is_independent_of_crafted_resolution(self):
        data = config()
        original = deepcopy(data)
        with patch('pipeline.planning.resolve_benchmark', side_effect=AssertionError('must not resolve CRAFTED')):
            plan = benchmark.build_run_plan(data)
        self.assertEqual(data, original)
        self.assertEqual(plan['module']['id'], 'D')
        self.assertEqual(len(plan['cases']), 60)
        self.assertEqual(len({c['case_id'] for c in plan['cases']}), 60)
        self.assertAlmostEqual(plan['state_points'][8]['pressure_bar'], 0.8 * 0.05197)
        self.assertIsNone(plan['state_points'][8]['fugacity_bar'])
        self.assertFalse(plan['benchmark']['applicability']['can_attempt_simulation'])
        self.assertTrue(plan['benchmark']['metrics'])
        self.assertTrue(any(a['status'] == 'missing' for a in plan['benchmark']['applicability']['assets']))

    def test_explicit_fugacity_is_not_replaced_by_pressure(self):
        data = config()
        reservoir = data['benchmark']['flexibility']['reservoir']
        reservoir.update(fugacities_bar=[0.001] * 10, fugacity_source='test table')
        point = benchmark.build_run_plan(data)['state_points'][0]
        self.assertEqual(point['fugacity_bar'], 0.001)
        self.assertNotEqual(point['fugacity_bar'], point['pressure_bar'])
        del reservoir['fugacity_source']
        with self.assertRaisesRegex(ValueError, 'fugacity_source'):
            benchmark.build_run_plan(data)

    def test_invalid_thermodynamic_inputs_fail(self):
        for rh in ([80], [0], [-0.1], [float('nan')], [True], [0.1, 0.1], []):
            with self.subTest(rh=rh):
                data = config()
                data['conditions']['relative_humidity'] = rh
                with self.assertRaises(ValueError):
                    benchmark.build_run_plan(data)
        data = config()
        data['conditions']['temperature_K'] = 298.15
        with self.assertRaisesRegex(ValueError, '300 K'):
            benchmark.build_run_plan(data)
        data = config()
        data['benchmark']['flexibility']['reservoir']['model_saturation_pressure_bar'] = 0.036
        with self.assertRaisesRegex(ValueError, '5197 Pa'):
            benchmark.build_run_plan(data)

    def test_protocol_boundaries_fail_explicitly(self):
        mutations = [
            lambda d: d['adsorbates'].update(components=['CO2']),
            lambda d: d['material'].update(name='MOF-5'),
            lambda d: d['material'].update(charge_scheme='auto'),
            lambda d: d['simulation'].update(framework='rigid'),
            lambda d: d['simulation'].update(engine='RASPA'),
            lambda d: d['benchmark']['flexibility']['md'].update(ensemble='NPT'),
            lambda d: d['benchmark']['flexibility'].update(sr_cycles=10),
            lambda d: d['benchmark']['flexibility'].update(discard_cycles=90),
            lambda d: d['benchmark']['flexibility']['md'].update(steps=1.5),
            lambda d: d['benchmark']['flexibility'].update(assets=[]),
            lambda d: d['output'].update(run_id='../other'),
        ]
        for mutate in mutations:
            data = config()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                benchmark.build_run_plan(data)

    def test_original_inputs_have_verified_checksums_and_water_tail_override(self):
        manifest = verify_reference_inputs()
        self.assertEqual(manifest['commit'], '7c8ddb138b5cda0a53c55b38906261d998397e47')
        overrides = Path('data/module_d/mofaff_reference/example/gcmc/force_field.def').read_text()
        self.assertRegex(overrides, r'Ow\s+Ow\s+truncated\s+yes')

    def test_prepare_preserves_results_and_resume_verifies_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = config()
            data['output'].update(directory=str(Path(tmp) / 'campaign'), run_id=None)
            plan = benchmark.build_run_plan(data)
            prepared = prepare_flexibility(plan)
            self.assertFalse(Path(prepared['working_directory']).exists())
            result = materialize_flexibility(prepared)
            self.assertFalse(result['simulation_started'])
            root = Path(result['working_directory'])
            self.assertFalse((root / 'cycle_results.csv').exists())
            with self.assertRaises(FileExistsError):
                materialize_flexibility(prepared)
            marker = root / 'cycle_results.csv'
            marker.write_text('user results remain untouched')
            data['output']['resume'] = True
            resumed = prepare_flexibility(benchmark.build_run_plan(data))
            self.assertEqual(materialize_flexibility(resumed)['status'], 'reused_plan')
            self.assertEqual(marker.read_text(), 'user results remain untouched')
            (root / 'upstream_reference/example/in.lammps').write_text('modified input')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                materialize_flexibility(resumed)

    def test_resume_rejects_changed_case_table_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = config()
            data['output'].update(directory=str(Path(tmp) / 'campaign'), run_id=None)
            materialize_flexibility(prepare_flexibility(benchmark.build_run_plan(data)))
            data['output']['resume'] = True
            data['conditions']['relative_humidity'] = [0.8]
            with self.assertRaises(FileExistsError):
                materialize_flexibility(prepare_flexibility(benchmark.build_run_plan(data)))
            data['conditions']['relative_humidity'] = config()['conditions']['relative_humidity']
            (Path(tmp) / 'campaign/cases.csv').write_text('case_id\nwrong\n')
            with self.assertRaisesRegex(ValueError, 'cases.csv'):
                materialize_flexibility(prepare_flexibility(benchmark.build_run_plan(data)))

    def test_asset_changes_between_plan_and_prepare_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'charges.txt'
            path.write_text('first version')
            data = config()
            data['output'].update(directory=str(Path(tmp) / 'campaign'), run_id=None)
            data['benchmark']['flexibility']['assets']['charges'] = str(path)
            prepared = prepare_flexibility(benchmark.build_run_plan(data))
            path.write_text('second version')
            with self.assertRaisesRegex(ValueError, 'Asset changed'):
                materialize_flexibility(prepared)
            self.assertFalse(Path(prepared['working_directory']).exists())

    def test_cli_is_read_only_by_default_for_module_d(self):
        with patch('benchmark.write_generated_registries') as registry, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(benchmark.main([str(CONFIG), '--dry-run']), 0)
        registry.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())['module']['id'], 'D')

    def test_simulation_is_blocked_even_with_override(self):
        for flag in ('--run-test', '--run-isotherm'):
            with self.subTest(flag=flag), self.assertRaisesRegex(NotImplementedError, 'execution is unavailable'):
                benchmark.main([str(CONFIG), flag, '--allow-unsupported', '--skip-registry-update'])
        self.assertEqual(module_by_id('D').metadata()['id'], 'D')


class ModuleDAnalysisTests(unittest.TestCase):
    def test_paired_comparison_uses_complete_cases(self):
        plan = small_plan()
        result = summarize_cycles(plan, measured_rows(plan))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['validation_status'], 'not_validated')
        self.assertEqual(result['comparisons'][0]['adsorption_change_mol_kg'], 1.0)
        self.assertEqual(result['comparisons'][0]['relative_adsorption_change'], 1.0)
        sr = result['cases'][1]
        self.assertEqual(sr['stationarity'], 'passed')
        self.assertEqual(sr['mean_cell_volume_A3'], 1000.0)
        self.assertIsNone(sr['mean_pore_volume_A3'])
        self.assertEqual(sr['total_runtime_s'], 12.0)

    def test_partial_data_do_not_create_complete_comparisons(self):
        plan = small_plan()
        result = summarize_cycles(plan, measured_rows(plan)[:-1])
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['comparisons'], [])
        self.assertEqual(result['cases'][1]['stationarity'], 'insufficient_cycles')
        self.assertEqual(summarize_cycles(plan, [])['status'], 'partial')

    def test_drift_is_reported_as_failed(self):
        plan = small_plan()
        rows = measured_rows(plan)
        for row in rows[-2:]:
            row['uptake_mol_kg'] = 10
        result = summarize_cycles(plan, rows)
        self.assertEqual(result['cases'][1]['stationarity'], 'failed')
        self.assertEqual(result['comparisons'][0]['stationarity'], 'failed')

    def test_zero_rigid_loading_has_no_relative_error(self):
        plan = small_plan()
        result = summarize_cycles(plan, measured_rows(plan, rigid=0))
        self.assertIsNone(result['comparisons'][0]['relative_adsorption_change'])
        json.dumps(result, allow_nan=False)

    def test_malformed_measurements_fail(self):
        plan = small_plan()
        for column, value in [('cycle', 1.5), ('cycle', 0), ('case_id', 'unknown'),
                              ('uptake_mol_kg', float('nan')), ('framework_rmsd_A', -1),
                              ('cell_volume_A3', 0), ('runtime_s', float('inf'))]:
            rows = measured_rows(plan)
            rows[0][column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                summarize_cycles(plan, rows)
        rows = measured_rows(plan)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize_cycles(plan, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
            summarize_cycles(plan, rows[:2] + rows[3:])
        rows[-1]['cycle'] = 5
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            summarize_cycles(plan, rows)

    def test_discard_cycles_affects_mean_only_for_flexible_branch(self):
        plan = small_plan()
        plan['benchmark']['flexibility']['discard_cycles'] = 1
        rows = measured_rows(plan)
        rows[1]['uptake_mol_kg'] = 100
        result = summarize_cycles(plan, rows)
        self.assertEqual(result['cases'][0]['mean_uptake_mol_kg'], 1)
        self.assertEqual(result['cases'][1]['mean_uptake_mol_kg'], 2)
        self.assertEqual(result['cases'][1]['retained_cycles'], 3)

    def test_evaluation_round_trip_does_not_modify_raw_results(self):
        plan = small_plan()
        rows = measured_rows(plan)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'module_d_plan.json').write_text(json.dumps(plan))
            raw = root / 'cycle_results.csv'
            with raw.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            original = raw.read_bytes()
            result = evaluate_flexibility(root)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(raw.read_bytes(), original)
            self.assertTrue((root / 'evaluation/rigid_vs_flexible.csv').is_file())
            raw.write_text('wrong_header\n')
            with self.assertRaisesRegex(ValueError, 'requires columns'):
                evaluate_flexibility(root)

    def test_geometry_handles_wrapping_and_affine_cell_change(self):
        ref = Atoms('CO', positions=[[0.1, 1, 1], [2, 2, 2]], cell=[10, 10, 10], pbc=True)
        current = ref.copy()
        current.positions[0, 0] += 10
        self.assertAlmostEqual(structural_metrics(ref, current, [0, 1])['framework_rmsd_A'], 0)
        current = ref.copy()
        current.set_cell([11, 11, 11], scale_atoms=True)
        result = structural_metrics(ref, current, [0, 1])
        self.assertAlmostEqual(result['framework_rmsd_A'], 0)
        self.assertAlmostEqual(result['relative_cell_volume_change'], 0.331)
        self.assertIsNone(result['pore_volume_A3'])

    def test_geometry_detects_displacement_and_excludes_guests(self):
        ref = Atoms('COH', positions=[[1, 1, 1], [2, 2, 2], [3, 3, 3]], cell=[10]*3, pbc=True)
        current = ref.copy()
        current.positions[0, 0] += 0.2
        current.positions[2] += 3
        self.assertAlmostEqual(structural_metrics(ref, current, [0, 1])['framework_rmsd_A'], 0.2 / 2**0.5)
        with self.assertRaises(ValueError):
            structural_metrics(ref, current, [0, 0])
        current.numbers[0] = 7
        with self.assertRaisesRegex(ValueError, 'species/order'):
            structural_metrics(ref, current, [0, 1])

    def test_linker_rotation_is_periodic_and_validated(self):
        ref = Atoms('CCCC', positions=[[1, 2, 1], [1, 1, 1], [2, 1, 1], [2, 2, 1]], cell=[10]*3, pbc=True)
        current = ref.copy()
        current.positions[3] = [2, 1, 2]
        metrics = structural_metrics(ref, current, [0, 1, 2, 3], [(0, 1, 2, 3)])
        self.assertAlmostEqual(abs(metrics['linker_rotation_deg'][0]), 90)
        with self.assertRaises(ValueError):
            structural_metrics(ref, current, [0, 1, 2, 3], [(0, 1, 2, 2)])


if __name__ == '__main__':
    unittest.main()
