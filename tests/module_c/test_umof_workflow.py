"""Synthetic structures are fixtures only, never production or smoke inputs."""
from pathlib import Path
import copy
import json
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.io import read, write

from modules.module_c_mlips import umof_finetuning as d
from modules.module_c_mlips import umof_workflow as w
from test_umof_finetuning import _frame


def mixture(components=('CO2', 'N2'), explicit=True):
    atoms = _frame('CO2')[:2]
    rows = []
    for i, component in enumerate(components):
        guest = _frame(component)[2:]
        guest.positions[:, 1] += i * 5
        indices = list(range(len(atoms), len(atoms) + len(guest)))
        atoms += guest
        rows.append({'component_id': component, 'atom_indices': indices})
    atoms.info['dft_energy'] = -10.
    atoms.set_array('dft_forces', np.zeros((len(atoms), 3)))
    if explicit:
        atoms.info['umof_guest_assignments'] = json.dumps(rows)
        atoms.info['umof_framework_indices'] = '[0,1]'
    return atoms


class CompositionTests(unittest.TestCase):
    def prepare(self, atoms, **kwargs):
        return d._prepare_frame(atoms, Path('Fixture-adsorbates-train.extxyz'), 'train', **kwargs)

    def test_binary_ternary_and_integer_counts(self):
        for gases in [('CO2', 'N2'), ('CO2', 'CO2', 'N2', 'CH4')]:
            a = self.prepare(mixture(gases))
            self.assertEqual(a.info['umof_configuration_kind'], 'mixture')
            self.assertEqual(json.loads(a.info['umof_guest_counts']), {g: gases.count(g) for g in gases})
            self.assertEqual(a.info['umof_composition_basis'], 'adsorbed_molecule_counts')
            self.assertAlmostEqual(sum(json.loads(a.info['umof_gas_composition']).values()), 1)

    def test_component_order_invariant_nested_selection(self):
        frames = [mixture(c) for c in [('CO2', 'N2'), ('CH4', 'N2', 'CO2'), ('CO2',)*3 + ('N2',)]] * 9
        reordered = [a.copy() for a in frames]
        for a in reordered:
            a.info['umof_guest_assignments'] = json.dumps(list(reversed(json.loads(a.info['umof_guest_assignments']))))
        previous = set()
        for fraction in (.1, .25, .5, 1.):
            ids = d._stratified_nested_indices(frames, train_fraction=fraction, rng=random.Random(12345))
            other = d._stratified_nested_indices(reordered, train_fraction=fraction, rng=random.Random(12345))
            self.assertEqual(ids, other)
            self.assertTrue(previous <= set(ids))
            previous = set(ids)

    def test_excluded_component_excludes_whole_configuration(self):
        self.assertIsNone(self.prepare(mixture(('CO2', 'H2O'))))

    def test_unknown_and_unassigned_guests_are_rejected(self):
        a = mixture()
        rows = json.loads(a.info['umof_guest_assignments'])
        rows[1]['component_id'] = 'unknown'
        a.info['umof_guest_assignments'] = json.dumps(rows)
        with self.assertRaises(d.GuestAssignmentError): self.prepare(a)
        a = mixture(); a.info['umof_framework_indices'] = '[0]'
        with self.assertRaises(d.GuestAssignmentError): self.prepare(a)

    def test_ambiguous_geometric_identity_requires_explicit_assignment(self):
        with self.assertRaisesRegex(d.GuestAssignmentError, 'Ambiguous'):
            self.prepare(mixture(explicit=False), gas_definitions={'CO2_alias': {'C': 1, 'O': 2}})
        self.assertIsNotNone(self.prepare(mixture(), gas_definitions={'CO2_alias': {'C': 1, 'O': 2}}))

    def test_unknown_geometric_guest_not_silently_dropped(self):
        from ase import Atoms
        a = mixture(explicit=False) + Atoms('He', positions=[[20, 20, 20]])
        a.info['dft_energy'] = -10
        a.set_array('dft_forces', np.zeros((len(a), 3)))
        with self.assertRaises(d.GuestAssignmentError): self.prepare(a)

    def test_missing_and_nonfinite_labels_fail(self):
        a = mixture(); del a.info['dft_energy']
        with self.assertRaises(ValueError): self.prepare(a)
        a = mixture(); a.arrays['dft_forces'][0, 0] = np.nan
        with self.assertRaises(ValueError): self.prepare(a)

    def test_contradictory_metadata_and_float_counts_fail(self):
        for field, value in [('umof_guest_counts', '{"CO2": 1.0, "N2": 1}'),
                             ('umof_guest_counts', '{"CO2": 2, "N2": 1}'),
                             ('umof_component_ids', '["CO2"]'),
                             ('umof_gas_composition', '{"CO2": 0.8, "N2": 0.2}'),
                             ('umof_configuration_kind', 'pure_gas'),
                             ('umof_composition_basis', 'gas_phase_mole_fractions')]:
            a = mixture(); a.info[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.prepare(a)

    def test_stress_preserved_without_training_weight(self):
        a = mixture(); a.info['dft_stress'] = np.arange(9).reshape(3, 3)
        a.info['stress_convention'] = 'source fixture convention'
        b = self.prepare(a)
        np.testing.assert_array_equal(a.info['dft_stress'], b.info['dft_stress'])
        self.assertEqual(b.info['stress_convention'], a.info['stress_convention'])
        self.assertEqual(b.info['config_stress_weight'], 0.)

    def test_metrics_group_whole_mixtures(self):
        frames = [self.prepare(mixture()), self.prepare(_frame('CO2'))]
        energy, forces, _ = w.reference_arrays(frames)
        metrics = w.grouped_metrics(frames, (energy, forces))
        self.assertEqual(metrics['components:CO2']['configuration_count'], 1)
        self.assertEqual(metrics['components:CO2+N2']['configuration_count'], 1)


class LifecycleTests(unittest.TestCase):
    def test_missing_replay_fails_before_model_loading_with_setup_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(w, 'model_info') as model_info:
                with self.assertRaisesRegex(FileNotFoundError, 'setup --download-missing'):
                    w.prepare_campaign(root / 'data', root / 'model', root / 'campaign',
                                       replay_source=root / 'missing.extxyz')
                model_info.assert_not_called()
            self.assertFalse((root / 'campaign' / 'assets').exists())

    def test_replay_own_labels_weights_disjoint_and_no_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'replay.extxyz'
            frames = []
            for i in range(5):
                a = _frame('CO2'); a.positions[0, 0] += i * .01
                a.info['energy'] = 9999
                frames.append(a)
            write(source, frames)
            manifest = w.prepare_replay(source, root/'selected', seed=12345, count=3, valid_count=2)
            train = read(root/'selected/train.extxyz', index=':')
            valid = read(root/'selected/valid.extxyz', index=':')
            self.assertEqual(len(set(manifest['geometry_ids'])), 5)
            for a in train + valid:
                self.assertEqual(a.info['REF_energy'], -10)
                self.assertEqual(a.info['head'], 'pt_head')
                self.assertEqual(a.info['config_weight'], .25)
            with self.assertRaisesRegex(ValueError, 'no duplicate padding'):
                w.prepare_replay(source, root/'bad', seed=12345, count=4, valid_count=2)

    def test_lock_contention_atomic_json_and_readonly_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with w.campaign_lock(root):
                with self.assertRaises(RuntimeError):
                    with w.campaign_lock(root): pass
            w.atomic_json(root/'state.json', {'status': 'prepared'})
            (root/'state.json').chmod(0o444)
            w.verify_files(root, {'state.json': d.sha256_file(root/'state.json')})
            self.assertFalse(list(root.glob('*.tmp')))

    def test_transfer_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'old'; source.mkdir()
            (source/'data').write_text('data')
            hashes = {'data': d.sha256_file(source/'data')}
            shutil.copytree(source, root/'transferred')
            w.verify_files(root/'transferred', hashes)
            (root/'transferred/data').write_text('altered')
            with self.assertRaises(ValueError): w.verify_files(root/'transferred', hashes)

    def test_epoch_increase_only_with_identical_schedule(self):
        a = w.protocol(max_epochs=2)
        w.compare_protocol(a, dict(a, max_epochs=3))
        for changed in [dict(a, max_epochs=1), dict(a, max_epochs=3, start_swa=2),
                        dict(a, max_epochs=3, dtype='float32')]:
            with self.assertRaises(ValueError): w.compare_protocol(a, changed)

    def test_test_labels_do_not_change_training_identity(self):
        run = {'directory': '010pct/seed-12345', 'seed': 12345}
        campaign = {'protocol': w.protocol(max_epochs=2), 'files': {
            '010pct/seed-12345/datasets/train.extxyz': 'training',
            '010pct/seed-12345/datasets/valid.extxyz': 'valid',
            '010pct/seed-12345/datasets/test.extxyz': 'test labels 1'}}
        before = w.training_identity(campaign, run)
        campaign['files']['010pct/seed-12345/datasets/test.extxyz'] = 'test labels 2'
        self.assertEqual(before, w.training_identity(campaign, run))
        campaign['files']['010pct/seed-12345/datasets/train.extxyz'] = 'changed'
        self.assertNotEqual(before, w.training_identity(campaign, run))

    def test_checkpoint_sidecar_rejects_changed_or_untracked_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'checkpoints').mkdir()
            p = root/'checkpoints/model_epoch-1.pt'; p.write_bytes(b'weights')
            with self.assertRaises(ValueError): w.resume_checkpoint(root, 'identity')
            w.atomic_json(str(p)+'.manifest.json', {'sha256': d.sha256_file(p), 'training_identity': 'identity', 'epoch': 1})
            self.assertEqual(w.resume_checkpoint(root, 'identity'), str(p))
            p.write_bytes(b'altered')
            with self.assertRaises(ValueError): w.resume_checkpoint(root, 'identity')

    def test_validation_selection_never_reads_test_and_tie_prefers_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'run'; (folder/'datasets').mkdir(parents=True)
            (folder/'models').mkdir()
            frame = d._prepare_frame(mixture(), Path('fixture'), 'valid')
            write(folder/'datasets/valid.extxyz', [frame])
            for suffix in ('', '_stagetwo'):
                (folder/'models'/f'{w.NAME}{suffix}.model').write_bytes(b'fixture')
            campaign = {'protocol': {'device': 'cpu'}}; run = {'directory': 'run'}
            info = {'heads': ['Default', 'pt_head'], 'sha256': 'fixture'}
            prediction = (np.array([-10.]), frame.arrays['REF_forces'])
            with patch.object(w, 'model_info', return_value=info), patch.object(w, 'predict', return_value=prediction) as predict:
                a = w.select_model(root, campaign, run)
                (folder/'datasets/test.extxyz').write_text('Invalid test labels are not a selection input')
                b = w.select_model(root, campaign, run)
                self.assertEqual(a, b); self.assertEqual(a['stage'], 'stage1')
                self.assertTrue(all(call.args[0].name == 'valid.extxyz' for call in predict.call_args_list))

    def test_float32_selection_and_export_keep_campaign_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'run'
            (folder/'datasets').mkdir(parents=True); (folder/'models').mkdir()
            frame = _frame('CO2')
            frame.info['REF_energy'] = -10.
            frame.set_array('REF_forces', np.zeros((len(frame), 3)))
            write(folder/'datasets/valid.extxyz', [frame])
            (folder/'models/umof_mh_adsorption_finetuned.model').write_text('fixture')
            campaign = {'protocol': {'device': 'cuda', 'dtype': 'float32'}}
            info = {'heads': ['Default', 'pt_head'], 'sha256': 'fixture', 'dtype': 'float32'}
            prediction = (np.array([-10.]), frame.arrays['REF_forces'])
            with patch.object(w, 'model_info', return_value=info), patch.object(w, 'predict', return_value=prediction) as predict:
                w.select_model(root, campaign, {'directory': 'run'})
                self.assertEqual(predict.call_args.kwargs['dtype'], 'float32')
            self.assertEqual(w.read_json(folder/'module_c_model.json')['default_dtype'], 'float32')
            with patch.object(w, 'model_info', return_value=dict(info, dtype='float64')):
                with self.assertRaisesRegex(ValueError, 'datatype'):
                    w.select_model(root, campaign, {'directory': 'run'})

    def test_interrupted_evaluation_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'run'; folder.mkdir()
            (root/'model').write_bytes(b'model')
            run = {'directory': 'run', 'seed': 12345}
            campaign = {'protocol': w.protocol(max_epochs=3), 'files': {}, 'runs': [run]}
            w.atomic_json(root/'campaign.json', campaign)
            w.atomic_json(folder/'status.json', {'training_identity': w.training_identity(campaign, run),
                'trained_to_limit': 3, 'selected_model': {'path': 'model', 'sha256': d.sha256_file(root/'model')}, 'status': 'failed'})
            with patch.object(w.subprocess, 'run') as train, patch.object(w, 'evaluate_run'), patch.object(w, 'report_campaign'):
                w.execute_campaign(root, 'resume')
                train.assert_not_called()
            self.assertEqual(w.read_json(folder/'status.json')['status'], 'completed')

    def test_resume_device_override_preserves_archived_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'run'; folder.mkdir()
            (root/'model').write_bytes(b'model')
            run = {'directory': 'run', 'seed': 12345}
            campaign = {'protocol': w.protocol(max_epochs=3), 'files': {}, 'runs': [run]}
            identity = w.training_identity(campaign, run)
            w.atomic_json(root/'campaign.json', campaign)
            w.atomic_json(folder/'status.json', {'training_identity': identity,
                'trained_to_limit': 3, 'selected_model': {'path': 'model',
                'sha256': d.sha256_file(root/'model')}, 'status': 'failed'})
            with patch.object(w, 'resume_checkpoint', return_value=None) as checkpoint, \
                    patch.object(w, 'evaluate_run') as evaluate, patch.object(w, 'report_campaign', return_value={}):
                w.main(['resume', '--output-directory', tmp, '--device', 'cuda'])
                checkpoint.assert_called_with(folder, identity)
                self.assertEqual(evaluate.call_args.args[1]['protocol']['device'], 'cuda')
            self.assertEqual(w.read_json(root/'campaign.json'), campaign)
            self.assertEqual(w.read_json(folder/'status.json')['training_identity'], identity)

    def test_resume_rejects_explicit_protocol_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            w.atomic_json(Path(tmp)/'campaign.json', {'protocol': {'learning_rate': .0001}})
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                w.main(['resume', '--output-directory', tmp, '--learning-rate', '.1'])

class SplitAndRuntimeTests(unittest.TestCase):
    def make_sources(self, root):
        for index, mof in enumerate(('Train', 'Valid', 'Test')):
            for split in ('train', 'valid', 'test'):
                frame = _frame('CO2'); frame.positions[:, 2] += index * .1
                write(root/f'{mof}-adsorbates-{split}.extxyz', [frame])

    def test_overlapping_mof_splits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_sources(root)
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                d.prepare_adsorption_datasets(root, root/'out', validation_mofs={'Valid-adsorbates'}, test_mofs={'Valid-adsorbates'})

    def test_unsupported_gas_and_mof_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_sources(root)
            options = dict(validation_mofs={'Valid-adsorbates'}, test_mofs={'Test-adsorbates'})
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                d.prepare_adsorption_datasets(root, root/'out', gases={'N2'}, supported_elements={1, 6, 8}, **options)
            path = root/'Train-adsorbates-train.extxyz'
            frame = read(path); frame.info['umof_mof_id'] = 'Wrong'
            write(path, frame)
            with self.assertRaisesRegex(ValueError, 'Contradictory MOF'):
                d.prepare_adsorption_datasets(root, root/'out', **options)

    def test_custom_splits_and_gas_definitions_reach_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(w, 'prepare_campaign', return_value={}) as prepare:
                w.main(['prepare', '--output-directory', str(root), '--validation-mofs', 'CustomV',
                        '--test-mofs', 'CustomT', '--gases', 'N2'])
            self.assertEqual(prepare.call_args.kwargs['validation_mofs'], {'CustomV'})
            self.assertEqual(prepare.call_args.kwargs['test_mofs'], {'CustomT'})
            self.assertEqual(prepare.call_args.kwargs['gases'], {'N2'})

    def test_actual_mace_parser_uses_replay_reference_and_weight(self):
        from mace.data.utils import config_from_atoms, KeySpecification
        a = _frame('CO2')
        a.info.update(REF_energy=-12., energy=999., config_weight=.25, head='pt_head')
        a.set_array('REF_forces', np.ones((len(a), 3)))
        spec = KeySpecification(info_keys={'energy': 'REF_energy'}, arrays_keys={'forces': 'REF_forces'})
        config = config_from_atoms(a, key_specification=spec, head_name='pt_head')
        self.assertEqual(config.weight, .25)
        self.assertEqual(config.head, 'pt_head')
        self.assertEqual(config.properties['energy'], -12.)
        np.testing.assert_array_equal(config.properties['forces'], np.ones((len(a), 3)))
