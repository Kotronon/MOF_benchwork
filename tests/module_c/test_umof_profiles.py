"""Profile isolation, portable inference and POLAR replay regression tests."""
from pathlib import Path
from collections import Counter
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import read, write

from modules.module_c_mlips import umof_profiles as p
from modules.module_c_mlips import umof_workflow as w
from modules.module_c_mlips import umof_finetuning as d
from test_umof_finetuning import _frame


class ProfileTests(unittest.TestCase):
    def test_python_api_uses_profile_defaults(self):
        with patch.object(w, 'prepare_campaign') as prepare, patch.object(w, 'execute_campaign', return_value={}) as execute:
            d.run_finetuning_matrix(model='polar')
            self.assertEqual(prepare.call_args.args[1], p.profile('polar').checkpoint)
            self.assertEqual(prepare.call_args.args[2], p.profile('polar').output_directory)
            self.assertEqual(prepare.call_args.kwargs['model'], 'polar')
            self.assertIsNone(prepare.call_args.kwargs['replay_source'])
            execute.assert_called_once_with(p.profile('polar').output_directory, 'train')
    def test_existing_mh_prediction_cache_is_reused(self):
        import hashlib
        import importlib.metadata
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('dataset', 'model'): (root/name).write_text(name)
            identity = [d.sha256_file(root/'dataset'), d.sha256_file(root/'model'),
                        'r2scan+d4', 'float64', 'cpu', importlib.metadata.version('mace-torch')]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            np.savez_compressed(root/f'{key}.npz', energies=[1.], forces=np.zeros((1, 3)))
            with patch.object(w, 'calculator') as calculator:
                energies, forces = w.predict(root/'dataset', root/'model', 'r2scan+d4', root)
                calculator.assert_not_called()
                self.assertEqual(energies[0], 1.)

    def test_smoke_cannot_reuse_other_model_or_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); production = root/'production'; smoke = root/'smoke'
            w.atomic_json(production/'campaign.json', {'protocol': {'purpose': 'production'}, 'files': {}})
            for model in ('mh', 'polar'):
                w.atomic_json(smoke/'campaign.json', {'protocol': {'purpose': 'smoke', 'model': model},
                                                     'production_source_sha256': 'other-source'})
                with self.assertRaisesRegex(ValueError, 'Smoke source/model changed'):
                    w.smoke_campaign(production, smoke)

    def test_profile_selection_and_architecture(self):
        self.assertEqual(p.campaign_profile({'protocol': {}}).key, 'mh')
        self.assertNotEqual(p.profile('mh').output_directory, p.profile('polar').output_directory)
        for model in ('mh', 'polar'):
            profile = p.profile(model)
            profile.validate({'class': profile.model_type, 'heads': ['r2scan+d4']})
            with self.assertRaises(ValueError):
                profile.validate({'class': 'Other', 'heads': ['r2scan+d4']})
        with self.assertRaises(ValueError):
            p.profile('polar').validate({'class': 'PolarMACE', 'heads': ['other']})

    def test_cli_defaults_and_explicit_model_conflict(self):
        with patch.object(w, 'status_campaign', return_value={}) as status:
            w.main(['status', '--model', 'polar'])
            self.assertEqual(status.call_args.args[0], p.profile('polar').output_directory)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            w.atomic_json(root/'campaign.json', {'protocol': {'model': 'polar'}})
            for action in ('status', 'prepare', 'train', 'resume', 'evaluate', 'report', 'smoke', 'setup'):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, '--model conflicts'):
                    w.main([action, '--output-directory', tmp, '--model', 'mh'])
            with patch.object(w, 'status_campaign', return_value={}) as status:
                w.main(['status', '--output-directory', tmp])
                self.assertEqual(status.call_args.args[0].resolve(), root)

    def test_training_commands_use_local_profile_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); (root/'assets/replay-1').mkdir(parents=True)
            for split in ('train', 'valid'):
                (root/f'assets/replay-1/{split}.extxyz').write_text('fixture')
            for model in ('mh', 'polar'):
                profile = p.profile(model); (root/profile.asset).write_text('checkpoint fixture')
                campaign = {'protocol': w.protocol(model=model, max_epochs=50, learning_rate=1e-4,
                    replay_weight=.25, replay_samples=1000, device='cpu')}
                cmd = w.training_command(root, campaign, {'directory': 'run', 'seed': 1})
                self.assertEqual(cmd[cmd.index('--model')+1], profile.model_type)
                self.assertEqual(cmd[cmd.index('--name')+1], profile.name)
                self.assertEqual(cmd[cmd.index('--foundation_model')+1], str(root/profile.asset))
                self.assertEqual(cmd[cmd.index('--real_pt_data_ratio_threshold')+1], '0.0')
                self.assertIn(str(root/'assets/replay-1/train.extxyz'), cmd)
                self.assertNotIn('--test_file', cmd)

    def test_profiles_and_foundation_hashes_change_training_identity(self):
        run = {'directory': 'run', 'seed': 1}
        c = {'protocol': w.protocol(model='polar', max_epochs=2), 'files': {
            p.profile('polar').asset: 'one', 'run/datasets/test.extxyz': 'test'}}
        old = w.training_identity(c, run)
        c['files']['run/datasets/test.extxyz'] = 'changed test labels'
        self.assertEqual(old, w.training_identity(c, run))
        c['files'][p.profile('polar').asset] = 'two'
        self.assertNotEqual(old, w.training_identity(c, run))
        with self.assertRaises(ValueError):
            w.compare_protocol(w.protocol(model='mh', max_epochs=2), w.protocol(model='polar', max_epochs=3))

    def test_polar_conditioning_defaults_aliases_and_conflicts(self):
        atoms = Atoms('H2')
        p.polar_conditioning(atoms)
        self.assertEqual(atoms.info['charge'], atoms.info['total_charge'])
        self.assertEqual(atoms.info['spin'], 1)
        self.assertIn('not a reference', atoms.info['umof_polar_conditioning'])
        for info in ({'spin': 2, 'total_spin': 1}, {'charge': float('nan')}, {'spin': 0}):
            atoms = Atoms('H2'); atoms.info.update(info)
            with self.assertRaises(ValueError): p.polar_conditioning(atoms)
        atoms = Atoms('H2'); atoms.info['spin'] = 3
        p.polar_conditioning(atoms)
        self.assertEqual(atoms.info['total_spin'], 3)

    def test_polar_replay_excludes_framework_composition_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); frames = []
            for i, symbols in enumerate(('H2', 'He2', 'Li2', 'Be2')):
                atoms = Atoms(symbols, positions=[[0, 0, 0], [1+i*.1, 0, 0]])
                atoms.info['dft_energy'] = -float(i+1)
                atoms.set_array('dft_forces', np.zeros((2, 3)))
                frames.append(atoms)
            write(root/'source.extxyz', frames)
            excluded = {p.reduced_composition(Counter('HHHH'))}
            manifest = w.prepare_replay(root/'source.extxyz', root/'replay', seed=1, count=2,
                valid_count=1, model='polar', excluded_compositions=excluded)
            self.assertEqual(manifest['reference'], 'uMOF r2SCAN-D4 DFT')
            self.assertIn('not restored OMol', manifest['replay_kind'])
            self.assertEqual(len(set(manifest['geometry_ids'])), 3)
            for split in ('train', 'valid'):
                for atoms in read(root/f'replay/{split}.extxyz', index=':'):
                    self.assertNotIn('H', atoms.get_chemical_symbols())
                    self.assertEqual(atoms.info['REF_energy'], atoms.info['dft_energy'])
                    self.assertEqual(atoms.info['head'], 'pt_head')
                    self.assertEqual(atoms.info['config_weight'], .25)
                    self.assertEqual(atoms.info['total_spin'], atoms.info['spin'])

    def test_export_preserves_polar_type_after_directory_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); (root/'model').write_bytes(b'fixture')
            spec = {'model': 'model', 'model_type': 'PolarMACE', 'sha256': d.sha256_file(root/'model'),
                    'head': 'Default', 'default_dtype': 'float64'}
            w.atomic_json(root/'module_c_model.json', spec)
            info = {'class': 'PolarMACE', 'dtype': 'float64', 'heads': ['Default', 'pt_head']}
            with patch.object(w, 'model_info', return_value=info):
                result = w.load_model_specification(root/'module_c_model.json')
                self.assertEqual(result['model_type'], 'PolarMACE')
                self.assertEqual(result['model'], str(root/'model'))
            with patch.object(w, 'model_info', return_value=dict(info, **{'class': 'ScaleShiftMACE'})):
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    w.load_model_specification(root/'module_c_model.json')

    def test_test_labels_do_not_affect_polar_replay_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); manifest = {'training_data_fraction': 1.0}
            for split in ('train', 'valid', 'test'):
                a = d._prepare_frame(_frame('CO2'), Path('Fixture'), split)
                write(root/f'{split}.extxyz', a)
                manifest[split+'_file'] = str(root/f'{split}.extxyz')
            before = p.adsorption_framework_compositions([manifest])
            a = read(root/'test.extxyz'); a.info['REF_energy'] = 999
            a.set_array('REF_forces', np.ones((len(a), 3))*999); write(root/'test.extxyz', a)
            self.assertEqual(before, p.adsorption_framework_compositions([manifest]))

    def test_preparation_transfer_and_same_adsorption_frames(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source = root/'source'; source.mkdir()
            for i, mof in enumerate(('Train', 'Valid', 'Test')):
                for j, split in enumerate(('train', 'valid', 'test')):
                    a = _frame('CO2'); a.positions[0, 2] += .02*(i*3+j)
                    write(source/f'{mof}-adsorbates-{split}.extxyz', [a])
            replay = []
            for i in range(4):
                a = _frame('CO2'); a.positions[0, 2] += 1+i*.02; replay.append(a)
            write(source/'replay.extxyz', replay)
            datasets = {}
            for model in ('mh', 'polar'):
                checkpoint = root/p.profile(model).filename; checkpoint.write_bytes(model.encode())
                info = {'class': p.profile(model).model_type, 'heads': ['r2scan+d4'],
                    'sha256': d.sha256_file(checkpoint), 'atomic_numbers': [1, 6, 7, 8, 30], 'dtype': 'float64'}
                with patch.object(w, 'model_info', return_value=info), patch.object(w, 'environment', return_value={'packages': {}}):
                    campaign = w.prepare_campaign(source, checkpoint, root/model, fractions=[1.0],
                        validation_mofs={'Valid-adsorbates'}, test_mofs={'Test-adsorbates'},
                        replay_source=source/'replay.extxyz', replay_samples=2, replay_valid_count=1,
                        purpose='fixture', model=model)
                folder = root/model/campaign['runs'][0]['directory']
                datasets[model] = read(folder/'datasets/train.extxyz', index=':')
                if model == 'polar':
                    manifest = w.read_json(folder/'datasets/dataset_manifest.json')
                    self.assertEqual(manifest['replay_dataset'], 'umof_r2scan_d4_gaussian_noise_train')
                moved = root/(model+'-moved'); shutil.copytree(root/model, moved)
                w.verify_files(moved, campaign['files'])
                command = w.training_command(moved, campaign, campaign['runs'][0])
                self.assertIn(str(moved/p.profile(model).asset), command)
                self.assertNotIn(str(root/model/p.profile(model).asset), command)
                (moved/p.profile(model).asset).write_bytes(b'tampered')
                with self.assertRaises(ValueError): w.verify_files(moved, campaign['files'])
            for mh, polar in zip(datasets['mh'], datasets['polar']):
                self.assertEqual(mh.info['umof_frame_id'], polar.info['umof_frame_id'])
                self.assertEqual(mh.info['REF_energy'], polar.info['REF_energy'])
                np.testing.assert_array_equal(mh.arrays['REF_forces'], polar.arrays['REF_forces'])
