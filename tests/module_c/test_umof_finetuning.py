from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ase import Atoms
from ase.io import read, write
import numpy as np

from modules.module_c_mlips.umof_finetuning import (
    build_umof_mh_training_command,
    prepare_adsorption_datasets,
    _target_gas_composition,
    _stratified_nested_indices,
)


class UmofFinetuningTests(unittest.TestCase):
    def test_preparation_filters_gases_and_holds_out_mofs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "prepared"
            data.mkdir()
            for mof_index, mof in enumerate(("TrainMOF", "ValidMOF", "TestMOF")):
                for split in ("train", "valid", "test"):
                    frames = [_frame(gas) for gas in ("CO2", "N2", "CH4", "H2O")]
                    for frame in frames:
                        frame.positions[:, 2] += mof_index * 0.1
                    write(
                        data / f"{mof}-adsorbates-{split}.extxyz",
                        frames,
                        format="extxyz",
                    )
                    # AppleDouble metadata from the release ZIP is not an XYZ dataset.
                    (data / f"._{mof}-adsorbates-{split}.extxyz").write_bytes(
                        b"\x00\x05\x16\x07\xa2"
                    )

            manifest = prepare_adsorption_datasets(
                data,
                output,
                validation_mofs={"ValidMOF-adsorbates"},
                test_mofs={"TestMOF-adsorbates"},
            )

            train = read(manifest["train_file"], index=":")
            valid = read(manifest["valid_file"], index=":")
            test = read(manifest["test_file"], index=":")
            self.assertEqual((len(train), len(valid), len(test)), (3, 3, 3))
            self.assertEqual(manifest["training_mofs"], ["TrainMOF-adsorbates"])
            self.assertEqual(manifest["validation_mofs"], ["ValidMOF-adsorbates"])
            self.assertEqual(manifest["test_mofs"], ["TestMOF-adsorbates"])
            self.assertFalse(manifest["test_files_used_for_training"])
            self.assertTrue(all("H2O" not in atoms.info["umof_gas_composition"] for atoms in train + valid + test))
            self.assertEqual(
                {atoms.info["umof_source_split"] for atoms in train}, {"train"}
            )
            self.assertEqual(
                {atoms.info["umof_source_split"] for atoms in valid}, {"valid"}
            )
            self.assertEqual(
                {atoms.info["umof_source_split"] for atoms in test}, {"test"}
            )

    def test_float32_training_command_and_short_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('model', 'replay', 'valid'):
                (root / name).write_text('fixture')
            options = dict(checkpoint=root/'model', output_directory=root/'out',
                           replay_dataset=str(root/'replay'), replay_valid_file=root/'valid',
                           max_epochs=5, start_swa=3, device='cuda')
            manifest = {'train_file': root/'train', 'valid_file': root/'valid'}
            command = build_umof_mh_training_command(manifest, dtype='float32', **options)
            self.assertEqual(command[command.index('--default_dtype') + 1], 'float32')
            self.assertEqual(command[command.index('--start_swa') + 1], '3')
            self.assertEqual(command[command.index('--max_num_epochs') + 1], '5')
            with self.assertRaisesRegex(ValueError, 'dtype'):
                build_umof_mh_training_command(manifest, dtype='float16', **options)

    def test_training_command_uses_foundation_replay_and_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "umof.model"
            checkpoint.write_bytes(b"checkpoint")
            replay = root / "replay.extxyz"
            replay.write_text("fixture")
            replay_valid = root / "valid.extxyz"
            replay_valid.write_text("fixture")
            command = build_umof_mh_training_command(
                {
                    "train_file": str(root / "train.extxyz"),
                    "valid_file": str(root / "valid.extxyz"),
                    "replay_dataset": "matpes_r2scan",
                },
                checkpoint,
                root / "output",
                max_epochs=2,
                replay_dataset=str(replay),
                replay_valid_file=replay_valid,
                start_swa=1,
            )

        self.assertIn("--foundation_head", command)
        self.assertEqual(command[command.index("--foundation_head") + 1], "r2scan+d4")
        self.assertIn("--foundation_model_elements", command)
        self.assertEqual(
            command[command.index("--foundation_model_elements") + 1], "True"
        )
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(command[command.index("--default_dtype") + 1], "float64")
        self.assertEqual(command[command.index("--batch_size") + 1], "1")
        self.assertEqual(command[command.index("--real_pt_data_ratio_threshold") + 1], "0.0")
        self.assertEqual(command[command.index("--weight_pt_head") + 1], "0.25")
        self.assertEqual(command[command.index("--pt_valid_file") + 1], str(replay_valid.resolve()))
        self.assertEqual(command[command.index("--compute_stress") + 1], "False")
        self.assertNotIn("--test_file", command)
        self.assertEqual(command[command.index("--lr") + 1], "0.0001")
        self.assertEqual(command[command.index("--swa_lr") + 1], "0.0001")
        self.assertEqual(command[command.index("--multiheads_finetuning") + 1], "True")
        self.assertEqual(command[command.index("--E0s") + 1], "foundation")
        self.assertEqual(command[command.index("--pt_train_file") + 1], str(replay.resolve()))
        self.assertIn("--swa", command)
        self.assertEqual(command[command.index("--max_num_epochs") + 1], "2")
        self.assertIn("--restart_latest", command)
        self.assertIn("--save_all_checkpoints", command)
        self.assertEqual(command[command.index("--seed") + 1], "12345")
        self.assertEqual(command[command.index("--pt_train_file") + 1], str(replay.resolve()))

    def test_stratified_fraction_is_nested_and_covers_each_stratum(self) -> None:
        frames = [
            _frame("CO2"),
            _frame("CO2"),
            _frame("N2"),
            _frame("N2"),
            _frame("CH4"),
            _frame("CH4"),
        ]

        selected_ten = _stratified_nested_indices(
            frames,
            train_fraction=0.5,
            rng=__import__("random").Random(17),
        )
        selected_full = _stratified_nested_indices(
            frames,
            train_fraction=1.0,
            rng=__import__("random").Random(17),
        )

        self.assertEqual(len(selected_ten), 3)
        self.assertTrue(set(selected_ten) <= set(selected_full))
        self.assertEqual(
            {
                gas
                for index in selected_ten
                for gas in _target_gas_composition(frames[index])
            },
            {"CO2", "N2", "CH4"},
        )


def _frame(gas: str) -> Atoms:
    host = Atoms(
        "ZnO",
        positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )
    guest_definitions = {
        "CO2": Atoms("CO2", positions=[[10.0, 0.0, 0.0], [11.16, 0.0, 0.0], [8.84, 0.0, 0.0]]),
        "N2": Atoms("N2", positions=[[10.0, 0.0, 0.0], [11.1, 0.0, 0.0]]),
        "CH4": Atoms("CH4", positions=[[10.0, 0.0, 0.0], [10.6, 0.6, 0.6], [10.6, -0.6, -0.6], [9.4, 0.6, -0.6], [9.4, -0.6, 0.6]]),
        "H2O": Atoms("H2O", positions=[[10.0, 0.0, 0.0], [10.76, 0.0, 0.59], [9.24, 0.0, 0.59]]),
    }
    atoms = host + guest_definitions[gas]
    atoms.set_cell(host.cell)
    atoms.set_pbc(False)
    atoms.info["dft_energy"] = -10.0
    atoms.new_array("dft_forces", np.zeros((len(atoms), 3)))
    return atoms


if __name__ == "__main__":
    unittest.main()
