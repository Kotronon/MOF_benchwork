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
)


class UmofFinetuningTests(unittest.TestCase):
    def test_preparation_filters_gases_and_holds_out_mofs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "prepared"
            data.mkdir()
            for mof in ("TrainMOF", "ValidMOF", "TestMOF"):
                for split in ("train", "valid", "test"):
                    write(
                        data / f"{mof}-adsorbates-{split}.extxyz",
                        [
                            _frame("CO2"),
                            _frame("N2"),
                            _frame("CH4"),
                            _frame("H2O"),
                        ],
                        format="extxyz",
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

    def test_training_command_uses_umof_mh_head_and_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "umof.model"
            checkpoint.write_bytes(b"checkpoint")
            command = build_umof_mh_training_command(
                {
                    "train_file": str(root / "train.extxyz"),
                    "valid_file": str(root / "valid.extxyz"),
                },
                checkpoint,
                root / "output",
                max_epochs=2,
            )

        self.assertIn("--foundation_head", command)
        self.assertEqual(command[command.index("--foundation_head") + 1], "r2scan+d4")
        self.assertIn("--foundation_model_elements", command)
        self.assertEqual(
            command[command.index("--foundation_model_elements") + 1], "True"
        )
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(command[command.index("--lr") + 1], "0.0001")
        self.assertEqual(command[command.index("--multiheads_finetuning") + 1], "False")
        self.assertEqual(command[command.index("--max_num_epochs") + 1], "2")


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