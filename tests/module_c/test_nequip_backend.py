from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from modules.module_c_mlips.potential_backends.calculators import (
    build_ase_calculator,
    build_interaction_direct_calculator,
)

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    import numpy as np
except ImportError:
    Atoms = None


class NequipCalculatorFactoryTests(unittest.TestCase):
    def test_legacy_loader_receives_explicit_mapping_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "model.pth"
            model.write_bytes(b"model")
            expected = object()

            class FakeCalculator:
                @classmethod
                def from_deployed_model(cls, **arguments):
                    self.assertEqual(arguments["model_path"], str(model))
                    self.assertEqual(arguments["device"], "cuda")
                    self.assertEqual(arguments["species_to_type_name"], {"C": "C"})
                    self.assertEqual(arguments["energy_units_to_eV"], 1.0)
                    return expected

            module = types.SimpleNamespace(NequIPCalculator=FakeCalculator)
            with patch(
                "modules.module_c_mlips.potential_backends.calculators.importlib.import_module",
                return_value=module,
            ):
                result = build_ase_calculator(
                    {
                        "backend": "nequip",
                        "model": str(model),
                        "loader": "legacy",
                        "device": "cuda",
                        "species_to_type_name": {"C": "C"},
                    }
                )

            self.assertIs(result, expected)

    def test_finetuned_nequip_rejects_additional_d3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory) / "model.pth"
            model.write_bytes(b"model")

            with self.assertRaisesRegex(ValueError, "Do not add D3"):
                build_ase_calculator(
                    {
                        "backend": "nequip",
                        "model": str(model),
                        "dispersion": True,
                    }
                )

    def test_missing_model_has_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "NequIP model file"):
            build_ase_calculator(
                {
                    "backend": "nequip",
                    "model": "missing-model.pth",
                    "loader": "legacy",
                }
            )

    @unittest.skipIf(Atoms is None, "ASE is required for calculator adapter test.")
    def test_interaction_direct_adapter_zeroes_only_baselines(self) -> None:
        class ConstantCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(
                self,
                atoms=None,
                properties=("energy",),
                system_changes=all_changes,
            ) -> None:
                if atoms.calc is not None:
                    raise NotImplementedError("Foreign calculator on model input")
                super().calculate(atoms, properties, system_changes)
                self.results = {
                    "energy": -0.25,
                    "forces": np.ones((len(atoms), 3)),
                }

        adapter = build_interaction_direct_calculator(
            ConstantCalculator(),
            framework_atom_count=2,
            adsorbate_atom_count=3,
        )
        framework = Atoms("Zn2", positions=[[0, 0, 0], [1, 0, 0]])
        adsorbate = Atoms("CO2", positions=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        combined = framework + adsorbate
        combined.calc = adapter

        self.assertEqual(adapter.get_potential_energy(framework), 0.0)
        self.assertEqual(adapter.get_potential_energy(adsorbate), 0.0)
        self.assertAlmostEqual(adapter.get_potential_energy(combined), -0.25)
        self.assertTrue(np.allclose(adapter.get_forces(combined), 1.0))
        self.assertIs(combined.calc, adapter)


if __name__ == "__main__":
    unittest.main()
