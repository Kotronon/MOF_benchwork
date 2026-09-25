from __future__ import annotations

import importlib.util
import math
import os
import unittest

import numpy as np

from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.mace import MaceBackend


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_MACE = importlib.util.find_spec("mace") is not None
RUN_MACE_INTEGRATION = os.environ.get("RUN_MACE_INTEGRATION") == "1"


if HAS_ASE:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class DeterministicCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self) -> None:
            super().__init__()
            self.calculation_count = 0

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            self.calculation_count += 1
            atom_count = len(self.atoms)
            self.results = {
                "energy": float(atom_count),
                "forces": np.full((atom_count, 3), 0.25),
            }


@unittest.skipUnless(HAS_ASE, "ASE is required for MACE backend tests.")
class MaceBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        atoms = Atoms(
            symbols=["Zn", "O", "C", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [3.0, 3.0, 2.0],
                [3.0, 3.0, 3.2],
                [3.0, 3.0, 4.4],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        self.configuration = InteractionConfiguration(
            configuration_id="test_mof_co2",
            material="test_mof",
            adsorbate="CO2",
            atoms=atoms,
            framework_indices=[0],
            adsorbate_indices=[1, 2, 3],
        )

    def test_returns_potential_result_from_injected_calculator(self) -> None:
        calculator = DeterministicCalculator()
        backend = MaceBackend(model="test-model", calculator=calculator)

        result = backend.evaluate(self.configuration)

        self.assertEqual(result.energy_ev, 4.0)
        self.assertEqual(result.forces_ev_per_angstrom, [[0.25] * 3] * 4)
        self.assertGreaterEqual(result.runtime_seconds, 0.0)
        self.assertEqual(result.metadata["backend"], "mace_mp")
        self.assertEqual(result.metadata["model"], "test-model")
        self.assertEqual(result.metadata["atom_count"], 4)
        self.assertIsNone(self.configuration.atoms.calc)
        self.assertEqual(calculator.calculation_count, 1)

    def test_reuses_injected_calculator_and_ase_cache(self) -> None:
        calculator = DeterministicCalculator()
        backend = MaceBackend(calculator=calculator)

        backend.evaluate(self.configuration)
        backend.evaluate(self.configuration)

        self.assertIs(backend.calculator, calculator)
        self.assertEqual(calculator.calculation_count, 1)

    def test_rejects_invalid_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "default_dtype"):
            MaceBackend(default_dtype="float16")


@unittest.skipUnless(
    HAS_ASE and HAS_MACE and RUN_MACE_INTEGRATION,
    "Set RUN_MACE_INTEGRATION=1 in an environment with MACE to run this test.",
)
class RealMaceBackendTests(unittest.TestCase):
    def test_cached_small_model_evaluates_host_guest_interaction(self) -> None:
        from modules.module_c_mlips.datasets import build_smoke_configurations
        from modules.module_c_mlips.interaction import evaluate_interaction

        configuration = build_smoke_configurations(
            "CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif",
            "CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def",
        )[0]
        backend = MaceBackend(
            model="small",
            device="cpu",
            default_dtype="float32",
        )

        result = evaluate_interaction(configuration, backend)

        self.assertTrue(math.isfinite(result.interaction_energy_ev))
        self.assertEqual(len(result.interaction_forces_ev_per_angstrom), 109)
        self.assertTrue(
            all(
                math.isfinite(component)
                for force in result.interaction_forces_ev_per_angstrom
                for component in force
            )
        )
        self.assertEqual(result.backend, "mace_mp")
        self.assertEqual(result.combined.metadata["atom_count"], 109)
        self.assertEqual(result.framework.metadata["atom_count"], 106)
        self.assertEqual(result.adsorbate.metadata["atom_count"], 3)
        self.assertGreaterEqual(
            result.combined.metadata["model_load_seconds"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
