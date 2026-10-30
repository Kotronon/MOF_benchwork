"""Physical invariants for the start-ready adsorption comparison."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from modules.module_c_mlips.umof_adsorption_comparison import (
    DEFAULT_CONFIG, consistent_calculator, engine_classes, fingerprint,
    get_completed, jobs, load_config, log_acceptance, production_diagnostics,
    widom_pose_hash,
)
from modules.module_c_mlips.umof_finetuning import sha256_file


class PoseDependentCalculator(Calculator):
    """A periodic/orientation artifact must not redefine the gas reference."""
    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -10.*len(atoms)+float(atoms.pbc.any())+float(atoms.positions[:, 0].sum())}


class AdsorptionComparisonTests(unittest.TestCase):
    def test_metropolis_forward_reverse_ratio_includes_fugacity(self):
        delta, beta, prefactor = -.35, 38., .002
        forward = log_acceptance(delta, beta, prefactor)
        reverse = log_acceptance(-delta, beta, 1/prefactor)
        self.assertAlmostEqual(forward-reverse, math.log(prefactor)-beta*delta)

    def test_very_favorable_energy_does_not_override_vanishing_fugacity(self):
        _, cls = engine_classes(.8)
        engine = cls.__new__(cls)
        engine.V, engine.beta, engine.fugacity, engine.Z_ads = 1., 10., 1e-200, 1
        engine.moves = {"insertion": {"accepted": 0}, "deletion": {"accepted": 0}}
        with patch("numpy.random.rand", return_value=.5):
            self.assertFalse(engine._insertion_acceptance(-20., 0.))
            self.assertTrue(engine._deletion_acceptance(0., -20.))

    def test_gas_reference_is_pose_and_periodicity_independent(self):
        gas = Atoms("OCO", positions=[[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
        base = PoseDependentCalculator()
        calc = consistent_calculator(base, gas)
        translated = gas.copy()
        translated.rotate(45, "y"); translated.translate([5, 3, 2])
        translated.set_cell([20, 20, 20]); translated.pbc = True; translated.calc = calc
        self.assertEqual(translated.get_potential_energy(), calc.gas_energy_eV)
        host = Atoms("MgO", positions=[[0, 0, 0], [2, 0, 0]], cell=[20]*3, pbc=True)
        combined = host+translated
        combined.calc = calc
        value = combined.get_potential_energy()
        direct = combined.copy(); direct.calc = base
        self.assertEqual(value, direct.get_potential_energy())
        translated.positions[0, 0] += .1
        with self.assertRaisesRegex(ValueError, "rigid CO2"):
            translated.get_potential_energy()

    def test_rejected_mc_steps_contribute_to_production_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"transitions.bin"
            with path.open("wb") as handle:
                for step, uptake in [(1, 1), (3, 2)]:
                    handle.write(struct.pack("iiddi", step, uptake, -uptake, -uptake, 10))
            result = production_diagnostics(path, equilibration=1, production=4)
            self.assertEqual(result["production_block_means_molecules"], [1., 2., 2., 2.])
            self.assertEqual(result["first_half_mean_molecules"], 1.5)
            self.assertEqual(result["second_half_mean_molecules"], 2.)

    def test_pose_hash_ignores_energy_but_detects_different_insertions(self):
        with tempfile.TemporaryDirectory() as temporary:
            hashes = []
            for name, energy, position in [("a", -1., 2.), ("b", 3., 2.), ("c", -1., 2.1)]:
                path = Path(temporary)/name
                with path.open("wb") as handle:
                    handle.write(struct.pack("iddi", 4, energy, energy, 1))
                    handle.write(struct.pack("i", 6))
                    handle.write(struct.pack("3d", position, 0, 0))
                    handle.write(struct.pack("9d", *np.eye(3).ravel()))
                hashes.append(widom_pose_hash(path))
            self.assertEqual(hashes[0], hashes[1])
            self.assertNotEqual(hashes[0], hashes[2])

    def test_completed_result_rejects_changed_input_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = directory/"result.json"; result.write_text('{"status":"completed"}')
            config = {"trials": 2000}
            marker = {"config_sha256": fingerprint(config), "result_path": "result.json", "result_sha256": sha256_file(result)}
            (directory/"completed.json").write_text(json.dumps(marker))
            self.assertEqual(get_completed(directory, config)["status"], "completed")
            with self.assertRaisesRegex(ValueError, "Changed settings"):
                get_completed(directory, {"trials": 3000})
            result.write_text('{}')
            with self.assertRaisesRegex(ValueError, "result changed"):
                get_completed(directory, config)

    def test_prepared_budget_is_one_seed_and_42000_total_steps(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["seeds"], [12345])
        self.assertEqual(config["gcmc"]["equilibration_steps"], 1000)
        self.assertEqual(config["gcmc"]["production_steps"], 2000)
        self.assertEqual(config["widom"]["trials"], 2000)
        all_jobs = list(jobs(config, Path("output")))
        self.assertEqual(len(all_jobs), 15)
        self.assertEqual([j[2] for j in all_jobs[:3]], ["widom"]*3)
        self.assertEqual(len({j[-1] for j in all_jobs}), 15)
        total = sum(config["widom"]["trials"] if j[2] == "widom" else 3000 for j in all_jobs)
        self.assertEqual(total, 42000)


if __name__ == "__main__":
    unittest.main()
