from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from modules.module_c_mlips.classical_parity import (
    compare_classical_results,
    evaluate_module_a_static_files,
)
from modules.module_c_mlips.datasets import build_smoke_configurations
from modules.module_c_mlips.datasets import load_lammps_dump_configuration
from modules.module_c_mlips.datasets import parse_lammps_forcefield_type_map
from modules.module_c_mlips.interaction import (
    build_adsorbate_configuration,
    evaluate_interaction,
)
from modules.module_c_mlips.potential_backends.lammps_structure import (
    build_lammps_structure,
    write_lammps_data,
)
from converter.forcefield_to_lammps import (
    VIRTUAL_SITE_MASS_AMU,
    parse_pseudo_atoms,
)
from modules.module_c_mlips.potential_backends.base import PotentialResult
from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)
from pipeline.config import load_benchmark_data
from pipeline.materialization import materialize_benchmark
from pipeline.planning import build_run_plan
from pipeline.prepare import prepare_benchmark


HAS_ASE = importlib.util.find_spec("ase") is not None
HAS_COOLPROP = importlib.util.find_spec("CoolProp") is not None
HAS_LAMMPS = shutil.which("lmp") is not None
HAS_CRAFTED = Path("CRAFTED-2.0.0").is_dir()
CAN_RUN_REAL_PARITY = HAS_ASE and HAS_COOLPROP and HAS_LAMMPS and HAS_CRAFTED
MODULE_A_SNAPSHOT = Path(
    "outputs/module_A_co2_isotherm/work/dumps/gcmc_1bar.lammpstrj"
)
MODULE_A_SNAPSHOT_FORCEFIELD = Path(
    "outputs/module_A_co2_isotherm/work/forcefield/forcefield.in"
)
HAS_MODULE_A_SNAPSHOT = (
    MODULE_A_SNAPSHOT.is_file() and MODULE_A_SNAPSHOT_FORCEFIELD.is_file()
)

CIF_PATH = Path("CRAFTED-2.0.0/CIF_FILES/DDEC/IRMOF-1.cif")
CO2_PATH = Path("CRAFTED-2.0.0/FORCEFIELDS/UFF/CO2.def")
PSEUDO_ATOMS_PATH = Path(
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/pseudo_atoms.def"
)
MIXING_RULES_PATH = Path(
    "CRAFTED-2.0.0/FORCEFIELDS/UFF/force_field_mixing_rules.def"
)


def _result(
    energy_ev: float,
    forces: list[list[float]],
    components: dict[str, float],
) -> PotentialResult:
    return PotentialResult(
        energy_ev=energy_ev,
        forces_ev_per_angstrom=forces,
        runtime_seconds=0.0,
        metadata={"energy_components_kcal_per_mol": components},
    )


class ClassicalParityReportTests(unittest.TestCase):
    def test_identical_results_pass_and_are_json_serializable(self) -> None:
        reference = _result(
            -1.0,
            [[1.0, 2.0, 3.0]],
            {"potential_energy": -23.0, "vdw": -20.0},
        )

        report = compare_classical_results(reference, reference)

        self.assertTrue(report.passed)
        self.assertEqual(report.absolute_energy_difference_ev, 0.0)
        self.assertEqual(report.maximum_force_difference_ev_per_angstrom, 0.0)
        json.dumps(report.to_dict())

    def test_difference_larger_than_tolerance_fails(self) -> None:
        reference = _result(
            -1.0,
            [[0.0, 0.0, 0.0]],
            {"potential_energy": -23.0},
        )
        candidate = _result(
            -0.9,
            [[0.0, 0.0, 0.01]],
            {"potential_energy": -22.0},
        )

        report = compare_classical_results(
            reference,
            candidate,
            energy_tolerance_ev=1e-3,
            force_tolerance_ev_per_angstrom=1e-3,
        )

        self.assertFalse(report.passed)
        self.assertAlmostEqual(report.energy_difference_ev, 0.1)
        self.assertAlmostEqual(
            report.maximum_force_difference_ev_per_angstrom,
            0.01,
        )

    def test_mismatched_force_counts_are_rejected(self) -> None:
        reference = _result(0.0, [[0.0, 0.0, 0.0]], {})
        candidate = _result(0.0, [], {})

        with self.assertRaisesRegex(ValueError, "same number of forces"):
            compare_classical_results(reference, candidate)

    def test_mismatched_energy_components_are_rejected(self) -> None:
        reference = _result(
            0.0,
            [[0.0, 0.0, 0.0]],
            {"potential_energy": 0.0, "vdw": 0.0},
        )
        candidate = _result(
            0.0,
            [[0.0, 0.0, 0.0]],
            {"potential_energy": 0.0},
        )

        with self.assertRaisesRegex(ValueError, "same energy components"):
            compare_classical_results(reference, candidate)


@unittest.skipUnless(
    CAN_RUN_REAL_PARITY,
    "ASE, CoolProp, CRAFTED, and the lmp executable are required.",
)
class ClassicalModuleAParityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_directory.name)
        config = load_benchmark_data(
            "input_json_files/benchmark_mof5_co2_smoke_test.json"
        )
        config["simulation"]["cell_representation"] = "conventional"
        config["simulation"]["minimum_image_policy"] = "error"
        config["simulation"]["unit_cells"] = [1, 1, 1]
        config["simulation"]["kspace_accuracy"] = 1e-4
        config["output"].update(
            {
                "directory": str(cls.root / "module_a"),
                "run_id": "parity_reference",
                "overwrite": True,
                "save_dumps": False,
            }
        )
        run_plan = build_run_plan(config)
        cls.materialized = materialize_benchmark(prepare_benchmark(run_plan))
        cls.configurations = build_smoke_configurations(
            CIF_PATH,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
            cell_representation="conventional",
            cutoff_A=12.8,
            minimum_image_policy="error",
        )
        cls.configuration = cls.configurations[0]
        module_a_type_map = parse_lammps_forcefield_type_map(
            cls.materialized["files"]["forcefield_include"]
        )
        cls.module_a_type_ids = {
            label: type_id for type_id, label in module_a_type_map.items()
        }
        pseudo_atoms = parse_pseudo_atoms(PSEUDO_ATOMS_PATH)
        from ase.data import atomic_masses, atomic_numbers

        cls.module_a_type_masses = {}
        for label in cls.module_a_type_ids:
            pseudo_atom = pseudo_atoms.get(label)
            if pseudo_atom is not None:
                cls.module_a_type_masses[label] = (
                    pseudo_atom.mass
                    if pseudo_atom.mass > 0.0
                    else VIRTUAL_SITE_MASS_AMU
                )
            else:
                cls.module_a_type_masses[label] = float(
                    atomic_masses[atomic_numbers[label]]
                )
        cls.backend = ClassicalLAMMPSBackend(
            lammps_command="lmp",
            pseudo_atoms_file=PSEUDO_ATOMS_PATH,
            mixing_rules_file=MIXING_RULES_PATH,
            working_directory=cls.root / "module_c",
            pair_style="lj/cut/coul/long 12.8",
            kspace_style="pppm 0.0001",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_framework_energy_and_forces_match_module_a_files(self) -> None:
        from modules.module_c_mlips.interaction import (
            build_framework_configuration,
        )

        framework_configuration = build_framework_configuration(
            self.configuration
        )
        candidate = self.backend.evaluate(framework_configuration)
        reference = evaluate_module_a_static_files(
            data_file=self.materialized["files"]["framework_data"],
            forcefield_file=self.materialized["files"]["forcefield_include"],
            atom_count=len(framework_configuration.atoms),
            working_directory=self.root / "module_a_framework_run_zero",
            has_bonds=True,
        )

        report = compare_classical_results(
            reference,
            candidate,
            energy_tolerance_ev=1e-4,
            force_tolerance_ev_per_angstrom=1e-4,
        )

        self.assertTrue(report.passed, report.to_dict())

    def test_all_deterministic_configurations_match_module_a_forcefield(self) -> None:
        positive_configurations = [
            configuration
            for configuration in self.configurations
            if configuration.region != "technical_repulsive_probe"
        ]
        self.assertEqual(len(positive_configurations), 2)
        for configuration in positive_configurations:
            with self.subTest(configuration=configuration.configuration_id):
                candidate = self.backend.evaluate(configuration)
                module_c_run = (
                    self.root / "module_c" / configuration.configuration_id
                )
                reference = evaluate_module_a_static_files(
                    data_file=module_c_run / "combined.data",
                    forcefield_file=self.materialized["files"]["forcefield_include"],
                    atom_count=len(configuration.atoms),
                    working_directory=(
                        self.root
                        / f"module_a_{configuration.configuration_id}_run_zero"
                    ),
                    has_bonds=True,
                )

                report = compare_classical_results(
                    reference,
                    candidate,
                    energy_tolerance_ev=1e-8,
                    force_tolerance_ev_per_angstrom=1e-8,
                )

                self.assertTrue(report.passed, report.to_dict())

    def test_overlap_probe_is_rejected_as_non_finite(self) -> None:
        overlap_probe = next(
            configuration
            for configuration in self.configurations
            if configuration.region == "technical_repulsive_probe"
        )

        with self.assertRaisesRegex(ValueError, "energy_ev must be finite"):
            self.backend.evaluate(overlap_probe)

    def test_interaction_energy_and_forces_match_module_a_path(self) -> None:
        interaction = evaluate_interaction(self.configuration, self.backend)
        combined_directory = (
            self.root / "module_c" / self.configuration.configuration_id
        )
        reference_combined = evaluate_module_a_static_files(
            data_file=combined_directory / "combined.data",
            forcefield_file=self.materialized["files"]["forcefield_include"],
            atom_count=len(self.configuration.atoms),
            working_directory=self.root / "module_a_interaction_combined",
            has_bonds=True,
        )
        reference_framework = evaluate_module_a_static_files(
            data_file=self.materialized["files"]["framework_data"],
            forcefield_file=self.materialized["files"]["forcefield_include"],
            atom_count=len(self.configuration.framework_indices),
            working_directory=self.root / "module_a_interaction_framework",
            has_bonds=True,
        )
        adsorbate_configuration = build_adsorbate_configuration(
            self.configuration
        )
        adsorbate_structure = build_lammps_structure(
            adsorbate_configuration.atoms,
            adsorbate_configuration.framework_indices,
            adsorbate_configuration.adsorbate_indices,
            self.module_a_type_ids,
            adsorbate_configuration.bonds,
        )
        adsorbate_data = write_lammps_data(
            adsorbate_structure,
            self.root / "module_a_interaction_adsorbate" / "co2.data",
            type_masses=self.module_a_type_masses,
        )
        reference_adsorbate = evaluate_module_a_static_files(
            data_file=adsorbate_data,
            forcefield_file=self.materialized["files"]["forcefield_include"],
            atom_count=len(adsorbate_configuration.atoms),
            working_directory=self.root / "module_a_interaction_adsorbate",
            has_bonds=True,
        )
        reference_interaction_energy = (
            reference_combined.energy_ev
            - reference_framework.energy_ev
            - reference_adsorbate.energy_ev
        )
        self.assertAlmostEqual(
            interaction.interaction_energy_ev,
            reference_interaction_energy,
            delta=2e-4,
        )

        reference_interaction_forces = [
            force.copy()
            for force in reference_combined.forces_ev_per_angstrom
        ]
        for local_index, combined_index in enumerate(
            self.configuration.framework_indices
        ):
            reference_interaction_forces[combined_index] = [
                combined - isolated
                for combined, isolated in zip(
                    reference_combined.forces_ev_per_angstrom[combined_index],
                    reference_framework.forces_ev_per_angstrom[local_index],
                    strict=True,
                )
            ]
        for local_index, combined_index in enumerate(
            self.configuration.adsorbate_indices
        ):
            reference_interaction_forces[combined_index] = [
                combined - isolated
                for combined, isolated in zip(
                    reference_combined.forces_ev_per_angstrom[combined_index],
                    reference_adsorbate.forces_ev_per_angstrom[local_index],
                    strict=True,
                )
            ]
        maximum_force_difference = max(
            abs(candidate - reference)
            for candidate_force, reference_force in zip(
                interaction.interaction_forces_ev_per_angstrom,
                reference_interaction_forces,
                strict=True,
            )
            for candidate, reference in zip(
                candidate_force,
                reference_force,
                strict=True,
            )
        )
        self.assertLessEqual(maximum_force_difference, 2e-4)

    @unittest.skipUnless(
        HAS_MODULE_A_SNAPSHOT,
        "An existing Module-A GCMC dump is required for snapshot parity.",
    )
    def test_existing_module_a_gcmc_snapshot_matches(self) -> None:
        configuration = load_lammps_dump_configuration(
            MODULE_A_SNAPSHOT,
            MODULE_A_SNAPSHOT_FORCEFIELD,
            CO2_PATH,
            PSEUDO_ATOMS_PATH,
        )
        candidate = self.backend.evaluate(configuration)
        module_c_run = self.root / "module_c" / configuration.configuration_id
        reference = evaluate_module_a_static_files(
            data_file=module_c_run / "combined.data",
            forcefield_file=MODULE_A_SNAPSHOT_FORCEFIELD,
            atom_count=len(configuration.atoms),
            working_directory=self.root / "module_a_sampled_snapshot",
            has_bonds=True,
        )

        report = compare_classical_results(
            reference,
            candidate,
            energy_tolerance_ev=1e-8,
            force_tolerance_ev_per_angstrom=1e-8,
        )

        self.assertGreater(len(configuration.adsorbate_indices), 0)
        self.assertTrue(report.passed, report.to_dict())


if __name__ == "__main__":
    unittest.main()
