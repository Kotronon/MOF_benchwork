"""General-profile contracts and opt-in real LAMMPS integration tests.

Fixtures below are invented numerical systems, NOT parameterized MOFs or
scientific adsorption benchmarks. They test state handling and units only.
"""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from benchmark import build_run_plan
from analysis.module_d_flexibility import evaluate_flexibility
from modules.module_d_flexibility import materialize_flexibility, prepare_flexibility
from modules.module_d_general import execute_bundle, render_input, verify_bundle


def fixture(directory, molecule=False, units="real"):
    root = Path(directory)
    data = root / "dry.data"
    data.write_text("""Synthetic technical fixture, not a MOF force field

8 atoms
4 bonds
3 atom types
2 bond types
1 angle types

0 30 xlo xhi
0 30 ylo yhi
0 30 zlo zhi

Masses

1 12.011
2 12.011
3 15.999

Atoms # full

1 1 1 0.1 5 5 5
2 1 1 -0.1 7 5 5
3 2 1 0.1 5 15 5
4 2 1 -0.1 7 15 5
5 3 1 0.1 5 5 15
6 3 1 -0.1 7 5 15
7 4 1 0.1 5 15 15
8 4 1 -0.1 7 15 15

Bonds

1 1 1 2
2 1 3 4
3 1 5 6
4 1 7 8
""")
    template = root / "guest.mol"
    template.write_text("""Synthetic neutral linear triatomic guest

3 atoms
2 bonds
1 angles

Coords

1 0 0 0
2 1.16 0 0
3 -1.16 0 0

Types

1 2
2 3
3 3

Charges

1 0.4
2 -0.2
3 -0.2

Bonds

1 2 1 2
2 2 1 3

Angles

1 1 2 1 3
""")
    energy_scale = 1 if units == "real" else .04336410424180094
    return dict(material=dict(name="arbitrary-framework", charge_scheme="in_data"),
        adsorbates=dict(components=["CO2" if molecule else "AR"]),
        conditions=dict(temperature_K=300., pressures_bar=[100.]),
        simulation=dict(engine="LAMMPS", framework="flexible", seeds=[12345]),
        output=dict(directory=str(root / "bundle")),
        benchmark=dict(task="module_d", flexibility=dict(profile="general", sr_cycles=4,
            discard_cycles=0, convergence_window=2,
            potential=dict(kind="classical", name="synthetic numerical fixture", source="test only, not a MOF model",
                units=units, styles=["pair_style lj/cut/coul/long 8", "bond_style harmonic", "angle_style harmonic"],
                coefficients=[f"pair_coeff * * {0.01 * energy_scale} 2", f"bond_coeff 1 {20 * energy_scale} 2",
                    f"bond_coeff 2 {20 * energy_scale} 1.16", f"angle_coeff 1 {20 * energy_scale} 180",
                    "special_bonds lj/coul 0 0 0", "kspace_style ewald 1e-5"]),
            assets=dict(initial_structure=str(data), guest_template=str(template)),
            guest=dict(kind="rigid_molecule" if molecule else "atomic", type=2,
                intra_energy=0., intra_energy_source="Zero harmonic energy at template geometry; intramolecular nonbonded exclusions",
                atoms_per_molecule=3 if molecule else 1, extra_bond_per_atom=2,
                extra_angle_per_atom=1, extra_special_per_atom=4),
            md=dict(ensemble="NVT", timestep_fs=.1, thermostat_damping_ps=.1),
            hybrid=dict(equilibration_steps=10, steps_per_cycle=10, exchange_interval=1,
                        exchange_attempts=2, move_attempts=0 if molecule else 2),
            reservoir=dict(fugacities_bar=[100.], source="explicit ideal-gas approximation for technical fixture"))))


class GeneralProfileTests(unittest.TestCase):
    def test_guest_count_charge_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, molecule=True)
            config["benchmark"]["flexibility"]["guest"]["atoms_per_molecule"] = 4
            with self.assertRaisesRegex(ValueError, "atoms_per_molecule"):
                build_run_plan(config)
            config = fixture(tmp, molecule=True)
            path = Path(config["benchmark"]["flexibility"]["assets"]["guest_template"])
            path.write_text(path.read_text().replace("1 0.4", "1 0.5"))
            with self.assertRaisesRegex(ValueError, "neutral"):
                build_run_plan(config)
            config = fixture(tmp)
            config["benchmark"]["flexibility"]["assets"]["initial_structure"] = None
            plan = build_run_plan(config)
            root = materialize_flexibility(prepare_flexibility(plan))["working_directory"]
            with self.assertRaisesRegex(ValueError, "Missing asset"):
                execute_bundle(root, smoke=True)

    def test_failed_subprocess_is_preserved_and_cannot_unlock_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_run_plan(fixture(tmp))
            root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
            help_result = subprocess.CompletedProcess(["lmp"], 0, "test build", "")
            with patch("modules.module_d_general.subprocess.run", side_effect=[help_result, subprocess.CalledProcessError(1, ["lmp"])]):
                with self.assertRaises(subprocess.CalledProcessError):
                    execute_bundle(root, smoke=True)
            self.assertEqual(json.loads((root / "smoke/status.json").read_text())["status"], "failed")
            self.assertFalse((root / "cycle_results.csv").exists())
            with patch("modules.module_d_general.subprocess.run", return_value=help_result):
                with self.assertRaisesRegex(ValueError, "passing smoke"):
                    execute_bundle(root)
                with self.assertRaises(FileExistsError):
                    execute_bundle(root, smoke=True)

    def test_arbitrary_material_no_paper_constraints_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            before = deepcopy(config)
            plan = build_run_plan(config)
            self.assertEqual(before, config)
            self.assertTrue(plan["benchmark"]["applicability"]["can_attempt_simulation"])
            self.assertNotIn("relative_humidity", plan["state_points"][0])
            self.assertEqual([c["variant"] for c in plan["cases"]], ["rigid", "hybrid_nvt"])
            self.assertEqual([c["expected_cycles"] for c in plan["cases"]], [4, 4])

    def test_mlip_is_plannable_but_not_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind, ensemble in (("mlip", "NVT"),):
                config = fixture(tmp)
                p = config["benchmark"]["flexibility"]
                p["potential"]["kind"] = kind
                p["md"].update(ensemble=ensemble, external_pressure_bar=1.)
                plan = build_run_plan(config)
                self.assertFalse(plan["benchmark"]["applicability"]["can_attempt_simulation"])
                with self.assertRaises(NotImplementedError):
                    render_input(plan, plan["cases"][0])

    def test_npt_requires_cell_and_pressure_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            md = config["benchmark"]["flexibility"]["md"]
            md.update(ensemble="NPT", external_pressure_bar=1.)
            with self.assertRaisesRegex(ValueError, "cell_mode"):
                build_run_plan(config)
            md.update(cell_mode="tri", barostat_damping_ps=1.)
            plan = build_run_plan(config)
            self.assertEqual([c["variant"] for c in plan["cases"]], ["rigid", "hybrid_nvt", "hybrid_npt"])
            deck = render_input(plan, plan["cases"][2])
            self.assertIn("dilate all mtk yes flip no", deck)
            self.assertIn("fix_modify hostmd press globalpress", deck)
            self.assertNotIn("host npt", render_input(plan, plan["cases"][1]))

    def test_invalid_inputs_do_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            mutations = [lambda p: p["reservoir"].pop("source"),
                         lambda p: p["reservoir"].update(fugacities_bar=[True]),
                         lambda p: p["potential"].update(kind="unknown"),
                         lambda p: p["potential"].update(styles=["pair_style lj/cut 8\nrun 100"]),
                         lambda p: p["potential"].update(coefficients=["shell touch bad"]),
                         lambda p: p["potential"].update(files={"../bad": "x"}),
                         lambda p: p["hybrid"].update(equilibration_steps=11),
                         lambda p: p["guest"].update(kind="flexible_molecule")]
            for mutate in mutations:
                config = fixture(tmp)
                mutate(config["benchmark"]["flexibility"])
                with self.assertRaises(ValueError):
                    build_run_plan(config)

    def test_bundle_portable_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            plan = build_run_plan(config)
            result = materialize_flexibility(prepare_flexibility(plan))
            root = Path(result["working_directory"])
            self.assertFalse((root / "upstream_reference").exists())
            self.assertEqual(verify_bundle(root), plan)
            config["output"]["resume"] = True
            self.assertEqual(materialize_flexibility(prepare_flexibility(build_run_plan(config)))["status"], "reused_plan")
            (root / "assets/initial_structure").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(root)

    def test_real_and_metal_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            for units, pressure, timestep in (("real", 100 / 1.01325, .1), ("metal", 100., .0001)):
                plan = build_run_plan(fixture(tmp, units=units))
                rendered = render_input(plan, plan["cases"][1])
                self.assertIn(f"pressure {pressure}", rendered)
                self.assertIn(f"timestep {timestep}", rendered)
                self.assertIn("full_energy", rendered)
                self.assertIn("fix hostmd", rendered)
                self.assertNotIn("fix hostmd", render_input(plan, plan["cases"][0]))


@unittest.skipUnless(os.environ.get("MODULE_D_LAMMPS"), "Set MODULE_D_LAMMPS to run real LAMMPS integration tests")
class RealLAMMPSTests(unittest.TestCase):
    def test_rigid_guest_npt_cell_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, molecule=True)
            config["benchmark"]["flexibility"]["md"].update(ensemble="NPT", cell_mode="tri",
                external_pressure_bar=100., barostat_damping_ps=1.)
            plan = build_run_plan(config)
            root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
            try:
                result = execute_bundle(root, smoke=True, command=[os.environ["MODULE_D_LAMMPS"]])
            except Exception:
                for log in root.rglob("console.log"):
                    print(log, log.read_text()[-3000:])
                raise
            self.assertEqual(result["status"], "passed")
            report = json.loads((root / "smoke/report.json").read_text())
            self.assertNotAlmostEqual(report["cases"][2]["mean_cell_volume_A3"], 27000., places=2)
            from ase.io import read
            import numpy as np
            final = read(root / "smoke" / plan["cases"][2]["case_id"] / "final.data",
                         format="lammps-data", atom_style="full")
            ids = final.arrays["mol-id"]
            for mol in set(ids[final.arrays["id"] > 8]):
                indices = np.flatnonzero(ids == mol)
                self.assertEqual(len(indices), 3)
                distances = sorted(final.get_distance(a, b, mic=True) for j, a in enumerate(indices) for b in indices[j+1:])
                self.assertTrue(np.allclose(distances,[1.16,1.16,2.32],atol=1e-6))

    def test_atomic_and_rigid_molecular_runs(self):
        for molecule, units in ((False, "real"), (True, "real"), (False, "metal")):
            with self.subTest(molecule=molecule, units=units), tempfile.TemporaryDirectory() as tmp:
                config = fixture(tmp, molecule=molecule, units=units)
                plan = build_run_plan(config)
                root = Path(materialize_flexibility(prepare_flexibility(plan))["working_directory"])
                command = [os.environ["MODULE_D_LAMMPS"]]
                try:
                    smoke = execute_bundle(root, smoke=True, command=command)
                    self.assertEqual(smoke["status"], "passed")
                    self.assertFalse((root / "cycle_results.csv").exists())
                    self.assertEqual(execute_bundle(root, command=command)["status"], "complete")
                except Exception:
                    for log in root.rglob("console.log"):
                        print(log, log.read_text()[-8000:])
                    raise
                report = evaluate_flexibility(root)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["validation_status"], "not_validated")
                self.assertEqual(len(report["comparisons"]), 1)
                rigid, flexible = report["cases"]
                self.assertAlmostEqual(rigid["mean_framework_rmsd_A"], 0.)
                self.assertGreater(flexible["mean_framework_rmsd_A"], 0.)
                self.assertGreater(flexible["mean_uptake_mol_kg"], 0.)
                self.assertAlmostEqual(flexible["mean_cell_volume_A3"], 27000.)
                with (root / "cycle_results.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 8)
                from ase.io import read
                import numpy as np
                initial = read(root / "assets/initial_structure", format="lammps-data", atom_style="full", units=units)
                for case in plan["cases"]:
                    final = read(root / "results" / case["case_id"] / "final.data",
                                 format="lammps-data", atom_style="full", units=units)
                    self.assertTrue(np.isfinite(final.positions).all())
                    self.assertTrue(np.allclose(initial.cell, final.cell))
                    ids = final.arrays["id"]
                    host = [list(ids).index(i) for i in initial.arrays["id"]]
                    self.assertTrue(np.allclose(final.get_initial_charges()[host], initial.get_initial_charges()))
                    if case["variant"] == "rigid":
                        self.assertTrue(np.allclose(final.positions[host], initial.positions))
                    if molecule:
                        guest_indices = [i for i, atom_id in enumerate(ids) if atom_id not in initial.arrays["id"]]
                        molecule_ids = final.arrays["mol-id"]
                        self.assertTrue(guest_indices)
                        for mol in {molecule_ids[i] for i in guest_indices}:
                            indices = [i for i in guest_indices if molecule_ids[i] == mol]
                            self.assertEqual(len(indices), 3)
                            distances = sorted(final.get_distance(a, b, mic=True) for j, a in enumerate(indices) for b in indices[j + 1:])
                            self.assertTrue(np.allclose(distances, [1.16, 1.16, 2.32], atol=1e-6))
                with self.assertRaises(FileExistsError):
                    execute_bundle(root, command=command)


if __name__ == "__main__":
    unittest.main()
