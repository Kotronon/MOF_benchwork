"""Integration checks using the shared Module-A material and LAMMPS path."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pipeline.planning import build_run_plan
from pipeline.prepare import prepare_benchmark
from pipeline.materialization import materialize_benchmark
from pipeline.config import normalize_config
from resolvers.material_resolver import MaterialResolver

ROOT = Path(__file__).resolve().parents[1]


class MaterialParityTests(unittest.TestCase):
    def test_adding_components_preserves_existing_forcefield(self):
        from resolvers.forcefield_resolver import ForcefieldResolver
        from converter.forcefield_to_lammps import load_forcefield_parameters

        resolver = ForcefieldResolver()
        base = load_forcefield_parameters(resolver.resolve({}, ["CO2", "N2"]))
        for names in (["CO2", "N2", "CH4"], ["AR", "CH4", "N2", "CO2"]):
            mixed = load_forcefield_parameters(resolver.resolve({}, names))
            for atom_type in ["C_co2", "O_co2", "N_n2"]:
                self.assertEqual(
                    base.lj_parameters[atom_type], mixed.lj_parameters[atom_type]
                )
                self.assertEqual(
                    base.pseudo_atoms[atom_type], mixed.pseudo_atoms[atom_type]
                )

    def test_conflicting_parameter_bundles_fail_explicitly(self):
        from resolvers.forcefield_resolver import ForcefieldResolver
        from converter.forcefield_to_lammps import load_forcefield_parameters

        config = ForcefieldResolver().resolve({}, ["CO2", "CH4"])
        config["adsorbate_parameter_files"][0]["atom_types"] = ["C_co2"]
        with self.assertRaisesRegex(ValueError, "Conflicting forcefield parameter"):
            load_forcefield_parameters(config)

    def test_entire_catalog_uses_same_framework_plan(self):
        resolver = MaterialResolver(use_nist_aliases=False)
        seed = build_run_plan(
            json.loads(
                (
                    ROOT
                    / "input_json_files/benchmark_mof5_co2_n2_mixture_smoke_test.json"
                ).read_text()
            )
        )
        # Exercise every catalog entry through both preparation paths without simulating the catalog.
        for material_id in resolver._cif_index:
            match = resolver.resolve(material_id)
            a = deepcopy(seed)
            a["module"]["id"] = "A"
            a["material"].update(
                material_id=match.material_id, charge_scheme=match.charge_scheme
            )
            a["resources"]["cif_path"] = str(match.cif_path)
            a["adsorbates"] = {"components": ["CO2"], "mixture": None}
            b = deepcopy(a)
            b["module"]["id"] = "B"
            b["adsorbates"] = seed["adsorbates"]
            pa, pb = prepare_benchmark(a), prepare_benchmark(b)
            self.assertEqual(
                pa["planned_files"]["framework_data"],
                pb["planned_files"]["framework_data"],
            )
            self.assertEqual(
                pa["inputs"]["framework_cif"], pb["inputs"]["framework_cif"]
            )
        self.assertGreater(len(resolver._cif_index), 10)

    def test_representative_framework_materialization(self):
        from converter.cif_to_lammps_data import load_framework_structure

        base = json.loads(
            (
                ROOT / "input_json_files/benchmark_mof5_co2_n2_mixture_smoke_test.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, scheme in [
                ("MOF-5", "DDEC"),
                ("IRMOF-7", "EQeq"),
                ("ZIF-8", "PACMOF"),
            ]:
                cfg = deepcopy(base)
                cfg["material"]["name"] = name
                cfg["material"]["charge_scheme"] = scheme
                cfg["simulation"].update(cutoff_A=6.0, cell_representation="source")
                cfg["output"].update(directory=tmp, run_id=name)
                cfg["benchmark"]["reference"] = {"source": "none"}
                plan = prepare_benchmark(build_run_plan(cfg))
                result = materialize_benchmark(plan)
                original = load_framework_structure(
                    plan["inputs"]["framework_cif"],
                    unit_cells=[1, 1, 1],
                    cell_representation="source",
                    cutoff_A=6.0,
                )
                self.assertEqual(
                    result["parameters"]["framework_atom_count"], original.atom_count
                )
                self.assertEqual(
                    result["parameters"]["framework_cell_lengths_A"],
                    list(original.cell_lengths),
                )
                self.assertTrue(Path(result["files"]["framework_data"]).is_file())

    def test_four_components_with_zero_fraction_materialize(self):
        cfg = json.loads(
            (
                ROOT / "input_json_files/benchmark_module_b_co2_ch4_smoke.json"
            ).read_text()
        )
        cfg["adsorbates"] = {
            "components": ["AR", "CO2", "CH4", "N2"],
            "mixture": {"AR": 0.0, "CO2": 0.4, "CH4": 0.3, "N2": 0.3},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output"].update(directory=tmp, run_id="four")
            result = materialize_benchmark(prepare_benchmark(build_run_plan(cfg)))
            script = Path(result["files"]["gcmc_inputs"][0]).read_text()
            self.assertNotIn("fix gcmc_ar ", script)
            self.assertIn("variable nar equal", script)
            for c in ["co2", "ch4", "n2"]:
                self.assertIn(f"fix gcmc_{c} ", script)
            self.assertEqual(
                set(result["parameters"]["component_atoms_per_molecule"]),
                {"AR", "CO2", "CH4", "N2"},
            )


class LammpsEnergyTests(unittest.TestCase):
    def test_flexible_tree_energy_and_framework_invariance(self):
        try:
            from lammps import lammps
        except ImportError:
            self.skipTest("LAMMPS Python API unavailable")
        from engines.flexible_guest_runner import load_models, LammpsGuestEnergy

        models = load_models(
            {"G": str(ROOT / "data/module_b/models/gaussian_branched.json")}, 300.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "frame.data"
            forcefield = Path(tmp) / "ff.inc"
            data.write_text(
                "test\n\n1 atoms\n2 atom types\n\n0 30 xlo xhi\n0 30 ylo yhi\n0 30 zlo zhi\n\nMasses\n\n1 12\n2 12\n\nAtoms # full\n\n1 0 1 0 5 5 5\n"
            )
            forcefield.write_text("pair_style lj/cut 10\npair_coeff * * 0 1\n")
            run = {
                "framework_data": str(data),
                "forcefield_include": str(forcefield),
                "guest_atoms": {
                    "G": [{"type": 2, "charge": 0, "position": [0, 0, 0]}] * 3
                },
            }
            engine = LammpsGuestEnergy(run, models)
            try:
                zero = engine([])
                molecule = {
                    "component": "G",
                    "positions": [[10, 10, 10], [11, 10, 10], [10, 12, 10]],
                }
                self.assertAlmostEqual(engine([molecule]) - zero, 5.0, places=8)
                energy, forces = engine.energy_forces([molecule])
                np.testing.assert_allclose(
                    forces[0], [[2, 4, 0], [-2, 0, 0], [0, -4, 0]], atol=1e-8
                )
                actual = np.ctypeslib.as_array(
                    engine.lmp.gather_atoms("x", 1, 3), shape=(12,)
                ).reshape(-1, 3)
                np.testing.assert_allclose(
                    actual[0], engine.reference_coords[0], atol=1e-12
                )
                # Deletion followed by reinsertion must preserve the same energy.
                self.assertEqual(engine([]), zero)
                self.assertAlmostEqual(engine([molecule]) - zero, 5.0, places=8)
            finally:
                engine.close()


class CampaignLoopTests(unittest.TestCase):
    def test_complete_import_resume_and_holdout_isolation(self):
        from modules.module_b_campaign import initialize, propose, ingest
        from modules.module_b_profiles import point
        from pipeline.config import save_benchmark_data
        from pipeline.runners import _aggregate_replicates

        cfg = json.loads(
            (
                ROOT / "input_json_files/benchmark_module_b_co2_ch4_smoke.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output"].update(directory=tmp, run_id="base")
            candidates = [
                point(p, 300, {"CO2": 0.5, "CH4": 0.5})
                for p in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
            ]
            state_path = Path(tmp) / "state.json"
            initialize(cfg, state_path, candidates=candidates, budget=8)
            decision = propose(state_path)
            self.assertEqual(propose(state_path)["request"], decision["request"])
            request = decision["request"]
            request_cfg = json.loads(Path(request["config_path"]).read_text())
            result = materialize_benchmark(
                prepare_benchmark(build_run_plan(request_cfg))
            )
            p = request["candidate"]
            reps = []
            for seed in [1, 2, 3]:
                reps.append(
                    {
                        "pressure_bar": p["pressure_bar"],
                        "seed": seed,
                        "summary": {
                            "mean_adsorbates": 3.0,
                            "components": {
                                c: {
                                    "mean_molecules": v,
                                    "gas_phase_mole_fraction": p["mixture"][c],
                                }
                                for c, v in [("CO2", 2.0), ("CH4", 1.0)]
                            },
                        },
                    }
                )
            summary = Path(result["working_directory"]) / "isotherm_summary.json"
            save_benchmark_data(
                summary,
                {
                    "status": "completed",
                    "replicate_results": reps,
                    "results": _aggregate_replicates(reps),
                },
            )
            state = ingest(state_path, summary)
            self.assertEqual(state, ingest(state_path, summary))
            self.assertEqual(len(state["results"]), 1)
            self.assertFalse(state["holdout_results"])
            held = {x["candidate_id"] for x in state["holdout"]}
            self.assertNotIn(state["results"][0]["candidate_id"], held)
            # A changed archived input cannot be silently imported.
            state["imports"] = []
            state["pending_request"] = request
            save_benchmark_data(state_path, state)
            archived = next(iter(result["files"]["source_sha256"]))
            Path(archived).write_text("changed")
            with self.assertRaisesRegex(ValueError, "Archived simulation input"):
                ingest(state_path, summary)


if __name__ == "__main__":
    unittest.main()
