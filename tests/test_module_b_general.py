from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch
import importlib.util
import numpy as np

from pipeline.config import normalize_config
from pipeline.eos import mixture_fugacities
from pipeline.lammps_inputs import gcmc_input_builder
from modules.registry import select_registered_module
from modules.module_b_profiles import paper_profile, simplex_candidates, point
from engines.configurational_bias import ConfigurationalBias, TreeProposal
from engines.flexible_guest_runner import load_models

CRITICAL = {
    "CO2": {
        "critical_temperature_K": 304.1282,
        "critical_pressure_Pa": 7377300.0,
        "acentric_factor": 0.22394,
    },
    "CH4": {
        "critical_temperature_K": 190.564,
        "critical_pressure_Pa": 4599200.0,
        "acentric_factor": 0.01142,
    },
}


class GeneralConfigurationTests(unittest.TestCase):
    def config(self):
        return {
            "material": {"name": "MOF5"},
            "adsorbates": {
                "components": ["CO2", "CH4", "N2"],
                "mixture": {"CO2": 0.5, "CH4": 0.5, "N2": 0},
            },
        }

    def test_zero_fraction_and_component_count(self):
        c = normalize_config(self.config())
        self.assertEqual(c["adsorbates"]["mixture"]["N2"], 0)
        self.assertEqual(select_registered_module(c).id, "B")
        c["simulation"]["method"] = "FLEXIBLE_GUEST_GCMC"
        self.assertEqual(select_registered_module(c).id, "B")
        c["simulation"]["framework"] = "flexible"
        self.assertEqual(select_registered_module(c).id, "D")

    def test_invalid_composition_and_duplicates(self):
        for value in (-0.2, float("nan"), float("inf")):
            c = self.config()
            c["adsorbates"]["mixture"]["N2"] = value
            with self.assertRaises(ValueError):
                normalize_config(c)
        c = self.config()
        c["adsorbates"]["components"].append("CO2")
        with self.assertRaises(ValueError):
            normalize_config(c)

    def test_custom_forcefield_options_survive(self):
        c = self.config()
        c["simulation"] = {
            "forcefield": {"framework": "UFF", "definitions": {"CH4": "methane.def"}}
        }
        self.assertIn("definitions", normalize_config(c)["simulation"]["forcefield"])

    def test_zero_fraction_pairs_do_not_fail_active_subsystem_precision(self):
        from pipeline.runners import _aggregate_replicates
        reps = [{"pressure_bar": 1., "seed": seed, "summary": {
            "mean_adsorbates": 3., "components": {
                c: {"mean_molecules": q, "gas_phase_mole_fraction": y}
                for c, q, y in [("CO2", 2., .5), ("CH4", 1., .5), ("N2", 0., 0.)]}}}
                for seed in [123, 456, 789]]
        result = _aggregate_replicates(reps)[0]
        self.assertTrue(result["precision_met"])
        self.assertIsNone(result["summary"]["selectivity"]["CO2/N2"]["value"])


class EquationOfStateTests(unittest.TestCase):
    def test_ideal_limit(self):
        result = mixture_fugacities(300, 1e-6, {"CO2": 0.4, "CH4": 0.6}, CRITICAL)
        for x in result["coefficients"].values():
            self.assertAlmostEqual(x, 1.0, places=6)

    def test_order_and_zero_component_invariance(self):
        a = mixture_fugacities(300, 50, {"CO2": 0.4, "CH4": 0.6}, CRITICAL)
        b = mixture_fugacities(300, 50, {"CH4": 0.6, "CO2": 0.4}, CRITICAL)
        self.assertEqual(a, b)
        pure = mixture_fugacities(300, 50, {"CO2": 1.0}, CRITICAL)
        zero = mixture_fugacities(300, 50, {"CO2": 1.0, "CH4": 0.0}, CRITICAL)
        self.assertAlmostEqual(pure["coefficients"]["CO2"], zero["coefficients"]["CO2"])

    @unittest.skipUnless(importlib.util.find_spec("CoolProp"), "CoolProp crosscheck")
    def test_pure_PR_independent_implementation(self):
        from CoolProp import CoolProp as CP

        state = CP.AbstractState("PR", "Methane")
        state.update(CP.PT_INPUTS, 5e6, 300)
        result = mixture_fugacities(300, 50, {"CH4": 1}, CRITICAL)
        self.assertAlmostEqual(
            result["coefficients"]["CH4"], state.fugacity_coefficient(0), delta=2e-4
        )

    def test_explicit_and_invalid(self):
        r = mixture_fugacities(
            300, 10, {"A": 0.5, "B": 0.5}, {}, fugacities_bar={"A": 3.0, "B": 4.0}
        )
        self.assertEqual(r["coefficients"], {"A": 0.6, "B": 0.8})
        with self.assertRaises(ValueError):
            mixture_fugacities(300, 10, {"A": 1.0}, {}, fugacities_bar={"A": -1.0})
        with self.assertRaises(ValueError):
            mixture_fugacities(300, 10, {"CO2": 1.0}, CRITICAL, root="auto")

    def test_multiple_roots(self):
        a = mixture_fugacities(250, 10, {"CO2": 1.0}, CRITICAL, root="vapor")
        b = mixture_fugacities(250, 10, {"CO2": 1.0}, CRITICAL, root="liquid")
        self.assertGreater(a["compressibility"], b["compressibility"])


class ProfileTests(unittest.TestCase):
    def test_all_six_grids(self):
        for case in ("CO2-CH4", "Xe-Kr", "H2S-CO2"):
            for space, n, initial in (("P-X", 2499, 54), ("P-X-T", 2541, 90)):
                p = paper_profile(case, space)
                self.assertEqual(len(p["candidates"]), n)
                self.assertEqual(len(p["initial_design"]), initial)
                self.assertEqual(len({x["candidate_id"] for x in p["candidates"]}), n)
                archived = json.loads(
                    (
                        Path(__file__).resolve().parents[1]
                        / "data/module_b/paper_profiles"
                        / f"{case.lower()}_{space.lower()}.json"
                    ).read_text()
                )
                self.assertEqual(archived["initial_design"], p["initial_design"])
                for key in [
                    "kernel",
                    "alpha",
                    "beta",
                    "optimizer_restarts",
                    "holdout_count",
                ]:
                    self.assertEqual(archived[key], p[key])

    def test_kernel_discrepancy_explicit(self):
        self.assertEqual(paper_profile("H2S-CO2", "P-X-T")["kernel"], "rq")
        self.assertEqual(
            paper_profile("H2S-CO2", "P-X-T", "upstream_compat")["kernel"], "sum3_rbf"
        )

    def test_simplex_and_order(self):
        for names in (["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]):
            a = simplex_candidates(names, [0.1, 100], [250, 350], 128)
            self.assertEqual(
                a,
                simplex_candidates(list(reversed(names)), [0.1, 100], [250, 350], 128),
            )
            self.assertTrue(all(abs(sum(p["mixture"].values()) - 1) < 1e-12 for p in a))
            self.assertEqual(len({p["candidate_id"] for p in a}), 128)

    def test_point_identity_includes_T_and_composition(self):
        a = point(1, 300, {"A": 0.2, "B": 0.8})
        self.assertEqual(a, point(1, 300, {"B": 0.8, "A": 0.2}))
        self.assertNotEqual(
            a["candidate_id"], point(1, 301, a["mixture"])["candidate_id"]
        )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn")
    def test_pinned_upstream_coordinates_and_GP_transform(self):
        import csv
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from modules.module_b_campaign import predict

        fixture = (
            Path(__file__).resolve().parents[1]
            / "data/module_b/fixtures/upstream_co2_ch4_px_training.csv"
        )
        with fixture.open() as handle:
            rows = [
                {k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)
            ]
        profile = paper_profile("CO2-CH4")
        coords = [
            point(
                r["Pressure(Pa)"] / 1e5,
                r["Temperature(K)"],
                {c: r[c + "_mole_frac"] for c in profile["components"]},
            )
            for r in rows
        ]
        self.assertEqual(
            {p["candidate_id"] for p in coords},
            {p["candidate_id"] for p in profile["initial_design"]},
        )
        # Treat upstream numbers as opaque algorithm fixtures, never as LAMMPS labels.
        profile["uptake_unit"] = "molecules/cell"
        state = {
            "mode": "paper",
            "profile": profile,
            "components": profile["components"],
            "candidates": profile["candidates"],
            "options": {"random_seed": 2026},
            "results": [
                {
                    **p,
                    "components": {
                        c: {"mean_molecules": r[c + "_uptake"]}
                        for c in profile["components"]
                    },
                }
                for p, r in zip(coords, rows)
            ],
        }
        subset = profile["candidates"][::250]

        def fixed_optimizer(**kwargs):
            kwargs.update(optimizer=None, n_restarts_optimizer=0)
            return GaussianProcessRegressor(**kwargs)

        with patch(
            "sklearn.gaussian_process.GaussianProcessRegressor",
            side_effect=fixed_optimizer,
        ):
            actual = predict(state, subset)
        axis = np.log10(np.linspace(1e-6, 300, 51))
        for c in profile["components"]:

            def upstream_x(points):
                return np.array(
                    [
                        [
                            (np.log10(p["pressure_bar"]) - axis.mean())
                            / axis.std(ddof=1),
                            (p["mixture"][c] - 0.5) * 25 / 12,
                        ]
                        for p in points
                    ]
                )

            y = np.array([r[c + "_uptake"] for r in rows])
            y[y == 0] = 1e-4
            gp = GaussianProcessRegressor(
                kernel=RBF(1.0, (1e-15, 1e15)),
                alpha=1e-4,
                normalize_y=True,
                optimizer=None,
            ).fit(upstream_x(coords), np.log10(y))
            expected, sigma = gp.predict(upstream_x(subset), return_std=True)
            np.testing.assert_allclose(
                [p["transformed_mean"] for p in actual["components"][c]],
                expected,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                [p["transformed_std"] for p in actual["components"][c]],
                sigma,
                atol=1e-10,
            )


class FlexibleSamplingTests(unittest.TestCase):
    def sampler(self, parents=(None, 0, 0)):
        T = 300.0
        proposal = TreeProposal(parents, [None] + [1.0] * (len(parents) - 1), T, 0.0)
        energy = lambda molecules: sum(
            proposal.reference_energy(np.array(m["positions"])) for m in molecules
        )
        return ConfigurationalBias(
            energy,
            {"G": proposal},
            {"G": 20.709735},
            T,
            np.eye(3) * 10,
            seed=814,
            trials=3,
        )

    def test_reference_partition_and_ideal_count(self):
        s = self.sampler()
        counts = []
        for i in range(6000):
            s.step()
            if i > 1000:
                counts.append(len(s.molecules))
        self.assertAlmostEqual(np.mean(counts), 0.5, delta=0.12)
        self.assertAlmostEqual(np.var(counts), 0.5, delta=0.18)
        # All Rosenbluth weights of this analytic ideal model are exactly one.
        _, weight = s._trials([], "G")
        self.assertAlmostEqual(weight, 0.0, places=10)

    def test_linear_and_branched_inner_distribution(self):
        for parents in ([None, 0, 1], [None, 0, 0]):
            s = self.sampler(parents)
            p = s.proposals["G"]
            samples = [p.draw(s.rng, s.cell) for _ in range(1000)]
            r2 = np.mean([np.sum((x[1] - x[0]) ** 2) for x in samples])
            self.assertAlmostEqual(r2, 3 / (2 * p.beta), delta=0.1)

    def test_resume_exact_sequence(self):
        a = self.sampler()
        for _ in range(20):
            a.step()
        b = self.sampler()
        b.restore(a.checkpoint())
        for _ in range(20):
            self.assertEqual(a.step(), b.step())
        self.assertEqual(a.checkpoint(), b.checkpoint())

    def test_ring_and_missing_normalization_rejected(self):
        with self.assertRaises(ValueError):
            TreeProposal([None, 2, 0], [None, 1, 1], 300, 0)
        with self.assertRaises(ValueError):
            load_models({"X": {"rings": True}}, 300)

    def test_nontrivial_rosenbluth_normalization_and_inner_distribution(self):
        # Physical springs twice as stiff as the proposal: Z_phys/Z_ref = 2^-3.
        s = self.sampler()
        s.proposals["G"].log_ideal = -3 * math.log(2)
        reference = s.energy
        s.energy = lambda molecules: 2 * reference(molecules)
        counts, lengths = [], []
        for i in range(16000):
            s.step()
            if i > 2000:
                counts.append(len(s.molecules))
                for m in s.molecules:
                    xyz = np.asarray(m["positions"])
                    lengths.append(np.sum((xyz[1] - xyz[0]) ** 2))
        self.assertAlmostEqual(np.mean(counts), 0.5, delta=0.08)
        self.assertAlmostEqual(np.var(counts), 0.5, delta=0.12)
        self.assertAlmostEqual(np.mean(lengths), 3 / (4 * s.beta), delta=0.04)

    def test_hmc_samples_flexible_modes_and_freezes_rigid_guests(self):
        from engines.configurational_bias import RigidProposal

        s = self.sampler()
        s.proposals["R"] = RigidProposal([[0, 0, 0]])
        s.molecules = [
            {"component": "G", "positions": [[5, 5, 5], [5, 5, 5], [5, 5, 5]]},
            {"component": "R", "positions": [[2, 3, 4]]},
        ]
        rigid = deepcopy(s.molecules[1])

        def energy_forces(molecules):
            xyz = np.asarray(molecules[0]["positions"])
            d = xyz[1:] - xyz[0]
            return float(np.sum(d * d)), [
                np.vstack([2 * d.sum(axis=0), -2 * d]),
                np.zeros((1, 3)),
            ]

        lengths = []
        for i in range(2200):
            s.hmc(energy_forces, {"G": [1, 1, 1]}, steps=8, step_size=0.12)
            if i > 200:
                xyz = np.asarray(s.molecules[0]["positions"])
                lengths.append(np.sum((xyz[1] - xyz[0]) ** 2))
        self.assertEqual(s.molecules[1], rigid)
        self.assertAlmostEqual(np.mean(lengths), 3 / (2 * s.beta), delta=0.08)
        self.assertGreater(s.counts["hmc"]["accepted"], 1900)
        old = deepcopy(s.molecules)
        self.assertFalse(s.hmc(energy_forces, {"G": [1, 1, 1]}, steps=8, step_size=100))
        self.assertEqual(s.molecules, old)


class GeneralCampaignTests(unittest.TestCase):
    def state(self):
        from modules.module_b_campaign import initialize

        names = ["CO2", "CH4", "N2"]
        c = {
            "material": {"name": "MOF5"},
            "adsorbates": {
                "components": names,
                "mixture": dict(zip(names, [0.3, 0.3, 0.4])),
            },
        }
        points = simplex_candidates(names, [0.1, 10], [300], 20)
        with tempfile.TemporaryDirectory() as d, patch(
            "modules.module_b_campaign.identity", return_value="physics"
        ):
            return initialize(c, Path(d) / "state.json", candidates=points, budget=15)

    def test_holdout_separate_and_budget(self):
        from modules.module_b_campaign import recommend

        s = self.state()
        self.assertEqual(len(s["holdout"]), 4)
        held = {p["candidate_id"] for p in s["holdout"]}
        self.assertNotIn(recommend(s)["candidate"]["candidate_id"], held)
        s["options"]["max_training_points"] = 0
        self.assertEqual(recommend(s)["status"], "budget_exhausted")

    def test_paper_pac_and_upstream_stop_rules_stay_distinct(self):
        from modules.module_b_campaign import recommend

        s = self.state()
        held = {p["candidate_id"] for p in s["holdout"]}
        s["results"] = [
            {**p, "precision_met": True}
            for p in s["candidates"]
            if p["candidate_id"] not in held
        ][:3]
        s["profile"] = {"components": s["components"], "beta": 0.02}
        prediction = {
            "kernels": {},
            "components": {
                c: [
                    {
                        "paper_relative_uncertainty": 0.01 if i < 18 else 0.03,
                        "transformed_std": 0.1 + i,
                    }
                    for i in range(20)
                ]
                for c in s["components"]
            },
        }
        s["holdout_results"] = [{"components": "must never be inspected for selection"}]
        with patch("modules.module_b_campaign.predict", return_value=prediction):
            s["mode"] = "paper"
            self.assertEqual(recommend(s)["status"], "pac_target_met")
            s["mode"] = "upstream_compat"
            selected = recommend(s)
            self.assertEqual(selected["status"], "request")
            self.assertNotIn(selected["candidate"]["candidate_id"], held)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn")
    def test_report_exports_all_components_pairs_and_separate_holdouts(self):
        import csv
        from modules.module_b_campaign import report

        s = self.state()
        held = {p["candidate_id"] for p in s["holdout"]}

        def observed(p):
            return {
                **p,
                "replicate_count": 3,
                "precision_met": True,
                "components": {
                    c: {
                        "mean_molecules": p["pressure_bar"] * y,
                        "standard_error_molecules": 0.01,
                    }
                    for c, y in p["mixture"].items()
                },
            }

        s["results"] = [
            observed(p) for p in s["candidates"] if p["candidate_id"] not in held
        ][:6]
        s["holdout_results"] = [observed(p) for p in s["holdout"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps(s))
            result = report(path)
            self.assertEqual(len(result["metrics"]), 6)
            with (Path(tmp) / "report/observations.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {r["quantity"] for r in rows},
                {"CO2", "CH4", "N2", "total", "CH4/CO2", "CH4/N2", "CO2/N2"},
            )
            self.assertEqual({r["role"] for r in rows}, {"training", "holdout"})
            self.assertGreater(
                (Path(tmp) / "report/surrogate_projections.png").stat().st_size, 1000
            )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn")
    def test_ternary_prediction(self):
        from modules.module_b_campaign import predict

        s = self.state()
        s["results"] = [
            {
                **p,
                "replicate_count": 3,
                "precision_met": True,
                "components": {
                    c: {
                        "mean_molecules": p["pressure_bar"] * y,
                        "standard_error_molecules": 0.01,
                    }
                    for c, y in p["mixture"].items()
                },
            }
            for p in s["candidates"][:6]
        ]
        result = predict(s, s["candidates"])
        self.assertEqual(set(result["components"]), set(s["components"]))
        self.assertTrue(
            all(
                np.isfinite(p["value"])
                for rows in result["components"].values()
                for p in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
