from __future__ import annotations

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from pipeline.evaluate import build_evaluation_rows, evaluate_all_references, evaluate_isotherm, sim_results_to_csv
from pipeline.mixture_evaluation import assess_reference, build_mixture_evaluation, corrected_summary, framework_mass
from pipeline.mixture_statistics import fieller_interval, mean_interval
from pipeline.runners import _aggregate_replicates
from modules.module_b_active_learning import DEFAULTS, compare_sampling_strategies, predict_components, recommend, propose_campaign, ingest_campaign, _context
from modules.module_b_selectivity import build_candidate_pool


def replicates(xs=(2., 2.1, 1.9), ys=(1., 1.05, .95), pressure=1.):
    return [{"pressure_bar": pressure, "seed": i + 1,
             "summary": {"mean_adsorbates": x+y, "components": {
                 "CO2": {"mean_molecules": x, "gas_phase_mole_fraction": .15},
                 "N2": {"mean_molecules": y, "gas_phase_mole_fraction": .85}}}}
            for i, (x, y) in enumerate(zip(xs, ys))]


def fixture(root):
    path = root / "isotherm_summary.json"
    path.write_text(json.dumps({"results": [], "replicate_results": replicates(),
                                "convergence": {"relative_ci95_target": .2}}))
    (root / "prepare_summary.json").write_text(json.dumps({"prepare_plan": {
        "module": {"id": "B"}, "parameters": {"material_id": "IRMOF-1", "temperature_K": 298.15},
        "convergence": {"minimum_replicates": 3, "relative_ci95_target": .2}}}))
    (root / "evaluated_isotherm.csv").write_text("framework_mass_amu\n1000\n1000\n")
    return path


def nist_reference():
    return {"adsorbent": {"name": "MOF-5"}, "temperature": 298.15,
            "isotherm_type": "absolute", "compositionType": "molefraction",
            "adsorptionUnits": "mol/kg", "pressureUnits": "bar",
            "adsorbates": [{"name": "Carbon Dioxide", "InChIKey": "co2"},
                            {"name": "Nitrogen", "InChIKey": "n2"}],
            "isotherm_data": [{"pressure": p, "species_data": [
                {"InChIKey": "co2", "composition": .15, "adsorption": 2*p},
                {"InChIKey": "n2", "composition": .85, "adsorption": p}],
                "total_adsorption": 999} for p in [.5, 2.]]}


class MixtureUncertaintyTests(unittest.TestCase):
    def test_components_fail_despite_precise_total(self):
        point = _aggregate_replicates(replicates((1., 2., 3.), (3., 2., 1.)), {"relative_ci95_target": .2})[0]
        self.assertEqual(point["summary"]["relative_ci95_half_width"], 0)
        self.assertFalse(point["converged"])

    def test_fieller_uses_paired_covariance(self):
        positive = fieller_interval([2., 4., 6.], [1., 2., 3.], 1., 1.)
        negative = fieller_interval([6., 4., 2.], [1., 2., 3.], 1., 1.)
        self.assertAlmostEqual(positive["ci95_lower"], 2.)
        self.assertAlmostEqual(positive["ci95_upper"], 2.)
        self.assertGreater(negative["ci95_half_width"], 0.5)

    def test_fieller_unbounded_zero_and_single_seed(self):
        self.assertEqual(fieller_interval([1., 2., 3.], [0., 0., 1.], 1., 4.303)["ci95_status"], "unbounded")
        self.assertIsNone(fieller_interval([1., 2., 3.], [0., 0., 0.], 1., 4.303)["ci95_upper"])
        self.assertEqual(fieller_interval([1.], [1.], 1., None)["ci95_status"], "insufficient_replicates")

    def test_student_t_interval_and_duplicate_seed(self):
        interval = mean_interval([1., 2., 3.], 4.303)
        self.assertAlmostEqual(interval["ci95_half_width"], 4.303 / math.sqrt(3))
        data = replicates()
        data[1]["seed"] = data[0]["seed"]
        with self.assertRaises(ValueError):
            _aggregate_replicates(data)


class MixtureEvaluationTests(unittest.TestCase):
    def test_total_is_sum_of_components_not_legacy_counter(self):
        data = replicates()
        for row in data:
            row["summary"]["mean_adsorbates"] = 999
        self.assertAlmostEqual(_aggregate_replicates(data)[0]["summary"]["mean_adsorbates"], 3.)

    def test_all_entrypoints_reject_pure_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture(root)
            refdir = root / "source" / "references"
            refdir.mkdir(parents=True)
            ref = refdir / "co2.csv"
            ref.write_text("# pressure[Pa],mean_volume[mol/kg],mean_error[mol/kg]\n100000,100,0\n")
            rows = build_evaluation_rows(path, reference_csv=ref)
            self.assertEqual({r["quantity"] for r in rows}, {"CO2", "N2", "total", "CO2/N2", "N2/CO2"})
            self.assertTrue(all(r["comparison_error"] is None for r in rows))
            self.assertAlmostEqual(next(r["value"] for r in rows if r["quantity"] == "CO2"), 2.)
            sim_results_to_csv(path, root / "test.csv", reference_csv=ref)
            with (root / "test.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 5)
            for report in (evaluate_isotherm(path, reference_csv=ref), evaluate_all_references(root)):
                self.assertIsNone(report["mean_comparison_relative_error_percent"])
                self.assertIn("Referenzvalidierung offen", report["status"])

    def test_reference_checks_and_species_identifier_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture(root)
            ref = root / "mixture.json"
            ref.write_text(json.dumps(nist_reference()))
            result = build_mixture_evaluation(path, reference_csv=ref)
            self.assertTrue(result["reference_assessments"][0]["accepted"])
            # The deliberately wrong total_adsorption must never replace species data.
            co2 = next(r for r in result["rows"] if r["quantity"] == "CO2")
            self.assertAlmostEqual(co2["reference_value"], 2.)
            self.assertAlmostEqual(co2["comparison_error"], 0.)
            for field, value, reason in [("temperature", 350, "temperature_mismatch"),
                                         ("isotherm_type", "", "adsorption_basis_unknown"),
                                         ("adsorbent", {}, "material_metadata_missing"),
                                         ("adsorbent", {"name": "Cu-BTC"}, "material_mismatch"),
                                         ("compositionType", "", "gas_composition_metadata_missing")]:
                data = nist_reference()
                data[field] = value
                ref.write_text(json.dumps(data))
                assessment = build_mixture_evaluation(path, reference_csv=ref)["reference_assessments"][0]
                self.assertFalse(assessment["accepted"])
                self.assertIn(reason, assessment["reasons"])

    def test_pressure_extrapolation_and_composition_interpolation_forbidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            ref = Path(temporary) / "ref.json"
            data = nist_reference()
            ref.write_text(json.dumps(data))
            args = ({"path": str(ref)}, {"material_id": "IRMOF-1", "temperature_K": 298.15}, {"CO2": .15, "N2": .85})
            self.assertIn("pressure_outside_matching_composition_range", assess_reference(*args, 3., 1000, {})["reasons"])
            data["isotherm_data"][1]["species_data"][0]["composition"] = .5
            ref.write_text(json.dumps(data))
            self.assertFalse(assess_reference(*args, 1., 1000, {})["accepted"])

    def test_mass_fallback_and_absence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture(root)
            self.assertEqual(framework_mass(path)[0], 1000.)
            (root / "evaluated_isotherm.csv").write_text("framework_mass_amu\n1000\n2000\n")
            result = build_mixture_evaluation(path)
            self.assertIsNone(result["framework_mass_amu"])
            self.assertEqual(result["rows"][0]["unit"], "molecules/cell")

    def test_repeated_evaluation_preserves_source_and_marks_old_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture(root)
            old = root / "evaluation_report.md"
            old.write_text("# Isotherm Evaluation\nold invalid comparison\n")
            before = path.read_bytes()
            first = evaluate_isotherm(path)
            second = evaluate_isotherm(path)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(old.read_text().count("ÜBERHOLT"), 1)


def synthetic_state():
    candidates = build_candidate_pool([.1, 1., 5., 10.], [.15, .5, .85])
    rows = [{**c, "mean_CO2": 2*c["pressure_bar"]*c["mixture"]["CO2"],
             "mean_N2": .4*c["pressure_bar"]*c["mixture"]["N2"],
             "standard_error_CO2": .001, "standard_error_N2": .001,
             "replicate_count": 3, "converged": True} for c in candidates]
    return {"candidates": candidates, "results": rows, "options": deepcopy(DEFAULTS), "pending_request": None}


class CampaignImportTests(unittest.TestCase):
    def test_import_idempotence_context_and_pending_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = fixture(root)
            data = json.loads(summary.read_text())
            data["status"] = "completed"
            summary.write_text(json.dumps(data))
            prep = json.loads((root / "prepare_summary.json").read_text())
            prep["prepare_plan"]["parameters"]["forcefield"] = "UFF"
            (root / "prepare_summary.json").write_text(json.dumps(prep))
            state = {"context": _context(summary), "results": [], "history": [], "imports": [],
                     "candidates": build_candidate_pool([1.], [.15]), "pending_request": None}
            path = root / "state.json"
            path.write_text(json.dumps(state))
            first = ingest_campaign(path, summary)
            second = ingest_campaign(path, summary)
            self.assertEqual(first, second)
            self.assertEqual(len(second["results"]), 1)
            data["wall_time_seconds"] = 1
            summary.write_text(json.dumps(data))
            second["pending_request"] = {"candidate_id": "p5_yco2_0p5"}
            path.write_text(json.dumps(second))
            with self.assertRaisesRegex(ValueError, "pending candidate"):
                ingest_campaign(path, summary)
            second["pending_request"] = None
            path.write_text(json.dumps(second))
            prep["prepare_plan"]["parameters"]["temperature_K"] = 300
            (root / "prepare_summary.json").write_text(json.dumps(prep))
            with self.assertRaisesRegex(ValueError, "context differs"):
                ingest_campaign(path, summary)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional scikit-learn unavailable")
class ActiveLearningTests(unittest.TestCase):
    def test_noise_aware_gp_produces_finite_nonnegative_intervals(self):
        state = synthetic_state()
        train = state["results"][:6]
        predicted = predict_components(train, state["candidates"], state["candidates"])
        for values in predicted["components"].values():
            self.assertTrue(all(0 <= x["lower"] <= x["value"] <= x["upper"] and math.isfinite(x["upper"]) for x in values))
        noisy = deepcopy(train)
        for r in noisy:
            r["standard_error_CO2"] = 20.
        uncertain = predict_components(noisy, state["candidates"], state["candidates"])
        self.assertGreater(uncertain["components"]["CO2"][0]["log1p_std"], predicted["components"]["CO2"][0]["log1p_std"])

    def test_priority_retry_then_coverage_then_gp(self):
        state = synthetic_state()
        state["results"] = [r for r in state["results"] if r["mixture"]["CO2"] == .15]
        state["results"][0]["converged"] = False
        self.assertEqual(recommend(state)["action"], "improve_precision")
        state["results"][0]["precision_retries"] = 1
        coverage = recommend(state)
        self.assertNotEqual(coverage["candidate"]["mixture"]["CO2"], .15)
        state = synthetic_state()
        state["results"] = state["results"][:6]
        state["options"]["relative_uncertainty_target"] = 1e-6
        result = recommend(state)
        self.assertIn("kernels", result)
        self.assertEqual(result["action"], "sample")
        self.assertNotIn(result["candidate"]["candidate_id"], {r["candidate_id"] for r in state["results"]})

    def test_repeated_propose_keeps_pending_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            state = synthetic_state()
            state["results"][0]["converged"] = False
            state.update(history=[], base_config={"conditions": {}, "adsorbates": {}, "output": {},
                         "simulation": {"cycles": 100, "initialization_cycles": 25}})
            path.write_text(json.dumps(state))
            first, second = propose_campaign(path), propose_campaign(path)
            self.assertEqual(first["request"], second["request"])
            self.assertEqual(len(list((path.parent / "requests").glob("*.json"))), 1)
            request = json.loads(Path(first["request"]["config_path"]).read_text())
            self.assertEqual(request["simulation"]["cycles"], 400)

    def test_offline_validation_keeps_holdout_out_and_matches_budgets(self):
        state = synthetic_state()
        holdout = [r["candidate_id"] for r in state["results"][-2:]]
        report = compare_sampling_strategies(state, state["results"], holdout, [3, 4], repeats=1)
        self.assertEqual({r["strategy"] for r in report["results"]}, {"active_learning", "maximin", "random"})
        for row in report["results"]:
            self.assertTrue(set(row["training_ids"]).isdisjoint(holdout))
            self.assertEqual(len(row["training_ids"]), row["point_budget"])
            self.assertEqual(row["seed_budget"], 3*row["point_budget"])

    def test_budget_and_insufficient_validation_are_not_success(self):
        state = synthetic_state()
        self.assertEqual(recommend(state)["status"], "candidate_pool_exhausted")
        with self.assertRaises(ValueError):
            compare_sampling_strategies(state, state["results"][:4], [], [3])


if __name__ == "__main__":
    unittest.main()
