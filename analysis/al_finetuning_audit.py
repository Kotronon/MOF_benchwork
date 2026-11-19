"""Audit the two methodology questions using archived results, without inference.

Run from the repository root with the Python environment containing ASE:
    python3 -m analysis.al_finetuning_audit --output reports/al_finetuning_review_2026-11-17
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analysis.module_c_closure import ROOT, Sources, digest, dump, write_csv
from analysis.module_c_local_review import RUNS, audit_predictions
from modules.module_c_mlips.active_learning.models import ActiveLearningSettings
from modules.module_c_mlips.active_learning.training import classify_validation_metrics


AL = ("outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/"
      "zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning")
B_AL = "outputs/module_B_active_learning_mof5_20bar/production/mof5/campaign.json"
B_RANDOM = "outputs/module_B_mof5_al_comparison/random/production/mof5/campaign.json"
B_REPORT = "outputs/module_B_mof5_al_comparison/report/comparison_report.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cp2k_audit(sources):
    state = sources.json(AL + "/state.json")
    config = sources.json("input_json_files/benchmark_zif8_co2_module_c_active_learning.json")
    settings = ActiveLearningSettings.from_dict(
        config["benchmark"]["potential_benchmark"]["active_learning"])
    split_ids = {k: set(state[k + "_ids"]) for k in ("training", "validation", "test")}
    require(all(not split_ids[a] & split_ids[b] for a, b in
                (("training", "validation"), ("training", "test"), ("validation", "test"))),
            "CP2K candidate IDs overlap between fitting, validation and assessment")
    labels = {}
    for path in sorted((ROOT / AL).glob("iteration_*/cp2k/*/label.json")):
        label = sources.json(str(path.relative_to(ROOT)))
        require(label["candidate_id"] not in labels, "Repeated CP2K candidate ID")
        labels[label["candidate_id"]] = label
    require(len(labels) == state["labeled_count"], "CP2K label count mismatch")
    require(set(labels) == set.union(*split_ids.values()), "CP2K split/label mismatch")
    rounds, selections = [], []
    for iteration in range(state["iteration"] + 1):
        prefix = f"{AL}/iteration_{iteration:02d}"
        saved = sources.json(prefix + "/validation_metrics.json")
        require(set(saved["test_ids"]) == split_ids["test"], "Assessment set changed")
        classified = classify_validation_metrics(saved["metrics"], settings)
        require(classified["status"] == saved["status"], "AL gate does not reproduce")
        commands = sources.json(prefix + "/training/training_commands.json")["commands"]
        require(all("--test_file" not in c["command"] for c in commands),
                "CP2K assessment set passed as training test_file")
        rounds.append({"iteration": iteration, "status": classified["status"],
                       **{k: saved["metrics"][k] for k in (
                           "energy_mae_ev", "force_mae_ev_per_A",
                           "maximum_committee_energy_std_ev", "maximum_committee_force_std_ev_per_A")}})
        if iteration:
            selected = sources.json(prefix + "/selection_manifest.json")["candidates"]
            require(len(selected) == settings.configurations_per_round, "Unexpected AL batch size")
            require(all(c["candidate_id"] in split_ids["training"] for c in selected),
                    "New AL labels outside the fitting set")
            selections.append({"iteration": iteration, "selected": len(selected),
                               "categories": dict(Counter(c["candidate_type"] for c in selected))})
    predictions = [sources.json(AL + f"/evaluation/umof_comparison/finetuned_{i}_predictions.json")
                   for i in range(settings.committee_size)]
    energy_errors, force_errors, energy_stds, force_stds = [], [], [], []
    for candidate_id in state["test_ids"]:
        energies = np.array([p[candidate_id]["energy_ev"] for p in predictions])
        forces = np.array([p[candidate_id]["forces_ev_per_A"] for p in predictions])
        reference = labels[candidate_id]
        energy_errors.append(abs(energies.mean() - reference["energy_ev"]))
        force_errors.extend(np.linalg.norm(forces.mean(axis=0) - reference["forces_ev_per_A"], axis=1))
        energy_stds.append(energies.std(ddof=0))
        force_stds.append(np.linalg.norm(forces.std(axis=0, ddof=0), axis=1).max())
    computed = dict(zip(("energy_mae_ev", "force_mae_ev_per_A", "maximum_committee_energy_std_ev",
                         "maximum_committee_force_std_ev_per_A"),
                        map(float, (np.mean(energy_errors), np.mean(force_errors),
                                    max(energy_stds), max(force_stds)))))
    require(all(np.isclose(value, rounds[-1][key], rtol=1e-10, atol=1e-12)
                for key, value in computed.items()), "Final CP2K committee errors do not reproduce")
    return {"state": {k: state[k] for k in ("status", "iteration", "stable_rounds", "candidate_count",
                                            "labeled_count", "selected_cutoff_Ry")},
            "split_counts": {k: len(v) for k, v in split_ids.items()},
            "rounds": rounds, "selections": selections, "recomputed_final_metrics": computed,
            "all_final_metrics_reproduced": True, "candidate_ids_disjoint": True,
            "cutoff": sources.json(AL + "/cutoff_test/cutoff_convergence.json"),
            "assessment_reused_for_stopping_and_member_ranking": True}


def umof_audit(sources):
    data = {"fine_tuning": []}
    for label, root, fraction in RUNS:
        for row in sources.csv(f"{root}/{fraction}/seed-12345/results/metrics.csv"):
            data["fine_tuning"].append({"run": label, "model": row["model"], "group": row["group"],
                **{k.replace("_ev_", "_meV_"): float(v) * 1000 for k, v in row.items()
                   if k.startswith(("energy_", "force_"))}})
    audit = audit_predictions(sources, data)
    summary = []
    for label, root, fraction in RUNS:
        folder = f"{root}/{fraction}/seed-12345"
        evaluation = sources.json(folder + "/results/evaluation.json")
        selection = sources.json(folder + "/results/selection.json")
        require(selection["selection_split"] == "validation" and not selection["test_used"],
                "uMOF checkpoint selection used test labels")
        manifest = sources.json(folder + "/datasets/dataset_manifest.json")
        row = {"run": label, "training_configurations": manifest["train_configuration_count"],
               "actual_training_fraction": manifest["actual_training_fraction"],
               "selected_stage": selection["selected"]["stage"]}
        for model in ("baseline", "finetuned"):
            metrics = evaluation["metrics"][model]["all"]
            row[model + "_energy_rmse_meV_per_atom"] = 1000 * metrics["energy_rmse_ev_per_atom"]
            row[model + "_force_rmse_meV_per_A"] = 1000 * metrics["force_rmse_ev_per_A"]
        summary.append(row)
    return {"summary": summary, "audit": audit,
            "dft_reference_status": sources.json("outputs/module_C_umof_dft_reference_025pct_mof5/run_status.json"),
            "dft_reference_manifest": sources.json("outputs/module_C_umof_dft_reference_025pct_mof5/manifest.json"),
            "energy_diagnostic": sources.json("outputs/module_C_umof_energy_diagnostic_025pct_mof5/summary.json")}


def module_b_audit(sources):
    comparison = sources.json(B_REPORT)
    states = {"al": sources.json(B_AL), "random": sources.json(B_RANDOM)}
    summaries = {}
    for role, state in states.items():
        seen = {r["candidate_id"] for r in state["results"]}
        held = {p["candidate_id"] for p in state["holdout"]}
        require(not seen & held, "Module B holdouts occur in training")
        require(len(seen) == comparison["current"][role]["training_points"],
                "Module B report does not match current training-state count")
        summaries[role] = {"status": state["status"], "mode": state["mode"],
                           "options": state["options"], "training_points": len(seen),
                           "candidate_points": len(state["candidates"]),
                           "initial_points": len(state["initial_design"]), "holdouts": len(held)}
    require(states["al"]["initial_design"] == states["random"]["initial_design"],
            "AL/random initial designs differ")
    return {"states": summaries, "training_holdout_ids_disjoint": True,
            "reported_current_results": comparison["current"],
            "validation_status": comparison["validation_status"],
            "completed_holdouts": comparison["completed_holdouts"],
            "criteria": comparison["criteria"], "test_set_sha256": comparison["test_set_sha256"],
            "scope": "Current archived GP scores read from the report, not refitted in this audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = Sources(ROOT)
    result = {"scope": "Existing code and archived predictions; no DFT, inference, fine-tuning or GCMC",
              "cp2k_al": cp2k_audit(sources), "umof": umof_audit(sources),
              "module_b": module_b_audit(sources)}
    for name in ("modules/module_b/campaign.py", "modules/module_b/cluster.py",
                 "modules/module_c_mlips/active_learning/candidates.py",
                 "modules/module_c_mlips/active_learning/workflow.py",
                 "modules/module_c_mlips/active_learning/training.py",
                 "modules/module_c_mlips/umof_finetuning.py", "modules/module_c_mlips/umof_workflow.py"):
        sources.path(name)
    result["isotherm_evidence"] = {
        "returned_42_jobs": sources.json("reports/module_c_cluster_return_2026-11-10/results.json"),
        "guarded_mof5_pilot": sources.json("reports/umof_mof5_review_2026-11-13/results.json"),
        "scope": "Prior trace reconstruction reports reused, no new trace reconstruction in this audit"}
    for path, expected in sources.files.items():
        require(digest(ROOT / path) == expected, f"Source changed during audit: {path}")
    dump(args.output / "evidence.json", result)
    write_csv(args.output / "cp2k_rounds.csv", result["cp2k_al"]["rounds"])
    write_csv(args.output / "umof_test_metrics.csv", result["umof"]["summary"])
    dump(args.output / "provenance.json", {"source_sha256": sources.files,
        "analysis_sha256": digest(__file__), "sources_unchanged": True,
        "python": __import__("sys").version, "numpy": np.__version__})
    print(json.dumps({"output": str(args.output), "cp2k_metrics_reproduced": 4,
                      "umof_metrics_reproduced": len(result["umof"]["audit"]["prediction_checks"]),
                      "source_files_verified": len(sources.files)}, indent=2))


if __name__ == "__main__":
    main()
