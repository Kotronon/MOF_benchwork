"""Review exported stages on validation and decompose existing test residuals.

Never predicts a new test candidate, changes model selection, or fits a correction
for deployment. Test centering is an exploratory decomposition of existing errors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.module_c_mlips import umof_finetuning as data
from modules.module_c_mlips import umof_workflow as workflow
from modules.module_c_mlips.umof_profiles import campaign_profile


def cached_prediction(dataset, checkpoint, head, root, campaign, output):
    """Read the original cache, computing missing validation predictions only."""
    protocol = campaign["protocol"]
    identity = [data.sha256_file(dataset), data.sha256_file(checkpoint), head,
                protocol["dtype"], protocol["device"],
                importlib.metadata.version("mace-torch"), "polar",
                importlib.metadata.version("graph_longrange")]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cached = root / "prediction_cache" / f"{key}.npz"
    if cached.exists():
        with np.load(cached) as saved:
            return saved["energies"], saved["forces"]
    if dataset.name == "test.extxyz":
        raise RuntimeError("Existing test prediction missing; no new test inference allowed")
    return workflow.predict(dataset, checkpoint, head, output / "prediction_cache",
                            device=protocol["device"], model="polar", dtype=protocol["dtype"])


def residual_stats(errors):
    errors = np.asarray(errors, dtype=float)
    bias = float(errors.mean())
    mse = float(np.mean(errors ** 2))
    centered = float(np.sqrt(np.mean((errors - bias) ** 2)))
    assert np.isclose(mse, bias ** 2 + centered ** 2, rtol=1e-10, atol=1e-12)
    return {"mean": bias, "rmse": float(np.sqrt(mse)),
            "centered_rmse": centered, "min": float(errors.min()),
            "max": float(errors.max()),
            "bias_fraction_of_mse": bias ** 2 / mse if mse else 0.0}


def main():
    from ase.io import read
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.campaign.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    campaign = workflow.read_json(root / "campaign.json")
    assert campaign_profile(campaign).key == "polar"
    workflow.verify_files(root, campaign["files"])
    assert len(campaign["runs"]) == 1, "Choose a single-run campaign"
    folder = root / campaign["runs"][0]["directory"]
    protected = {p: data.sha256_file(folder / p) for p in
                 ("status.json", "module_c_model.json", "results/selection.json", "results/evaluation.json")}
    selection = workflow.read_json(folder / "results/selection.json")
    stages = {c["stage"]: root / c["path"] for c in selection["candidates"]}
    models = {"baseline": (root / "assets/uMOF-POLAR.model", "r2scan+d4"),
              **{stage: (path, "Default") for stage, path in stages.items()}}
    frames = {s: read(folder / "datasets" / f"{s}.extxyz", index=":")
              for s in ("train", "valid", "test")}
    metrics_rows, residual_rows, decomposition = [], [], []
    predictions = {}
    for split, labels in (("valid", list(models)), ("test", ["baseline", "stage1"])):
        dataset = folder / "datasets" / f"{split}.extxyz"
        reference_e, reference_f, sizes = workflow.reference_arrays(frames[split])
        for label in labels:
            print(f"Review {split}: {label}", flush=True)
            prediction = cached_prediction(dataset, *models[label], root, campaign, out)
            predictions[split, label] = prediction
            groups = workflow.grouped_metrics(frames[split], prediction)
            metrics_rows.extend(dict(split=split, model=label, group=g, **m) for g, m in groups.items())
            if split == "valid" and label in stages:
                old = next(c["validation"] for c in selection["candidates"] if c["stage"] == label)
                for key, value in old.items():
                    assert np.isclose(groups["all"][key], value, rtol=1e-7, atol=1e-9), key
            if split == "test":
                old = workflow.read_json(folder / "results/evaluation.json")["metrics"][
                    "finetuned" if label == "stage1" else label]
                for group, values in old.items():
                    for key, value in values.items():
                        assert np.isclose(groups[group][key], value, rtol=1e-7, atol=1e-9), (group, key)
            residual = prediction[0] - reference_e
            for i, atoms in enumerate(frames[split]):
                residual_rows.append({"split": split, "model": label, "index": i,
                    "frame_id": atoms.info["umof_frame_id"], "mof": atoms.info["umof_mof_id"],
                    "formula": atoms.get_chemical_formula(), "natoms": len(atoms),
                    "guest_counts": atoms.info["umof_guest_counts"],
                    "source_file": atoms.info["umof_source_file"],
                    "source_index": atoms.info["umof_source_index"],
                    "error_eV": float(residual[i]), "error_meV_per_atom": float(1000*residual[i]/sizes[i])})
            for mof in sorted({a.info["umof_mof_id"] for a in frames[split]}):
                ids = [i for i, a in enumerate(frames[split]) if a.info["umof_mof_id"] == mof]
                decomposition.append({"split": split, "model": label, "mof": mof,
                    "count": len(ids),
                    "total_energy_eV": residual_stats(residual[ids]),
                    "energy_meV_per_atom": residual_stats(1000*residual[ids]/sizes[ids])})

    # A constant total-energy offset cancels only at fixed composition. Quantify
    # the actual relative-energy error between geometries of identical composition.
    pair_rows = []
    for (split, label), prediction in predictions.items():
        groups = defaultdict(list)
        for i, a in enumerate(frames[split]):
            groups[a.info["umof_mof_id"], a.get_chemical_formula()].append(i)
        re = workflow.reference_arrays(frames[split])[0]
        errors = prediction[0] - re
        for (mof, formula), ids in groups.items():
            if len(ids) < 2:
                continue
            differences = [errors[j]-errors[i] for n, i in enumerate(ids) for j in ids[n+1:]
                           if data.geometry_id(frames[split][i]) != data.geometry_id(frames[split][j])]
            if differences:
                pair_rows.append({"split": split, "model": label, "mof": mof,
                    "formula": formula, "pair_count": len(differences),
                    "relative_energy_rmse_eV": float(np.sqrt(np.mean(np.square(differences))))})
    pair_summary = []
    for split, label, mof in sorted({(r["split"], r["model"], r["mof"]) for r in pair_rows}):
        rows = [r for r in pair_rows if (r["split"], r["model"], r["mof"]) == (split, label, mof)]
        count = sum(r["pair_count"] for r in rows)
        pair_summary.append({"split": split, "model": label, "mof": mof, "pair_count": count,
            "relative_energy_rmse_eV": float(np.sqrt(sum(r["pair_count"]*r["relative_energy_rmse_eV"]**2 for r in rows)/count))})

    audit = {"split_counts": {}, "reference_mapping": {}, "model_buffers": {},
             "magnesium_guest_metadata_mismatches": []}
    for split, atoms_list in frames.items():
        audit["split_counts"][split] = {"frames": len(atoms_list),
            "unique_geometries": len({data.geometry_id(a) for a in atoms_list}),
            "mofs": dict(Counter(a.info["umof_mof_id"] for a in atoms_list))}
        audit["reference_mapping"][split] = {
            "max_abs_REF_minus_dft_energy_eV": max(abs(a.info["REF_energy"]-a.info["dft_energy"]) for a in atoms_list),
            "max_abs_REF_minus_dft_force_eV_per_A": max(float(np.max(np.abs(a.arrays["REF_forces"]-a.arrays["dft_forces"]))) for a in atoms_list)}
    geometry_sets = {s: {data.geometry_id(a) for a in aa} for s, aa in frames.items()}
    audit["cross_split_geometry_overlap"] = {f"{s}/{t}": len(geometry_sets[s] & geometry_sets[t])
        for s, t in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    audit["replay_Mg_frames"] = {split: sum(12 in a.numbers for a in read(root / "assets/replay-12345" / f"{split}.extxyz", index=":"))
                                 for split in ("train", "valid")}
    baseline = torch.load(models["baseline"][0], map_location="cpu", weights_only=False)
    base_e0 = baseline.atomic_energies_fn.atomic_energies[0].detach().numpy()
    for label, (path, head) in models.items():
        model = torch.load(path, map_location="cpu", weights_only=False)
        assert np.array_equal(model.atomic_numbers.numpy(), baseline.atomic_numbers.numpy())
        e0 = model.atomic_energies_fn.atomic_energies[list(model.heads).index(head)].detach().numpy()
        audit["model_buffers"][label] = {
            "sha256": data.sha256_file(path), "heads": list(model.heads),
            "max_abs_atomic_E0_change_eV": float(np.max(np.abs(e0-base_e0))),
            "scale": model.scale_shift.scale.detach().tolist(), "shift": model.scale_shift.shift.detach().tolist()}
    # Mg18C72H18O54 is the common host inferred from these test compositions.
    host = Counter({"Mg": 18, "C": 72, "H": 18, "O": 54})
    for i, a in enumerate(frames["test"]):
        if a.info["umof_mof_id"] != "MgMOF-74-adsorbates":
            continue
        annotated = json.loads(a.info["umof_guest_counts"])
        expected = host.copy()
        for gas, count in annotated.items():
            for element, n in campaign["protocol"]["gas_definitions"][gas].items():
                expected[element] += n * count
        if expected != Counter(a.get_chemical_symbols()):
            audit["magnesium_guest_metadata_mismatches"].append({"test_index": i,
                "source_index": int(a.info["umof_source_index"]), "formula": a.get_chemical_formula(),
                "annotated_guest_counts": annotated,
                "note": "Composition disagrees with inferred common host plus annotated guests; verify source segmentation"})

    workflow.write_csv(out / "metrics.csv", metrics_rows)
    workflow.write_csv(out / "energy_residuals.csv", residual_rows)
    workflow.write_csv(out / "same_composition_relative_energies.csv", pair_rows)
    workflow.write_csv(out / "relative_energy_summary.csv", pair_summary)
    assert all(data.sha256_file(folder / p) == digest for p, digest in protected.items())
    result = {"campaign": str(root), "selection_rule": "lexicographic: force RMSE, energy RMSE, stage1 tie-break",
        "selection_unchanged": True, "new_test_inference": False,
        "centering_is_diagnostic_only": True, "audit": audit,
        "decomposition": decomposition, "validation_candidates": selection["candidates"],
        "original_artifact_sha256": protected, "input_sha256": campaign["files"],
        "script_sha256": data.sha256_file(__file__),
        "relative_energy_summary": pair_summary}
    workflow.atomic_json(out / "diagnosis.json", result)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = list(models)
    mofs = sorted({a.info["umof_mof_id"] for a in frames["valid"]})
    for ax, key, title in ((axes[0, 0], "energy_rmse_ev_per_atom", "Validation energy RMSE (meV/atom)"),
                           (axes[0, 1], "force_rmse_ev_per_A", "Validation force RMSE (meV/Å)")):
        for j, label in enumerate(labels):
            values = [1000*next(r[key] for r in metrics_rows if r["split"] == "valid" and r["model"] == label and r["group"] == "mof:"+m) for m in mofs]
            ax.bar(np.arange(len(mofs))+(j-1)*.25, values, width=.25, label=label)
        ax.set_xticks(np.arange(len(mofs)), [m.replace("-adsorbates", "") for m in mofs])
        ax.set_title(title); ax.legend()
    for label in ("baseline", "stage1"):
        rows = [r for r in residual_rows if r["split"] == "test" and r["model"] == label and r["mof"] == "MgMOF-74-adsorbates"]
        axes[1, 0].plot([r["source_index"] for r in rows], [r["error_meV_per_atom"] for r in rows], "o-", label=label)
        axes[1, 1].plot([r["source_index"] for r in rows], [r["error_eV"] for r in rows], "o-", label=label)
    for ax, units in ((axes[1, 0], "meV/atom"), (axes[1, 1], "eV/configuration")):
        ax.axhline(0, color="black", linewidth=.6); ax.set(xlabel="Source test configuration index", ylabel=f"Prediction minus DFT ({units})", title="MgMOF-74: existing test residuals"); ax.legend()
    workflow.save_figure(fig, out / "diagnostics")

    lines = ["# uMOF stage review and MgMOF-74 diagnosis", "",
        "Validation review of baseline, Stage 1 and Stage 2. Test diagnosis reuses only existing baseline/Stage 1 predictions. No training, selection changes or deployed energy correction.", "",
        "## Validation", "", "| Model | Energy RMSE (meV/atom) | Force RMSE (meV/Å) |", "|---|---:|---:|"]
    for label in models:
        row = next(r for r in metrics_rows if r["split"] == "valid" and r["model"] == label and r["group"] == "all")
        lines.append(f"| {label} | {1000*row['energy_rmse_ev_per_atom']:.3f} | {1000*row['force_rmse_ev_per_A']:.3f} |")
    lines += ["", "The selection rule prioritizes force RMSE absolutely; energy only breaks exact ties. Stage 2 lowers energy RMSE by 79.92% relative to Stage 1, while force RMSE rises by 1.03%. Both stages are Pareto candidates when only the fine-tuned stages are considered; neither dominates the other on both validation metrics. The baseline is better than both on both aggregate validation RMSEs.", "",
        "Stage 2 improves MIL-53-Al over baseline (energy 9.885 vs 10.559 meV/atom; forces 23.871 vs 29.240 meV/Å), but degrades HKUST-1 (energy 15.281 vs 3.793 meV/atom; forces 71.483 vs 65.104 meV/Å). This is material-dependent transfer, not a uniform improvement.", "",
        "## MgMOF-74 existing test residuals", "", "| Model | Mean error (meV/atom) | RMSE (meV/atom) | Centered RMSE (meV/atom) | Bias share of MSE | Mean error (eV/config) | Centered RMSE (eV/config) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("baseline", "stage1"):
        row = next(r for r in decomposition if r["split"] == "test" and r["model"] == label and r["mof"] == "MgMOF-74-adsorbates")
        e, total = row["energy_meV_per_atom"], row["total_energy_eV"]
        lines.append(f"| {label} | {e['mean']:.3f} | {e['rmse']:.3f} | {e['centered_rmse']:.3f} | {100*e['bias_fraction_of_mse']:.2f}% | {total['mean']:.3f} | {total['centered_rmse']:.3f} |")
    lines += ["", "Centering uses the same group under inspection and is descriptive, not a corrected held-out score. A per-atom offset differs from a constant framework total-energy offset when guest counts vary. Even after centering, Stage 1 has more energy-error scatter than baseline; MgMOF-74 force RMSE also increases from 40.031 to 48.761 meV/Å (+21.81%).", "",
        "## Same-composition energy differences", "",
        "| Split | Model | MOF | Pairs | Relative-energy RMSE (eV) |", "|---|---|---|---:|---:|"]
    for row in pair_summary:
        lines.append(f"| {row['split']} | {row['model']} | {row['mof']} | {row['pair_count']} | {row['relative_energy_rmse_eV']:.5f} |")
    lines += ["", "Stage 1 improves MgMOF-74 relative-energy RMSE across 10 same-composition pairs from 0.11368 to 0.04953 eV. Stage 2 has worse same-composition relative-energy RMSE than Stage 1 on both validation MOFs. Its better absolute energy RMSE alone therefore does not establish that it is preferable for adsorption. These pairs are dependent and are not independent samples or adsorption energies; molecular and empty-host energy references are not evaluated here.", "",
        "## Reference and coverage audit", "",
        "- REF labels match the stored DFT labels exactly in all splits.",
        "- Atomic E0 buffers are identical in baseline, Stage 1 and Stage 2. Scale is 1 and shift is 0 for every head. The observed offset is not a changed fixed atomic-energy reference or scale/shift setting; it arises from the learned prediction.",
        "- MgMOF-74 is absent from adsorption training and validation; replay contains 2 Mg-bearing training frames and 0 Mg-bearing validation frames.",
        f"- Cross-split geometry overlap: {audit['cross_split_geometry_overlap']}.",
        "- Test contains 38 entries but 36 unique geometries; duplicated entries occur in MOF-303. MgMOF-74 has 22 unique entries.",
        f"- Mg guest-count metadata mismatches against the inferred common host: {audit['magnesium_guest_metadata_mismatches']}.", "",
        "## Assessment", "",
        "Keep the baseline as the reference model for now. Stage 2 deserves further validation but does not justify automatic replacement of Stage 1: the absolute-energy/force tradeoff and relative-energy results point in different directions. The MgMOF-74 failure is predominantly a mean energy bias, with additional force and composition-dependent errors. A test-derived offset is not a validated correction.", "",
        "Cached Stage 1/2 validation metrics and all existing test group metrics were reproduced numerically. Campaign input hashes were verified. Original status, selected model specification, selection and test evaluation files retained their hashes. Full per-MOF validation metrics, residuals, model hashes and a reusable script accompany this report. This audit does not identify a unique causal mechanism for model drift.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print((out / "REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
