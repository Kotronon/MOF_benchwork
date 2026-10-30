"""Framework geometry and measured SR-cycle summaries (no synthetic results).

Cycle windows diagnose stationarity only. Correlated SR cycles are never
reported as independent replicates or used for a naive confidence interval.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from modules.module_d_flexibility import finite, integer


def structural_metrics(reference: Any, current: Any, framework_indices: list[int],
                       linker_dihedrals: list[tuple[int, int, int, int]] | None = None) -> dict[str, Any]:
    """Compare stable atom mappings; PBC-aware RMSD in the reference cell.

    Affine cell strain is removed from RMSD and reported separately as volume
    change. No best-fit rotation or translation alignment is applied. Callers
    must supply stable framework atom IDs; species equality alone cannot prove
    the correspondence of identical elements.
    """
    import numpy as np
    from ase.geometry import find_mic

    indices = framework_indices
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("framework_indices must be non-empty and unique.")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or
           i >= len(reference) or i >= len(current) for i in indices):
        raise ValueError("Framework atom index outside the supplied structures.")
    if not np.array_equal(reference.pbc, current.pbc) or not reference.pbc.all():
        raise ValueError("Matching fully periodic structures are required.")
    if not np.array_equal(reference.numbers[indices], current.numbers[indices]):
        raise ValueError("Framework species/order changed; provide a stable atom mapping.")
    for atoms in (reference, current):
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("Structures must contain finite coordinates and cells.")
        if atoms.cell.rank != 3 or atoms.get_volume() <= 0:
            raise ValueError("Structures need non-singular cells.")
    displacement = (current.get_scaled_positions(wrap=False)[indices] -
                    reference.get_scaled_positions(wrap=False)[indices]) @ reference.cell.array
    _, distances = find_mic(displacement, reference.cell, pbc=True)
    v0, v1 = reference.get_volume(), current.get_volume()
    rotations = []
    for quartet in linker_dihedrals or []:
        if len(quartet) != 4 or len(set(quartet)) != 4 or not set(quartet).issubset(indices):
            raise ValueError("Linker dihedrals require four distinct framework indices.")
        initial = reference.get_dihedral(*quartet, mic=True)
        final = current.get_dihedral(*quartet, mic=True)
        rotations.append(float((final - initial + 180.0) % 360.0 - 180.0))
    return {"framework_rmsd_A": float(np.sqrt(np.mean(distances**2))),
            "rmsd_convention": "minimum image in reference cell; affine strain removed; no alignment",
            "cell_volume_A3": float(v1), "relative_cell_volume_change": float(v1 / v0 - 1),
            "linker_rotation_deg": rotations,
            "pore_volume_A3": None}


def summarize_cycles(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = plan["benchmark"]["flexibility"]
    cases = {case["case_id"]: case for case in plan["cases"]}
    grouped: dict[str, dict[int, dict[str, float | None]]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id not in cases:
            raise ValueError(f"Unknown case_id: {case_id}")
        cycle = integer(row.get("cycle"), "cycle")
        if cycle > cases[case_id]["expected_cycles"]:
            raise ValueError(f"Cycle exceeds planned length for {case_id}.")
        values: dict[str, float | None] = {}
        for key in ("uptake_mol_kg", "framework_rmsd_A", "cell_volume_A3"):
            values[key] = finite(row.get(key), key, positive=key == "cell_volume_A3")
        for key in ("potential_energy_eV", "pore_volume_A3", "linker_rotation_deg", "runtime_s"):
            raw = row.get(key)
            if raw in (None, ""):
                values[key] = None
            elif key in {"potential_energy_eV", "linker_rotation_deg"}:
                values[key] = finite(raw, key, minimum=-math.inf)
            else:
                values[key] = finite(raw, key)
        group = grouped.setdefault(case_id, {})
        if cycle in group:
            raise ValueError(f"Duplicate cycle {cycle} for {case_id}.")
        group[cycle] = values
    summaries = []
    for case_id, case in cases.items():
        group = grouped.get(case_id, {})
        order = sorted(group)
        if order and order != list(range(1, max(order) + 1)):
            raise ValueError(f"Non-contiguous cycles for {case_id}; include the full history.")
        flexible = case["variant"] == "sr_nvt"
        discard = protocol["discard_cycles"] if flexible else 0
        retained = [group[i] for i in order if i > discard]
        complete = len(order) == case["expected_cycles"]
        summary = {**case, "observed_cycles": len(order), "retained_cycles": len(retained),
                   "complete": complete, "mean_uptake_mol_kg": None,
                   "cycle_standard_deviation_mol_kg": None, "stationarity": "insufficient_cycles"}
        if retained:
            loadings = [r["uptake_mol_kg"] for r in retained]
            summary["mean_uptake_mol_kg"] = mean(loadings)
            summary["cycle_standard_deviation_mol_kg"] = stdev(loadings) if len(loadings) > 1 else None
        window = protocol["convergence_window"]
        if not flexible:
            summary["stationarity"] = "not_applicable"
        elif len(retained) >= 2 * window:
            before, after = retained[-2 * window:-window], retained[-window:]
            drift = {key: abs(mean(r[key] for r in after) - mean(r[key] for r in before))
                     for key in ("uptake_mol_kg", "framework_rmsd_A", "cell_volume_A3")}
            drift["relative_cell_volume"] = drift["cell_volume_A3"] / mean(r["cell_volume_A3"] for r in before)
            passed = (drift["uptake_mol_kg"] <= protocol["uptake_drift_tolerance_mol_kg"] and
                      drift["framework_rmsd_A"] <= protocol["rmsd_drift_tolerance_A"] and
                      drift["relative_cell_volume"] <= protocol["volume_relative_drift_tolerance"])
            summary.update(stationarity="passed" if passed else "failed", window_drift=drift)
        summaries.append(summary)
    baseline = {(s["point_id"], s["seed"]): s for s in summaries if s["variant"] == "rigid"}
    comparisons = []
    for summary in summaries:
        if summary["variant"] != "sr_nvt":
            continue
        rigid = baseline[(summary["point_id"], summary["seed"])]
        if not summary["complete"] or not rigid["complete"]:
            continue
        delta = summary["mean_uptake_mol_kg"] - rigid["mean_uptake_mol_kg"]
        comparisons.append({"point_id": summary["point_id"], "seed": summary["seed"],
                            "relative_humidity": summary["relative_humidity"],
                            "rigid_uptake_mol_kg": rigid["mean_uptake_mol_kg"],
                            "flexible_uptake_mol_kg": summary["mean_uptake_mol_kg"],
                            "adsorption_change_mol_kg": delta,
                            "relative_adsorption_change": delta / rigid["mean_uptake_mol_kg"] if rigid["mean_uptake_mol_kg"] else None})
    return {"status": "complete" if all(s["complete"] for s in summaries) else "partial",
            "validation_status": "not_validated", "cases": summaries, "comparisons": comparisons,
            "notes": ["Complete means all planned records were imported, not physically validated.",
                      "SR cycles can be correlated; cycle standard deviation is not a confidence interval.",
                      "Passing two-window stationarity is not proof of equilibrium or adequate sampling.",
                      "Cell volume is not accessible pore volume.",
                      "Experimental-reference scoring is not yet implemented."]}


def evaluate_flexibility(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory)
    plan = json.loads((root / "module_d_plan.json").read_text())
    with (root / "cycle_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = summarize_cycles(plan, rows)
    # Analysis may be refreshed; raw data and the archived plan are never edited.
    destination = root / "evaluation"
    destination.mkdir(exist_ok=True)
    (destination / "module_d_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ("point_id", "seed", "relative_humidity", "rigid_uptake_mol_kg",
              "flexible_uptake_mol_kg", "adsorption_change_mol_kg", "relative_adsorption_change")
    with (destination / "rigid_vs_flexible.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["comparisons"])
    return result
