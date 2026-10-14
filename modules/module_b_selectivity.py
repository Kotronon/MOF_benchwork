from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pipeline.materialization import materialize_benchmark
from pipeline.prepare import prepare_benchmark
from pipeline.runners import run_isotherm


def prepare_mixture(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Build a side-effect-free preparation plan for a classical mixture benchmark."""
    _validate_mixture_run_plan(run_plan)
    return prepare_benchmark(run_plan)


def run_mixture_isotherm(
    prepare_plan_or_materialized_plan: dict[str, Any],
    *,
    jobs: int = 1,
    materialized: bool = False,
) -> dict[str, Any]:
    """Materialize and run a Module-B mixture isotherm."""
    plan = (
        prepare_plan_or_materialized_plan
        if materialized
        else materialize_benchmark(prepare_plan_or_materialized_plan)
    )
    if not plan.get("parameters", {}).get("mixture"):
        raise ValueError("Module B mixture run requires parameters.mixture.")
    return run_isotherm(plan, jobs=jobs)


def evaluate_selectivity(
    component_mean_molecules: dict[str, float],
    gas_mole_fractions: dict[str, float],
) -> dict[str, Any]:
    """Calculate pairwise adsorption selectivity from adsorbed and gas mole fractions."""
    components = {str(component).upper(): float(value) for component, value in component_mean_molecules.items()}
    mixture = {str(component).upper(): float(value) for component, value in gas_mole_fractions.items()}
    if set(components) != set(mixture):
        raise ValueError("component_mean_molecules and gas_mole_fractions must contain the same components.")
    total_adsorbed = sum(components.values())
    result: dict[str, Any] = {}
    for left in components:
        for right in components:
            if left == right:
                continue
            key = f"{left}/{right}"
            if (
                total_adsorbed <= 0.0
                or components[right] <= 0.0
                or mixture[left] <= 0.0
                or mixture[right] <= 0.0
            ):
                result[key] = {
                    "value": None,
                    "reason": "undefined_zero_adsorbed_or_gas_fraction",
                }
                continue
            x_left = components[left] / total_adsorbed
            x_right = components[right] / total_adsorbed
            result[key] = {
                "value": (x_left / x_right) / (mixture[left] / mixture[right]),
                "adsorbed_mole_fraction_left": x_left,
                "adsorbed_mole_fraction_right": x_right,
                "gas_mole_fraction_left": mixture[left],
                "gas_mole_fraction_right": mixture[right],
            }
    return result


def build_candidate_pool(
    pressures_bar: list[float] | None = None,
    co2_mole_fractions: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic Module-B active-learning candidates for CO2/N2 mixtures."""
    pressures = pressures_bar or [0.1, 1.0, 5.0, 10.0, 20.0, 35.0]
    fractions = co2_mole_fractions or [0.05, 0.15, 0.5, 0.85]
    candidates = []
    for pressure in pressures:
        for co2_fraction in fractions:
            if not 0.0 < co2_fraction < 1.0:
                raise ValueError("CO2 mole fractions must be in the open interval (0, 1).")
            candidates.append(
                {
                    "candidate_id": f"p{_token(pressure)}_yco2_{_token(co2_fraction)}",
                    "pressure_bar": float(pressure),
                    "mixture": {
                        "CO2": float(co2_fraction),
                        "N2": float(1.0 - co2_fraction),
                    },
                    "status": "pending",
                }
            )
    return candidates


def initial_candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    """Select a small deterministic initial design spanning pressure and composition."""
    if not candidates:
        return []
    pressures = sorted({float(candidate["pressure_bar"]) for candidate in candidates})
    fractions = sorted({float(candidate["mixture"]["CO2"]) for candidate in candidates})
    selected_pairs = {
        (pressures[0], fractions[0]),
        (pressures[len(pressures) // 2], fractions[len(fractions) // 2]),
        (pressures[-1], fractions[-1]),
    }
    return [
        candidate["candidate_id"]
        for candidate in candidates
        if (float(candidate["pressure_bar"]), float(candidate["mixture"]["CO2"])) in selected_pairs
    ]


def select_next_candidate(campaign_state: dict[str, Any]) -> dict[str, Any] | None:
    """Select the next pending candidate without requiring scikit-learn."""
    candidates = campaign_state.get("candidates", [])
    completed = {
        result.get("candidate_id")
        for result in campaign_state.get("results", [])
        if result.get("candidate_id")
    }
    pending = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in completed and candidate.get("status", "pending") == "pending"
    ]
    if not pending:
        return None
    # Deterministic uncertainty proxy for V1: cover the sparsest pressure/composition grid first.
    return max(
        pending,
        key=lambda candidate: (
            _distance_to_completed(candidate, campaign_state.get("results", [])),
            float(candidate["pressure_bar"]),
            float(candidate["mixture"]["CO2"]),
        ),
    )


def initialize_campaign_state(
    output_path: str | Path,
    *,
    pressures_bar: list[float] | None = None,
    co2_mole_fractions: list[float] | None = None,
    completed_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create and write a JSON-serializable Module-B active-learning state."""
    candidates = build_candidate_pool(pressures_bar, co2_mole_fractions)
    results = completed_results or []
    completed_ids = {
        result.get("candidate_id")
        for result in results
        if result.get("candidate_id")
    }
    for candidate in candidates:
        if candidate["candidate_id"] in completed_ids:
            candidate["status"] = "completed"
    state = {
        "status": "initialized",
        "strategy": "deterministic_maximin_until_surrogate_available",
        "surrogate_model": "pending_optional_scikit_learn_gaussian_process",
        "candidates": candidates,
        "initial_candidate_ids": initial_candidate_ids(candidates),
        "results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def initialize_campaign_state_from_isotherm(
    isotherm_summary_path: str | Path,
    output_path: str | Path,
    *,
    pressures_bar: list[float] | None = None,
    co2_mole_fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Seed a Module-B active-learning state from an existing mixture run."""
    summary_path = Path(isotherm_summary_path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_results = mixture_results_from_isotherm_summary(data)
    state = initialize_campaign_state(
        output_path,
        pressures_bar=pressures_bar,
        co2_mole_fractions=co2_mole_fractions,
        completed_results=completed_results,
    )
    state["source_isotherm_summary"] = str(summary_path)
    Path(output_path).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def mixture_results_from_isotherm_summary(
    isotherm_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract compact campaign results from a Module-B isotherm summary."""
    results = []
    for point in isotherm_summary.get("results", []):
        summary = point.get("summary", {})
        components = summary.get("components", {})
        if not {"CO2", "N2"}.issubset(components):
            continue
        co2 = components["CO2"]
        n2 = components["N2"]
        y_co2 = co2.get("gas_phase_mole_fraction")
        y_n2 = n2.get("gas_phase_mole_fraction")
        pressure = point.get("pressure_bar")
        if pressure is None or y_co2 is None or y_n2 is None:
            continue
        selectivity = summary.get("selectivity", {}).get("CO2/N2", {})
        results.append(
            {
                "candidate_id": f"p{_token(float(pressure))}_yco2_{_token(float(y_co2))}",
                "pressure_bar": float(pressure),
                "mixture": {
                    "CO2": float(y_co2),
                    "N2": float(y_n2),
                },
                "mean_CO2": co2.get("mean_molecules"),
                "mean_N2": n2.get("mean_molecules"),
                "standard_error_CO2": co2.get("standard_error_molecules"),
                "standard_error_N2": n2.get("standard_error_molecules"),
                "selectivity_CO2_N2": selectivity.get("value"),
                "replicate_count": point.get("replicate_count"),
                "seeds": point.get("seeds", []),
                "converged": point.get("converged"),
                "status": "completed",
            }
        )
    return results


def append_mixture_result_csv(
    output_path: str | Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write compact mixture/selectivity results for active-learning campaigns."""
    fieldnames = [
        "candidate_id",
        "pressure_bar",
        "y_CO2",
        "y_N2",
        "mean_CO2",
        "mean_N2",
        "selectivity_CO2_N2",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _validate_mixture_run_plan(run_plan: dict[str, Any]) -> None:
    if run_plan.get("module", {}).get("id") != "B":
        raise ValueError("Module B mixture preparation requires run_plan.module.id == 'B'.")
    components = run_plan.get("adsorbates", {}).get("components", [])
    mixture = run_plan.get("adsorbates", {}).get("mixture")
    if len(components) < 2 or not mixture:
        raise ValueError("Module B requires at least two components and adsorbates.mixture.")


def _distance_to_completed(candidate: dict[str, Any], results: list[dict[str, Any]]) -> float:
    completed_points = [
        (
            float(result["pressure_bar"]),
            float(result["mixture"]["CO2"]),
        )
        for result in results
        if "pressure_bar" in result and "mixture" in result and "CO2" in result["mixture"]
    ]
    if not completed_points:
        return float(candidate["pressure_bar"])
    pressure = float(candidate["pressure_bar"])
    fraction = float(candidate["mixture"]["CO2"])
    return min(
        ((pressure - done_pressure) ** 2 + (fraction - done_fraction) ** 2) ** 0.5
        for done_pressure, done_fraction in completed_points
    )


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")
