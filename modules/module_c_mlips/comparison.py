"""Comparison utilities for evaluating different potential backends."""

from __future__ import annotations

from math import sqrt
from typing import Any

from modules.module_c_mlips.interaction import InteractionResult


def compare_interaction_results(
    results: list[InteractionResult],
    *,
    baseline_backend: str | None = None,
) -> dict[str, Any]:
    """Compare multiple backends evaluated on one identical configuration."""
    if len(results) < 2:
        raise ValueError("At least two interaction results are required.")

    _validate_results(results)
    baseline = _select_baseline(results, baseline_backend)
    candidates = [result for result in results if result is not baseline]

    return {
        "schema_version": 1,
        "configuration_id": baseline.configuration_id,
        "baseline_backend": baseline.backend,
        "results": {
            result.backend: result.to_dict()
            for result in results
        },
        "comparisons": [
            _compare_candidate(baseline, candidate)
            for candidate in candidates
        ],
    }


def _validate_results(results: list[InteractionResult]) -> None:
    configuration_ids = {result.configuration_id for result in results}
    if len(configuration_ids) != 1:
        raise ValueError(
            "All interaction results must use the same configuration_id."
        )

    backends = [result.backend for result in results]
    if len(set(backends)) != len(backends):
        raise ValueError("Each compared backend must have a unique name.")

    force_counts = {
        len(result.interaction_forces_ev_per_angstrom)
        for result in results
    }
    if len(force_counts) != 1:
        raise ValueError(
            "All interaction results must contain the same number of forces."
        )


def _select_baseline(
    results: list[InteractionResult],
    baseline_backend: str | None,
) -> InteractionResult:
    if baseline_backend is None:
        return results[0]

    matches = [result for result in results if result.backend == baseline_backend]
    if not matches:
        raise ValueError(
            f"Baseline backend {baseline_backend!r} is not present in the results."
        )
    return matches[0]


def _compare_candidate(
    baseline: InteractionResult,
    candidate: InteractionResult,
) -> dict[str, Any]:
    force_differences = [
        [
            candidate_component - baseline_component
            for baseline_component, candidate_component in zip(
                baseline_force,
                candidate_force,
                strict=True,
            )
        ]
        for baseline_force, candidate_force in zip(
            baseline.interaction_forces_ev_per_angstrom,
            candidate.interaction_forces_ev_per_angstrom,
            strict=True,
        )
    ]
    force_components = [
        component
        for force in force_differences
        for component in force
    ]
    component_count = len(force_components)
    force_mae = (
        sum(abs(component) for component in force_components) / component_count
        if component_count
        else 0.0
    )
    force_rmse = (
        sqrt(
            sum(component * component for component in force_components)
            / component_count
        )
        if component_count
        else 0.0
    )
    maximum_force_difference = max(
        (abs(component) for component in force_components),
        default=0.0,
    )
    energy_difference = (
        candidate.interaction_energy_ev - baseline.interaction_energy_ev
    )
    runtime_ratio = (
        candidate.runtime_seconds / baseline.runtime_seconds
        if baseline.runtime_seconds > 0.0
        else None
    )

    return {
        "candidate_backend": candidate.backend,
        "energy_difference_ev": energy_difference,
        "absolute_energy_difference_ev": abs(energy_difference),
        "force_mae_ev_per_angstrom": force_mae,
        "force_rmse_ev_per_angstrom": force_rmse,
        "maximum_force_difference_ev_per_angstrom": maximum_force_difference,
        "force_differences_ev_per_angstrom": force_differences,
        "baseline_runtime_seconds": baseline.runtime_seconds,
        "candidate_runtime_seconds": candidate.runtime_seconds,
        "runtime_difference_seconds": (
            candidate.runtime_seconds - baseline.runtime_seconds
        ),
        "runtime_ratio_candidate_to_baseline": runtime_ratio,
    }
