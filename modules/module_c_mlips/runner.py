"""Execution runner for Module C potential comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from modules.module_c_mlips.comparison import compare_interaction_results
import numpy as np

from modules.module_c_mlips.interaction import (
    build_framework_configuration,
    evaluate_interaction,
)
from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import PotentialBackend
from pipeline.config import save_benchmark_data


def run_potential_comparison(
    configurations: Sequence[InteractionConfiguration],
    backends: Sequence[PotentialBackend],
    *,
    baseline_backend: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every configuration with every backend."""

    if not configurations:
        raise ValueError("At least one configuration is required.")
    if len(backends) < 2:
        raise ValueError("At least two potential backends are required.")

    backend_names = [backend.name for backend in backends]
    if len(set(backend_names)) != len(backend_names):
        raise ValueError("Backend names must be unique.")
    if baseline_backend not in backend_names:
        raise ValueError(
            f"Baseline backend {baseline_backend!r} is not available."
        )

    started = perf_counter()
    configuration_reports = []
    shared_framework = _shared_framework_configuration(configurations)
    framework_results = {}
    framework_precompute_seconds = {}
    if shared_framework is not None:
        for backend in backends:
            try:
                result = backend.evaluate(shared_framework)
            except Exception as exc:
                raise RuntimeError(
                    f"Backend {backend.name!r} failed for "
                    f"{configurations[0].configuration_id!r} "
                    "during shared-framework evaluation."
                ) from exc
            framework_precompute_seconds[backend.name] = result.runtime_seconds
            framework_results[backend.name] = replace(
                result,
                runtime_seconds=0.0,
                metadata={
                    **result.metadata,
                    "reused_across_configurations": True,
                    "precompute_runtime_seconds": result.runtime_seconds,
                },
            )

    for configuration in configurations:
        results = []

        for backend in backends:
            try:
                result = evaluate_interaction(
                    configuration,
                    backend,
                    framework_result=framework_results.get(backend.name),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Backend {backend.name!r} failed for "
                    f"{configuration.configuration_id!r}."
                ) from exc

            results.append(result)

        configuration_report = compare_interaction_results(
            results,
            baseline_backend=baseline_backend,
        )
        configuration_report.update(
            {
                "material": configuration.material,
                "adsorbate": configuration.adsorbate,
                "region": configuration.region,
                "source": configuration.source,
                "metadata": configuration.metadata,
            }
        )
        configuration_reports.append(configuration_report)

    report = {
        "status": "completed",
        "baseline_backend": baseline_backend,
        "backends": backend_names,
        "configuration_count": len(configuration_reports),
        "runtime_seconds": perf_counter() - started,
        "shared_framework_cache": {
            "enabled": shared_framework is not None,
            "precompute_runtime_seconds": framework_precompute_seconds,
        },
        "configurations": configuration_reports,
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report["output_path"] = str(output)
        save_benchmark_data(output, report)

    return report


def _shared_framework_configuration(
    configurations: Sequence[InteractionConfiguration],
) -> InteractionConfiguration | None:
    """Return one framework when all configurations contain the same host."""
    reference = build_framework_configuration(configurations[0])
    reference_types = reference.atoms.arrays.get("forcefield_type")
    for configuration in configurations[1:]:
        current = build_framework_configuration(configuration)
        current_types = current.atoms.arrays.get("forcefield_type")
        if (
            current.atoms.get_chemical_symbols()
            != reference.atoms.get_chemical_symbols()
            or not np.allclose(current.atoms.positions, reference.atoms.positions)
            or not np.allclose(current.atoms.cell.array, reference.atoms.cell.array)
            or not np.allclose(
                current.atoms.get_initial_charges(),
                reference.atoms.get_initial_charges(),
            )
            or not np.array_equal(current_types, reference_types)
        ):
            return None
    return reference
