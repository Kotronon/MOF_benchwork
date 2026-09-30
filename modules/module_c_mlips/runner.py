"""Execution runner for Module C potential comparisons."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from modules.module_c_mlips.comparison import compare_interaction_results
import numpy as np

from modules.module_c_mlips.interaction import (
    build_reference_interaction_result,
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
    reference_baseline: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every configuration with every backend."""

    if not configurations:
        raise ValueError("At least one configuration is required.")
    minimum_backends = 1 if reference_baseline else 2
    if len(backends) < minimum_backends:
        message = (
            "At least one potential backend is required."
            if minimum_backends == 1
            else "At least two potential backends are required."
        )
        raise ValueError(message)

    backend_names = [backend.name for backend in backends]
    if len(set(backend_names)) != len(backend_names):
        raise ValueError("Backend names must be unique.")
    if not reference_baseline and baseline_backend not in backend_names:
        raise ValueError(
            f"Baseline backend {baseline_backend!r} is not available."
        )
    if reference_baseline and baseline_backend in backend_names:
        raise ValueError(
            "The reference baseline name must differ from evaluated backend names."
        )

    started = perf_counter()
    configuration_reports = []
    framework_configurations = {}
    for configuration in configurations:
        cache_key = _framework_cache_key(configuration)
        framework_configurations.setdefault(
            cache_key,
            (
                build_framework_configuration(configuration),
                configuration.configuration_id,
            ),
        )
    framework_results: dict[tuple[str, str], Any] = {}
    framework_precompute_seconds = {}
    for backend in backends:
        backend_seconds = 0.0
        for cache_key, cached_framework in framework_configurations.items():
            framework_configuration, source_configuration_id = cached_framework
            try:
                result = backend.evaluate(framework_configuration)
            except Exception as exc:
                raise RuntimeError(
                    f"Backend {backend.name!r} failed for "
                    f"{source_configuration_id!r} during cached framework "
                    "evaluation."
                ) from exc
            backend_seconds += result.runtime_seconds
            framework_results[(backend.name, cache_key)] = replace(
                result,
                runtime_seconds=0.0,
                metadata={
                    **result.metadata,
                    "cached_framework_evaluation": True,
                    "precompute_runtime_seconds": result.runtime_seconds,
                },
            )
        framework_precompute_seconds[backend.name] = backend_seconds

    for configuration in configurations:
        results = (
            [
                build_reference_interaction_result(
                    configuration,
                    backend_name=baseline_backend,
                )
            ]
            if reference_baseline
            else []
        )
        force_comparison_mode = str(
            configuration.metadata.get("reference_force_mode", "interaction")
        )

        for backend in backends:
            try:
                cache_key = _framework_cache_key(configuration)
                result = evaluate_interaction(
                    configuration,
                    backend,
                    framework_result=framework_results[(backend.name, cache_key)],
                    force_comparison_mode=force_comparison_mode,
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
        "baseline_source": (
            "configuration_reference" if reference_baseline else "backend"
        ),
        "backends": backend_names,
        "configuration_count": len(configuration_reports),
        "runtime_seconds": perf_counter() - started,
        "shared_framework_cache": {
            "enabled": len(framework_configurations) < len(configurations),
            "configuration_count": len(configurations),
            "unique_framework_count": len(framework_configurations),
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


def _framework_cache_key(configuration: InteractionConfiguration) -> str:
    """Hash all framework fields that can affect a backend evaluation."""
    framework = build_framework_configuration(configuration)
    digest = sha256()
    digest.update("\0".join(framework.atoms.get_chemical_symbols()).encode("utf-8"))
    digest.update(np.asarray(framework.atoms.positions, dtype=np.float64).tobytes())
    digest.update(np.asarray(framework.atoms.cell.array, dtype=np.float64).tobytes())
    digest.update(np.asarray(framework.atoms.pbc, dtype=np.bool_).tobytes())
    digest.update(
        np.asarray(
            framework.atoms.get_initial_charges(),
            dtype=np.float64,
        ).tobytes()
    )
    forcefield_types = framework.atoms.arrays.get("forcefield_type")
    if forcefield_types is not None:
        digest.update("\0".join(map(str, forcefield_types)).encode("utf-8"))
    return digest.hexdigest()
