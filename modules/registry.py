from __future__ import annotations

from typing import Any

from modules.base import BenchmarkModule, UnsupportedModule


def _normalized_task_alias(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _module_a_score(config: dict[str, Any]) -> int:
    return 10


def _module_b_score(config: dict[str, Any]) -> int:
    benchmark = config.get("benchmark", {})
    adsorbates = config.get("adsorbates", {})
    metrics = {str(metric).lower() for metric in benchmark.get("metrics", [])}
    components = adsorbates.get("components", [])
    mixture = adsorbates.get("mixture")

    if len(components) > 1 and mixture:
        return 70
    if len(components) > 1 and metrics & {"selectivity", "component_uptake"}:
        return 65
    return 0


def _module_c_score(config: dict[str, Any]) -> int:
    simulation = config.get("simulation", {})
    benchmark = config.get("benchmark", {})
    method = str(simulation.get("method", "")).lower()
    metrics = {str(metric).lower() for metric in benchmark.get("metrics", [])}
    if isinstance(benchmark.get("potential_benchmark"), dict):
        return 80
    if "widom" in method or "henry" in method:
        return 80
    if metrics & {"widom_insertion_energy", "henry_region", "qst"}:
        return 80
    return 0


def _module_d_score(config: dict[str, Any]) -> int:
    simulation = config.get("simulation", {})
    method = str(simulation.get("method", "")).lower()
    framework = str(simulation.get("framework", "rigid")).lower()
    if framework != "rigid":
        return 90
    return 0


MODULE_REGISTRY: tuple[BenchmarkModule, ...] = (
    BenchmarkModule(
        id="A",
        name="Baseline GCMC adsorption",
        slug="module_A_co2_isotherm",
        task_aliases=(
            "a",
            "module_a",
            "adsorption",
            "baseline",
            "baseline_gcmc",
            "gcmc",
        ),
        default_metrics=(
            "co2_uptake",
            "isotherm",
            "error_to_reference",
            "runtime",
            "validity_range",
        ),
        selection_score=_module_a_score,
    ),
    BenchmarkModule(
        id="B",
        name="Mixture adsorption / active learning",
        slug="module_B_mixture_selectivity",
        task_aliases=(
            "b",
            "module_b",
            "mixture",
            "selectivity",
            "active_learning",
        ),
        default_metrics=(
            "component_uptake",
            "selectivity",
            "mre",
            "r2",
            "required_gcmc_points",
        ),
        selection_score=_module_b_score,
    ),
    BenchmarkModule(
        id="C",
        name="Potential benchmark / Widom-Henry",
        slug="module_C_widom_henry",
        task_aliases=(
            "c",
            "module_c",
            "widom",
            "henry",
            "potential",
            "potential_benchmark",
        ),
        default_metrics=(
            "widom_insertion_energy",
            "henry_region",
            "qst",
            "error_to_reference",
            "runtime_per_step",
        ),
        selection_score=_module_c_score,
    ),
    UnsupportedModule(
        id="D",
        name="Framework flexibility benchmark",
        slug="module_D_flexibility",
        task_aliases=(
            "d",
            "module_d",
            "flexibility",
            "flexible",
            "md_gcmc",
        ),
        default_metrics=(
            "rigid_vs_flexible_isotherm",
            "structural_changes",
            "rmsd",
            "adsorption_change",
            "convergence",
        ),
        selection_score=_module_d_score,
    ),
)


def registered_modules() -> tuple[BenchmarkModule, ...]:
    """Return all benchmark modules in deterministic selection order."""
    return MODULE_REGISTRY


def module_by_id(module_id: str) -> BenchmarkModule:
    """Look up a registered module by its stable module id."""
    normalized = module_id.strip().upper()
    for module in MODULE_REGISTRY:
        if module.id == normalized:
            return module
    raise ValueError(f"Unknown benchmark module {module_id!r}.")


def module_for_task(task: str) -> BenchmarkModule:
    """Look up a module from an explicit task alias."""
    normalized = _normalized_task_alias(task)
    for module in MODULE_REGISTRY:
        if normalized in module.task_aliases:
            return module
    raise ValueError(f"Unknown benchmark task {task!r}.")


def select_registered_module(config: dict[str, Any]) -> BenchmarkModule:
    """Select the best matching module from normalized benchmark config."""
    candidates = [
        (module.selection_score(config), module)
        for module in MODULE_REGISTRY
    ]
    score, module = max(candidates, key=lambda item: item[0])
    if score <= 0:
        return module_by_id("A")
    return module
