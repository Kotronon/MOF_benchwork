from __future__ import annotations

from typing import Any


MODULES = {
    "A": {
        "id": "A",
        "name": "Baseline GCMC adsorption",
        "slug": "module_A_co2_isotherm",
    },
    "B": {
        "id": "B",
        "name": "Mixture adsorption / active learning",
        "slug": "module_B_mixture_selectivity",
    },
    "C": {
        "id": "C",
        "name": "Potential benchmark / Widom-Henry",
        "slug": "module_C_widom_henry",
    },
    "D": {
        "id": "D",
        "name": "Framework flexibility benchmark",
        "slug": "module_D_flexibility",
    },
}


DEFAULT_METRICS = {
    "A": ["co2_uptake", "isotherm", "error_to_reference", "runtime", "validity_range"],
    "B": ["component_uptake", "selectivity", "mre", "r2", "required_gcmc_points"],
    "C": ["widom_insertion_energy", "henry_region", "qst", "error_to_reference", "runtime_per_step"],
    "D": ["rigid_vs_flexible_isotherm", "structural_changes", "rmsd", "adsorption_change", "convergence"],
}


EXPLICIT_TASKS = {
    "a": "A",
    "module_a": "A",
    "adsorption": "A",
    "baseline": "A",
    "baseline_gcmc": "A",
    "gcmc": "A",
    "b": "B",
    "module_b": "B",
    "mixture": "B",
    "selectivity": "B",
    "active_learning": "B",
    "c": "C",
    "module_c": "C",
    "widom": "C",
    "henry": "C",
    "potential": "C",
    "potential_benchmark": "C",
    "d": "D",
    "module_d": "D",
    "flexibility": "D",
    "flexible": "D",
    "md_gcmc": "D",
}


def select_module(config: dict[str, Any]) -> dict[str, str]:
    task = str(config.get("benchmark", {}).get("task", "auto")).strip().lower().replace("-", "_")
    if task and task != "auto":
        module_id = EXPLICIT_TASKS.get(task)
        if module_id is None:
            raise ValueError(f"Unknown benchmark task {task!r}.")
        return MODULES[module_id].copy()

    module_id = _auto_select_module(config)
    return MODULES[module_id].copy()


def default_metrics_for_module(module_id: str) -> list[str]:
    try:
        return list(DEFAULT_METRICS[module_id])
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark module {module_id!r}.") from exc


def _auto_select_module(config: dict[str, Any]) -> str:
    simulation = config.get("simulation", {})
    benchmark = config.get("benchmark", {})
    adsorbates = config.get("adsorbates", {})

    method = str(simulation.get("method", "")).lower()
    framework = str(simulation.get("framework", "rigid")).lower()
    metrics = {str(metric).lower() for metric in benchmark.get("metrics", [])}
    components = adsorbates.get("components", [])
    mixture = adsorbates.get("mixture")

    if framework != "rigid" or "flex" in method or method in {"md/gcmc", "mlip-md/gcmc"}:
        return "D"
    if "widom" in method or "henry" in method or metrics & {"widom_insertion_energy", "henry_region", "qst"}:
        return "C"
    if mixture is not None or len(components) > 1 or metrics & {"selectivity", "component_uptake"}:
        return "B"
    return "A"
