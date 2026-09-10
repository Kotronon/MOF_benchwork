from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline.comparision import compare_variant_results, get_directory
from pipeline.config import normalize_config
from pipeline.planning import build_run_plan, select_module
from pipeline.materialization import materialize_benchmark
from pipeline.prepare import prepare_benchmark
from pipeline.runners import run_isotherm


SIMULATION_VARIANT_KEYS = {
    "forcefield",
    "cutoff_A",
    "kspace_style",
    "kspace_accuracy",
    "pair_style",
    "pair_modify_shift",
    "unit_cells",
    "cell_representation",
    "minimum_image_policy",
}

SIMULATION_VARIANT_ALIASES = {
    "pppm_accuracy": "kspace_accuracy",
}


def build_variant_plans(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build resolved run plans for all benchmark variants."""
    variant_plans = []
    for variant_config in build_variant_configs(config):
        run_plan = build_run_plan(variant_config)
        run_plan["variant"] = variant_config["benchmark"]["variant"]
        variant_plans.append(run_plan)
    return variant_plans


def build_variant_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build normalized config copies with one variant override applied each."""
    normalized = normalize_config(config)
    module = select_module(normalized)
    variants = normalized.get("benchmark", {}).get("variants", [])
    if not variants:
        raise ValueError("'benchmark.variants' must contain at least one variant.")

    variant_configs = []
    base_output = normalized["output"]
    base_run_id = base_output.get("run_id") or "variant_run"
    base_output_directory = base_output.get("directory") or f"outputs/{module['slug']}"

    for variant in variants:
        name = str(variant.get("name", "")).strip()
        if not name:
            raise ValueError("Each variant in 'benchmark.variants' needs a non-empty name.")

        variant_config = deepcopy(normalized)
        variant_config["benchmark"]["module"] = module["id"]
        variant_config["benchmark"]["variant"] = {"name": name}

        overrides = {}
        for key, value in variant.items():
            if key == "name":
                continue

            target_key = SIMULATION_VARIANT_ALIASES.get(key, key)
            if target_key not in SIMULATION_VARIANT_KEYS:
                allowed = sorted([*SIMULATION_VARIANT_KEYS, *SIMULATION_VARIANT_ALIASES])
                raise ValueError(
                    f"Unknown variant key {key!r} in variant {name!r}. "
                    f"Allowed keys: {allowed}."
                )
            variant_config["simulation"][target_key] = value
            overrides[target_key] = value

        variant_config["output"]["directory"] = str(
            Path(base_output_directory) / "runs" / base_run_id / name
        )
        variant_config["output"]["run_id"] = None
        variant_config["benchmark"]["variant"] = {"name": name, "overrides": overrides}
        variant_configs.append(variant_config)

    return variant_configs


def run_variant_benchmark(config: dict[str, Any], jobs: int = 1) -> dict[str, Any]:
    """Run the benchmark for each variant in the config."""
    variant_plans = build_variant_plans(config)
    results = []
    for run_plan in variant_plans:
        prepare_plan = prepare_benchmark(run_plan)
        materialized = materialize_benchmark(prepare_plan)
        result = run_isotherm(materialized, jobs=jobs)
        results.append(
            {
                "variant": run_plan["variant"],
                "working_directory": materialized["working_directory"],
                "result": result,
            }
        )
    comparison = compare_variant_results(
        config,
        get_directory(config) / "variant_comparison.json",
    )
    return {
        "status": "completed",
        "variant_count": len(results),
        "results": results,
        "comparison": comparison,
    }
