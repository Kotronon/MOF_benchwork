from __future__ import annotations

from pathlib import Path
from typing import Any

from resolvers.forcefield_resolver import ForcefieldResolver
from resolvers.material_resolver import MaterialResolver
from resolvers.reference_resolver import ReferenceResolver
from resolvers.task_resolver import default_metrics_for_module, select_module as resolve_task_module
from resolvers.unitcell_resolver import UnitcellResolver

from pipeline.config import normalize_config


def select_module(config: dict[str, Any]) -> dict[str, str]:
    """Select the benchmark module A-D from normalized config."""
    return resolve_task_module(config)


def resolve_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve material, force field, unit cells, and reference data."""
    material_config = config["material"]
    material_match = MaterialResolver().resolve(
        material_config["name"],
        material_config.get("charge_scheme", "auto"),
    )
    forcefield = ForcefieldResolver().resolve(
        config["simulation"]["forcefield"],
        config["adsorbates"]["components"],
    )
    unit_cells = UnitcellResolver().resolve(
        config["simulation"].get("unit_cells", "auto"),
        material_match.material_id,
    )
    references = ReferenceResolver().resolve(
        material_id=material_match.material_id,
        charge_scheme=material_match.charge_scheme,
        forcefield=forcefield["framework"],
        components=config["adsorbates"]["components"],
        temperature_K=float(config["conditions"]["temperature_K"]),
        source=config["benchmark"].get("reference", {}).get("source", "crafted"),
    )

    return {
        "material": {
            "query": material_match.query,
            "material_id": material_match.material_id,
            "cif_path": str(material_match.cif_path),
            "charge_scheme": material_match.charge_scheme,
            "matched_by": material_match.matched_by,
            "aliases": list(material_match.aliases),
        },
        "forcefield": forcefield,
        "unit_cells": unit_cells,
        "references": references,
    }


def build_run_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable dry-run plan for the benchmark."""
    normalized = normalize_config(config)
    module = select_module(normalized)
    resolved = resolve_benchmark(normalized)

    metrics = normalized["benchmark"].get("metrics") or default_metrics_for_module(module["id"])
    output_directory = normalized["output"].get("directory") or f"outputs/{module['slug']}"

    return {
        "status": "planned",
        "module": module,
        "material": resolved["material"],
        "adsorbates": normalized["adsorbates"],
        "conditions": normalized["conditions"],
        "simulation": {
            **normalized["simulation"],
            "unit_cells": resolved["unit_cells"],
        },
        "benchmark": {
            **normalized["benchmark"],
            "metrics": metrics,
        },
        "resources": {
            "cif_path": resolved["material"]["cif_path"],
            "forcefield": resolved["forcefield"],
            "references": resolved["references"],
        },
        "outputs": {
            "directory": output_directory,
            "report_json": str(Path(output_directory) / "benchmark_report.json"),
            "table_csv": str(Path(output_directory) / "benchmark_table.csv"),
            "plot_png": str(Path(output_directory) / "isotherm_plot.png"),
            "run_id": normalized["output"]["run_id"],
            "overwrite": normalized["output"]["overwrite"],
            "save_logs": normalized["output"]["save_logs"],
            "save_plots": normalized["output"]["save_plots"],
            "save_csv": normalized["output"]["save_csv"],
        },
    }
