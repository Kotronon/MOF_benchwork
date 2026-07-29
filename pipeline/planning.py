from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from resolvers.forcefield_resolver import ForcefieldResolver
from resolvers.material_resolver import MaterialResolver
from resolvers.reference_resolver import ReferenceResolver
from resolvers.task_resolver import default_metrics_for_module, select_module as resolve_task_module
from resolvers.unitcell_resolver import UnitcellResolver

from pipeline.applicability import assess_applicability
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
    reference_config = config["benchmark"].get("reference", {})
    references = ReferenceResolver().resolve(
        material_id=material_match.material_id,
        charge_scheme=material_match.charge_scheme,
        forcefield=forcefield["framework"],
        components=config["adsorbates"]["components"],
        temperature_K=float(config["conditions"]["temperature_K"]),
        source=reference_config.get("source", "crafted"),
        material_aliases=[material_match.query, *material_match.aliases],
        pressures_bar=config["conditions"]["pressures_bar"],
        temperature_tolerance_K=float(reference_config.get("temperature_tolerance_K", 2.0)),
        max_nist_candidates=int(reference_config.get("max_nist_candidates", 5)),
        preferred_sources=reference_config.get("preferred_sources"),
        deduplicate_nist_by_doi=bool(reference_config.get("deduplicate_nist_by_doi", True)),
        exclude_nist_outliers=bool(reference_config.get("exclude_nist_outliers", True)),
        nist_max_loading_mol_per_kg=reference_config.get("nist_max_loading_mol_per_kg"),
        nist_max_reference_ratio=reference_config.get("nist_max_reference_ratio"),
        allowed_reference_basis=reference_config.get("allowed_reference_basis"),
        allow_unknown_reference_basis=bool(reference_config.get("allow_unknown_reference_basis", True)),
        allow_excess_reference_basis=bool(reference_config.get("allow_excess_reference_basis", True)),
        require_known_nist_article_source=bool(reference_config.get("require_known_nist_article_source", False)),
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
    applicability = assess_applicability(normalized, module, resolved)
    evaluation = _resolve_evaluation_config(
        normalized["evaluation"],
        resolved["material"]["material_id"],
        temperature_K=float(normalized["conditions"]["temperature_K"]),
    )

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
            "applicability": applicability,
        },
        "evaluation": evaluation,
        "convergence": normalized["convergence"],
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
            "save_dumps": normalized["output"]["save_dumps"],
            "dump_every_steps": normalized["output"]["dump_every_steps"],
            "save_plots": normalized["output"]["save_plots"],
            "save_csv": normalized["output"]["save_csv"],
        },
    }


def _resolve_evaluation_config(
    evaluation: dict[str, Any],
    material_id: str,
    temperature_K: float,
) -> dict[str, Any]:
    resolved = dict(evaluation)
    resolved["temperature_K"] = temperature_K
    if resolved.get("pore_volume_cm3_g") in ("auto", None):
        pore_volume = _crafted_pore_volume_cm3_g(material_id)
        resolved["pore_volume_cm3_g"] = pore_volume
        if pore_volume is not None:
            resolved.setdefault("pore_volume_source", None)
            resolved.setdefault("pore_volume_method", None)
            resolved["pore_volume_source"] = (
                resolved["pore_volume_source"]
                or "CRAFTED-2.0.0/RAC_DBSCAN/CRAFTED_MOF_geometric.csv:AV_cm^3/g"
            )
            resolved["pore_volume_method"] = resolved["pore_volume_method"] or "Zeo++ accessible volume"
    return resolved


def _crafted_pore_volume_cm3_g(material_id: str) -> float | None:
    path = Path("CRAFTED-2.0.0/RAC_DBSCAN/CRAFTED_MOF_geometric.csv")
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("FrameworkName") == material_id:
                value = row.get("AV_cm^3/g")
                return float(value) if value not in ("", None) else None
    return None
