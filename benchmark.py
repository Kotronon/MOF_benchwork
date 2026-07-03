from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from resolvers.forcefield_resolver import ForcefieldResolver
from resolvers.material_resolver import MaterialResolver
from resolvers.reference_resolver import ReferenceResolver
from resolvers.task_resolver import default_metrics_for_module, select_module as resolve_task_module
from resolvers.unitcell_resolver import UnitcellResolver


DEFAULT_PRESSURES_BAR = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]


def load_benchmark_data(file_path: str | Path) -> dict[str, Any]:
    """Load benchmark input data from a JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data_to_dict(data)


def save_benchmark_data(file_path: str | Path, data: dict[str, Any]) -> None:
    """Save benchmark data to a JSON file."""
    with Path(file_path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def data_to_dict(data: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """Convert legacy list-style benchmark data to a dictionary."""
    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return {item["name"]: item for item in data if isinstance(item, dict) and "name" in item}

    raise TypeError("Benchmark data must be a dictionary or a list of dictionaries.")


def normalize_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Apply V1 defaults and normalize older benchmark.json shapes."""
    config = deepcopy(data_to_dict(raw_config))

    material = _require_mapping(config, "material")
    material.setdefault("cif_source", material.get("structure_file", "auto"))
    material.setdefault("charge_scheme", "auto")
    material.setdefault("source", "crafted")

    adsorbates = config.setdefault("adsorbates", {})
    if not isinstance(adsorbates, dict):
        raise TypeError("'adsorbates' must be an object.")
    components = adsorbates.get("components", ["CO2"])
    if isinstance(components, str):
        components = [components]
    if not components:
        components = ["CO2"]
    adsorbates["components"] = [str(component).upper() for component in components]
    adsorbates.setdefault("mixture", None)

    conditions = config.setdefault("conditions", {})
    if not isinstance(conditions, dict):
        raise TypeError("'conditions' must be an object.")
    conditions.setdefault("temperature_K", 298.15)
    conditions.setdefault("pressures_bar", DEFAULT_PRESSURES_BAR)
    _validate_pressures(conditions["pressures_bar"])

    benchmark = config.setdefault("benchmark", {})
    if not isinstance(benchmark, dict):
        raise TypeError("'benchmark' must be an object.")
    benchmark.setdefault("task", "auto")
    benchmark.setdefault("metrics", [])
    benchmark.setdefault(
        "reference",
        {
            "source": "crafted",
            "material": material.get("name"),
            "adsorbate": adsorbates["components"][0],
        },
    )

    simulation = config.setdefault("simulation", {})
    if not isinstance(simulation, dict):
        raise TypeError("'simulation' must be an object.")
    simulation["engine"] = str(simulation.get("engine", "LAMMPS")).upper()
    simulation["method"] = str(simulation.get("method", _default_method(config))).upper()
    simulation["forcefield"] = _normalize_forcefield(simulation.get("forcefield", "UFF"))
    simulation.setdefault("framework", "rigid")
    simulation.setdefault("units", "real")
    simulation.setdefault("atom_style", "full")
    simulation.setdefault("pair_style", "lj/cut/coul/long")
    simulation.setdefault("cutoff_A", 12.0)
    simulation.setdefault("kspace_style", "pppm")
    simulation.setdefault("kspace_accuracy", 1e-5)
    simulation.setdefault("unit_cells", simulation.get("supercell", "auto"))

    output = config.setdefault("output", {})
    if not isinstance(output, dict):
        raise TypeError("'output' must be an object.")
    output.setdefault("directory", None)
    output.setdefault("save_logs", True)
    output.setdefault("save_plots", True)
    output.setdefault("save_csv", True)

    return config


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
            "save_logs": normalized["output"]["save_logs"],
            "save_plots": normalized["output"]["save_plots"],
            "save_csv": normalized["output"]["save_csv"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a MOF benchmark run from benchmark.json.")
    parser.add_argument("config", nargs="?", default="benchmark.json", help="Path to benchmark JSON input.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without running LAMMPS.")
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("V1 only supports dry-run planning. Re-run with --dry-run.")

    config = load_benchmark_data(args.config)
    run_plan = build_run_plan(config)
    print(json.dumps(run_plan, indent=2))
    return 0


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"'{key}' must be an object.")
    if key == "material" and not value.get("name"):
        raise ValueError("'material.name' is required.")
    return value


def _normalize_forcefield(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        return {
            "framework": value,
            "adsorbate": "auto",
            "cross_interactions": "auto",
        }
    if isinstance(value, dict):
        return {
            "framework": str(value.get("framework", "UFF")),
            "adsorbate": str(value.get("adsorbate", "auto")),
            "cross_interactions": str(value.get("cross_interactions", "auto")),
        }
    raise TypeError("'simulation.forcefield' must be a string or object.")


def _default_method(config: dict[str, Any]) -> str:
    if "widom" in config and "gcmc" not in config:
        return "WIDOM"
    return "GCMC"


def _validate_pressures(pressures_bar: Any) -> None:
    if not isinstance(pressures_bar, list) or not pressures_bar:
        raise ValueError("'conditions.pressures_bar' must be a non-empty list.")
    for pressure in pressures_bar:
        if not isinstance(pressure, (int, float)) or pressure <= 0:
            raise ValueError("'conditions.pressures_bar' must contain positive numbers.")


if __name__ == "__main__":
    raise SystemExit(main())
