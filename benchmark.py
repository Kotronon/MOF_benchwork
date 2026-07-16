from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

from converter.cif_to_lammps_data import convert_cif_to_lammps_data, load_framework_structure, plan_cif_to_lammps_data
from converter.forcefield_to_lammps import (
    build_lammps_forcefield,
    load_forcefield_parameters,
    plan_forcefield_to_lammps,
    write_lammps_forcefield_include,
)
from converter.molecule_to_lammps_template import build_molecule_template, parse_crafted_molecule_def, plan_molecule_template
from engines.lammps_runner import build_lammps_command
from resolvers.forcefield_resolver import ForcefieldResolver
from resolvers.material_resolver import MaterialResolver
from resolvers.reference_resolver import ReferenceResolver
from resolvers.task_resolver import default_metrics_for_module, select_module as resolve_task_module
from resolvers.unitcell_resolver import UnitcellResolver
from parsers.lammps_log_parser import get_log, parse_lammps_log, summarize_adsorption, log_to_json


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
    
def prepare_benchmark(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Build a central side-effect-free preparation plan."""
    module_id = run_plan.get("module", {}).get("id")
    if module_id != "A":
        raise ValueError(f"Prepare currently supports module 'A' only, got {module_id!r}.")

    output_dir = Path(run_plan["outputs"]["directory"])
    working_dir = output_dir / "work"
    input_dir = working_dir / "inputs"
    data_dir = working_dir / "data"
    molecule_dir = working_dir / "molecules"
    forcefield_dir = working_dir / "forcefield"
    log_dir = working_dir / "logs"

    framework_data = data_dir / f"{run_plan['material']['material_id']}.data"
    forcefield_include = forcefield_dir / "forcefield.in"
    run0_input = input_dir / "in.run0"
    gcmc_test_input = input_dir / "in.gcmc_test"
    input_scripts = [
        {
            "kind": "gcmc",
            "pressure_bar": pressure_bar,
            "path": str(input_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.in"),
            "log": str(log_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.log"),
            "command": build_lammps_command(
                input_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.in",
                log_file=log_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.log",
            ),
        }
        for pressure_bar in run_plan["conditions"]["pressures_bar"]
    ]

    molecule_templates = {
        component: plan_molecule_template(
            molecule_def_path=Path(run_plan["resources"]["forcefield"]["adsorbates"][component]),
            output_path=molecule_dir / f"{component.lower()}.template",
        )
        for component in run_plan["adsorbates"]["components"]
    }

    return {
        "module": run_plan["module"],
        "status": "prepared_plan",
        "side_effects": "none",
        "working_directory": str(working_dir),
        "inputs": {
            "framework_cif": run_plan["resources"]["cif_path"],
            "forcefield_files": run_plan["resources"]["forcefield"]["files"],
            "adsorbate_definitions": run_plan["resources"]["forcefield"]["adsorbates"],
            "reference_files": run_plan["resources"]["references"],
        },
        "planned_files": {
            "framework_data": plan_cif_to_lammps_data(
                cif_path=Path(run_plan["resources"]["cif_path"]),
                output_path=framework_data,
                atom_style=run_plan["simulation"].get("atom_style", "full"),
            ),
            "forcefield_include": plan_forcefield_to_lammps(
                forcefield_config=run_plan["resources"]["forcefield"],
                components=run_plan["adsorbates"]["components"],
                output_path=forcefield_include,
            ),
            "molecule_templates": molecule_templates,
            "run0_input": {
                "kind": "run0",
                "path": str(run0_input),
                "log": str(log_dir / "run0.log"),
                "command": build_lammps_command(run0_input, log_file=log_dir / "run0.log"),
            },
            "gcmc_test_input": {
                "kind": "gcmc_test",
                "path": str(gcmc_test_input),
                "log": str(log_dir / "gcmc_test.log"),
                "command": build_lammps_command(gcmc_test_input, log_file=log_dir / "gcmc_test.log"),
            },
            "input_scripts": input_scripts,
            "logs": [
                str(log_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.log")
                for pressure_bar in run_plan["conditions"]["pressures_bar"]
            ],
            "summary": str(working_dir / "prepare_summary.json"),
        },
        "parameters": {
            "material_id": run_plan["material"]["material_id"],
            "charge_scheme": run_plan["material"]["charge_scheme"],
            "components": run_plan["adsorbates"]["components"],
            "temperature_K": run_plan["conditions"]["temperature_K"],
            "pressures_bar": run_plan["conditions"]["pressures_bar"],
            "unit_cells": run_plan["simulation"]["unit_cells"],
            "forcefield": run_plan["resources"]["forcefield"]["framework"],
        },
    }


def prepare(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the central prepare function."""
    return prepare_benchmark(run_plan)

def materialize_benchmark(prepare_plan: dict[str, Any]) -> dict[str, Any]:
    """Materialize a prepare plan into generated benchmark files on disk."""
    working_dir = Path(prepare_plan["working_directory"])
    working_dir.mkdir(parents=True, exist_ok=True)

    for directory in ["data", "forcefield", "inputs", "logs", "molecules", "source", "source/forcefield", "source/molecules", "source/references"]:
        (working_dir / directory).mkdir(parents=True, exist_ok=True)

    source_files = _copy_prepare_sources(prepare_plan, working_dir / "source")
    forcefield_config = prepare_plan["planned_files"]["forcefield_include"]["forcefield_config"]
    parameters = load_forcefield_parameters(forcefield_config)
    framework_structure = load_framework_structure(prepare_plan["inputs"]["framework_cif"])
    framework_symbols = _unique(framework_structure.symbols)

    molecule_definitions = prepare_plan["inputs"]["adsorbate_definitions"]
    molecules = {
        component: parse_crafted_molecule_def(molecule_def_path)
        for component, molecule_def_path in molecule_definitions.items()
    }
    adsorbate_atom_types = [
        atom.atom_type
        for molecule in molecules.values()
        for atom in molecule.atoms
    ]
    forcefield = build_lammps_forcefield(
        framework_symbols=framework_symbols,
        adsorbate_atom_types=adsorbate_atom_types,
        parameters=parameters,
    )
    framework_atom_type_ids = [
        atom_type.type_id
        for atom_type in forcefield.atom_types
        if atom_type.source == "framework"
    ]
    adsorbate_atom_type_ids = [
        atom_type.type_id
        for atom_type in forcefield.atom_types
        if atom_type.source == "adsorbate"
    ]
    extra_atom_masses = {
        atom_type.label: atom_type.mass
        for atom_type in forcefield.atom_types
        if atom_type.source == "adsorbate" and atom_type.mass is not None
    }
    atom_type_ids = {atom_type.label: atom_type.type_id for atom_type in forcefield.atom_types}
    atom_charges = {
        atom_type.label: atom_type.charge
        for atom_type in forcefield.atom_types
        if atom_type.source == "adsorbate" and atom_type.charge is not None
    }
    extra_bond_types = _count_unique_bond_types(molecules.values())
    extra_special_per_atom = _max_special_neighbors(molecules.values())

    framework_data_path = Path(prepare_plan["planned_files"]["framework_data"]["output_data"])
    convert_cif_to_lammps_data(
        prepare_plan["inputs"]["framework_cif"],
        framework_data_path,
        atom_style=prepare_plan["planned_files"]["framework_data"].get("atom_style", "full"),
        extra_atom_types=extra_atom_masses,
        extra_bond_types=extra_bond_types,
    )

    molecule_template_paths = {}
    for component, molecule_plan in prepare_plan["planned_files"]["molecule_templates"].items():
        template_path = Path(molecule_plan["output_template"])
        build_molecule_template(
            molecule_plan["input_definition"],
            template_path,
            atom_type_ids=atom_type_ids,
            atom_charges=atom_charges,
        )
        molecule_template_paths[component] = str(template_path)

    forcefield_path = Path(prepare_plan["planned_files"]["forcefield_include"]["output_file"])
    write_lammps_forcefield_include(forcefield, forcefield_path)

    run0_path = Path(prepare_plan["planned_files"]["run0_input"]["path"])
    run0_path.parent.mkdir(parents=True, exist_ok=True)
    run0_path.write_text(
        _render_run0_input(
            framework_data_path=framework_data_path,
            molecule_templates=molecule_template_paths,
            forcefield_path=forcefield_path,
            extra_special_per_atom=extra_special_per_atom,
        ),
        encoding="utf-8",
    )

    gcmc_test_path = Path(prepare_plan["planned_files"]["gcmc_test_input"]["path"])
    gcmc_test_path.parent.mkdir(parents=True, exist_ok=True)
    gcmc_test_path.write_text(
        gcmc_input_builder(
            {
                "framework_data": str(framework_data_path),
                "molecule_templates": molecule_template_paths,
                "forcefield_include": str(forcefield_path),
                "framework_atom_types": framework_atom_type_ids,
                "adsorbate_atom_types": adsorbate_atom_type_ids,
                "component": next(iter(molecule_template_paths)),
                "temperature_K": prepare_plan["parameters"]["temperature_K"],
                "pressure_bar": prepare_plan["parameters"]["pressures_bar"][0],
                "chemical_potential_kcal_mol": -10.0,
                "displacement_A": 1.0,
                "run_steps": 10000,
                "gcmc_every_steps": 1,
                "exchange_attempts": 10,
                "move_attempts": 10,
                "seed": 12345,
                "extra_bond_per_atom": extra_special_per_atom,
                "extra_special_per_atom": extra_special_per_atom,
            }
        ),
        encoding="utf-8",
    )

    result = {
        "status": "materialized",
        "working_directory": str(working_dir),
        "files": {
            "framework_data": str(framework_data_path),
            "forcefield_include": str(forcefield_path),
            "molecule_templates": molecule_template_paths,
            "run0_input": str(run0_path),
            "gcmc_test_input": str(gcmc_test_path),
            "summary": prepare_plan["planned_files"]["summary"],
            "source_files": source_files,
        },
    }
    save_benchmark_data(prepare_plan["planned_files"]["summary"], {"prepare_plan": prepare_plan, "result": result})
    return result


def _copy_prepare_sources(prepare_plan: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    copied["framework_cif"] = str(_copy_file(Path(prepare_plan["inputs"]["framework_cif"]), source_dir))
    copied["forcefield_files"] = {
        name: str(_copy_file(Path(path), source_dir / "forcefield"))
        for name, path in prepare_plan["inputs"]["forcefield_files"].items()
    }
    copied["adsorbate_definitions"] = {
        component: str(_copy_file(Path(path), source_dir / "molecules"))
        for component, path in prepare_plan["inputs"]["adsorbate_definitions"].items()
    }
    copied["reference_files"] = [
        {
            **reference,
            "copied_path": str(_copy_file(Path(reference["path"]), source_dir / "references")),
        }
        for reference in prepare_plan["inputs"]["reference_files"]
    ]
    return copied


def _copy_file(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    destination.write_bytes(source.read_bytes())
    return destination


def _render_run0_input(
    framework_data_path: Path,
    molecule_templates: dict[str, str],
    forcefield_path: Path,
    extra_special_per_atom: int,
) -> str:
    read_data = f"read_data {framework_data_path}"
    if extra_special_per_atom:
        read_data = f"{read_data} extra/special/per/atom {extra_special_per_atom}"

    molecule_lines = [
        f"molecule {component.lower()} {template_path}"
        for component, template_path in molecule_templates.items()
    ]
    lines = [
        "units real",
        "atom_style full",
        "boundary p p p",
        read_data,
        *molecule_lines,
        f"include {forcefield_path}",
        "run 0",
    ]
    return "\n".join(lines) + "\n"

def gcmc_input_builder(inputs: dict[str, Any]) -> str:
    """Build a short LAMMPS GCMC input script for a technical insertion smoke test."""
    component = str(inputs.get("component") or next(iter(inputs["molecule_templates"]))).upper()
    molecule_id = component.lower()
    temperature_K = float(inputs["temperature_K"])
    chemical_potential = float(inputs.get("chemical_potential_kcal_mol", -10.0))
    displacement_A = float(inputs.get("displacement_A", 1.0))
    run_steps = int(inputs.get("run_steps", 10000))
    gcmc_every_steps = int(inputs.get("gcmc_every_steps", 10))
    exchange_attempts = int(inputs.get("exchange_attempts", 10))
    move_attempts = int(inputs.get("move_attempts", 10))
    seed = int(inputs.get("seed", 12345))
    extra_bond_per_atom = int(inputs.get("extra_bond_per_atom", 0))
    extra_special_per_atom = int(inputs.get("extra_special_per_atom", 0))

    read_data_options = []
    if extra_bond_per_atom:
        read_data_options.extend(["extra/bond/per/atom", str(extra_bond_per_atom)])
    if extra_special_per_atom:
        read_data_options.extend(["extra/special/per/atom", str(extra_special_per_atom)])
    read_data = f"read_data {inputs['framework_data']}"
    if read_data_options:
        read_data = f"{read_data} {' '.join(read_data_options)}"

    molecule_templates = inputs["molecule_templates"]
    if component not in molecule_templates:
        raise ValueError(f"Molecule template for component {component!r} is missing.")


    lines = [
        "units real",
        "atom_style full",
        "boundary p p p",
        read_data,
        *[
            f"molecule {component.lower()} {template_path}"
            for component, template_path in inputs["molecule_templates"].items()
        ],
        f"include {inputs['forcefield_include']}",
        f"group framework type {_format_type_ids(inputs['framework_atom_types'])}",
        f"group adsorbate type {_format_type_ids(inputs['adsorbate_atom_types'])}",
        f"# pressure_bar metadata: {inputs.get('pressure_bar', 'not_set')}",
        "thermo 100",
        "thermo_style custom step atoms temp pe etotal press",
        (
            f"fix gcmc_{molecule_id} adsorbate gcmc "
            f"{gcmc_every_steps} {exchange_attempts} {move_attempts} 0 {seed} "
            f"{temperature_K:g} {chemical_potential:g} {displacement_A:g} "
            f"mol {molecule_id} group adsorbate full_energy"
        ),
        f"run {run_steps}",
    ]
    return "\n".join(lines) + "\n"

def run_benchmark(materialized_plan: dict[str, Any]) -> dict[str, Any]:
    gcmc_input = materialized_plan["files"]["gcmc_test_input"]
    log_file = Path(materialized_plan["working_directory"]) / "logs" / "gcmc_test.log"
    summary_file = Path(materialized_plan["working_directory"]) / "gcmc_test_summary.json"

    command = build_lammps_command(gcmc_input, log_file=log_file)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)

    summary = log_to_json(
        log_file,
        summary_file,
        framework_atoms=106,
        adsorbate_atoms_per_molecule=3,
        discard_fraction=0.2,
    )

    return {
        "status": "completed",
        "command": command,
        "log_file": str(log_file),
        "summary_file": str(summary_file),
        "summary": summary,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a MOF benchmark run from benchmark.json.")
    parser.add_argument("config", nargs="?", default="benchmark.json", help="Path to benchmark JSON input.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without running LAMMPS.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the benchmark environment.")
    parser.add_argument("--run-test", action="store_true", help="Run a test GCMC simulation after preparation.")
    args = parser.parse_args(argv)


    config = load_benchmark_data(args.config)
    run_plan = build_run_plan(config)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2))
        return 0
    if args.prepare:
        prepare_plan = prepare_benchmark(run_plan)
        result = materialize_benchmark(prepare_plan)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_test:
        run_plan = build_run_plan(config)
        prepare_plan = prepare_benchmark(run_plan)
        materialized_plan = materialize_benchmark(prepare_plan)
        result = run_benchmark(materialized_plan)
        print(json.dumps(result, indent=2))
        return 0
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


def _pressure_token(pressure_bar: int | float) -> str:
    return str(pressure_bar).replace(".", "p").replace("-", "m")


def _unique(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _count_unique_bond_types(molecules: Any) -> int:
    return len(_unique(bond.bond_type for molecule in molecules for bond in molecule.bonds))


def _max_special_neighbors(molecules: Any) -> int:
    max_neighbors = 0
    for molecule in molecules:
        if molecule.bonds:
            max_neighbors = max(max_neighbors, len(molecule.atoms) - 1)
    return max_neighbors


def _format_type_ids(type_ids: Any) -> str:
    if isinstance(type_ids, str):
        return type_ids
    return " ".join(str(type_id) for type_id in type_ids)


if __name__ == "__main__":
    raise SystemExit(main())
