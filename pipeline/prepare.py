from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from converter.cif_to_lammps_data import plan_cif_to_lammps_data
from converter.forcefield_to_lammps import plan_forcefield_to_lammps
from converter.molecule_to_lammps_template import plan_molecule_template
from engines.lammps_runner import build_lammps_command
from pipeline.utils import pressure_token


def prepare_benchmark(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Build a central side-effect-free preparation plan."""
    module_id = run_plan.get("module", {}).get("id")
    if module_id != "A":
        raise ValueError(f"Prepare currently supports module 'A' only, got {module_id!r}.")

    production_steps = int(run_plan["simulation"].get("cycles", 10000))
    equilibration_steps = int(run_plan["simulation"].get("initialization_cycles", 0))
    if production_steps <= 0:
        raise ValueError("'simulation.cycles' must be positive.")
    if equilibration_steps < 0:
        raise ValueError("'simulation.initialization_cycles' must be non-negative.")

    output_dir = Path(run_plan["outputs"]["directory"])
    working_dir = _working_directory(output_dir, run_plan["outputs"].get("run_id"))
    input_dir = working_dir / "inputs"
    data_dir = working_dir / "data"
    molecule_dir = working_dir / "molecules"
    forcefield_dir = working_dir / "forcefield"
    log_dir = working_dir / "logs"
    dump_dir = working_dir / "dumps"

    framework_data = data_dir / f"{run_plan['material']['material_id']}.data"
    forcefield_include = forcefield_dir / "forcefield.in"
    run0_input = input_dir / "in.run0"
    gcmc_test_input = input_dir / "in.gcmc_test"
    seeds = list(run_plan["simulation"].get("seeds", [12345]))
    multiple_replicates = len(seeds) > 1
    save_dumps = bool(run_plan["outputs"].get("save_dumps", True))
    dump_every_steps = int(run_plan["outputs"].get("dump_every_steps", 1000))
    input_scripts = []
    for pressure_bar in run_plan["conditions"]["pressures_bar"]:
        pressure_name = f"gcmc_{pressure_token(pressure_bar)}bar"
        for replicate_index, seed in enumerate(seeds, start=1):
            run_name = f"{pressure_name}_seed{seed}" if multiple_replicates else pressure_name
            input_path = input_dir / f"{run_name}.in"
            log_path = log_dir / f"{run_name}.log"
            input_scripts.append(
                {
                    "kind": "gcmc",
                    "pressure_bar": pressure_bar,
                    "replicate_index": replicate_index,
                    "seed": seed,
                    "path": str(input_path),
                    "log": str(log_path),
                    "dump": str(dump_dir / f"{run_name}.lammpstrj") if save_dumps else None,
                    "command": build_lammps_command(input_path, log_file=log_path),
                }
            )

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
        "overwrite": run_plan["outputs"].get("overwrite", True),
        "resume": run_plan["outputs"].get("resume", False),
        "inputs": {
            "framework_cif": run_plan["resources"]["cif_path"],
            "forcefield_files": run_plan["resources"]["forcefield"]["files"],
            "adsorbate_definitions": run_plan["resources"]["forcefield"]["adsorbates"],
            "reference_files": run_plan["resources"]["references"],
        },
        "evaluation": run_plan.get("evaluation", {}),
        "applicability": run_plan.get("benchmark", {}).get("applicability", {}),
        "convergence": run_plan.get("convergence", {}),
        "output": {
            "save_dumps": save_dumps,
            "dump_every_steps": dump_every_steps,
        },
        "planned_files": {
            "framework_data": plan_cif_to_lammps_data(
                cif_path=Path(run_plan["resources"]["cif_path"]),
                output_path=framework_data,
                atom_style=run_plan["simulation"].get("atom_style", "full"),
                cell_representation=run_plan["simulation"].get("cell_representation", "source"),
                unit_cells=run_plan["simulation"]["unit_cells"],
                cutoff_A=float(run_plan["simulation"].get("cutoff_A", 12.0)),
                minimum_image_policy=run_plan["simulation"].get("minimum_image_policy", "error"),
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
                script["log"] for script in input_scripts
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
            "cell_representation": run_plan["simulation"].get("cell_representation", "source"),
            "cutoff_A": float(run_plan["simulation"].get("cutoff_A", 12.0)),
            "minimum_image_policy": run_plan["simulation"].get("minimum_image_policy", "error"),
            "seeds": seeds,
            "replicate_count": len(seeds),
            "save_dumps": save_dumps,
            "dump_every_steps": dump_every_steps,
            "forcefield": run_plan["resources"]["forcefield"]["framework"],
            "pair_style": run_plan["simulation"].get("pair_style", "lj/cut/coul/long"),
            "kspace_style": run_plan["simulation"].get("kspace_style", "pppm"),
            "kspace_accuracy": run_plan["simulation"].get("kspace_accuracy", 1e-5),
            "production_steps": production_steps,
            "equilibration_steps": equilibration_steps,
            "run_steps": equilibration_steps + production_steps,
        },
    }


def prepare(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the central prepare function."""
    return prepare_benchmark(run_plan)


def _working_directory(output_dir: Path, run_id: str | None) -> Path:
    if run_id:
        return output_dir / "runs" / run_id
    return output_dir / "work"


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
