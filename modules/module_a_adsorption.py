from __future__ import annotations

from pathlib import Path
from typing import Any

from converter.cif_to_lammps_data import plan_cif_to_lammps_data
from converter.molecule_to_lammps_template import plan_molecule_template
from engines.lammps_runner import build_lammps_command


def prepare(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Build a side-effect-free preparation plan for Module A adsorption."""
    module_id = run_plan.get("module", {}).get("id")
    if module_id != "A":
        raise ValueError(f"Module A prepare expected module id 'A', got {module_id!r}.")

    output_dir = Path(run_plan["outputs"]["directory"])
    working_dir = output_dir / "work"
    input_dir = working_dir / "inputs"
    data_dir = working_dir / "data"
    molecule_dir = working_dir / "molecules"

    framework_data = data_dir / f"{run_plan['material']['material_id']}.data"
    input_scripts = [
        {
            "pressure_bar": pressure_bar,
            "path": str(input_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.in"),
            "command": build_lammps_command(input_dir / f"gcmc_{_pressure_token(pressure_bar)}bar.in"),
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
            "molecule_templates": molecule_templates,
            "input_scripts": input_scripts,
            "logs": [
                str(working_dir / "logs" / f"gcmc_{_pressure_token(pressure_bar)}bar.log")
                for pressure_bar in run_plan["conditions"]["pressures_bar"]
            ],
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


def _pressure_token(pressure_bar: int | float) -> str:
    return str(pressure_bar).replace(".", "p").replace("-", "m")
