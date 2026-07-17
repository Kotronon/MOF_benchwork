from __future__ import annotations

from pathlib import Path
from typing import Any


def render_run0_input(
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
    chemical_potential = float(inputs.get("chemical_potential_kcal_mol", 0.0))
    displacement_A = float(inputs.get("displacement_A", 1.0))
    run_steps = int(inputs.get("run_steps", 10000))
    gcmc_every_steps = int(inputs.get("gcmc_every_steps", 10))
    exchange_attempts = int(inputs.get("exchange_attempts", 10))
    move_attempts = int(inputs.get("move_attempts", 10))
    seed = int(inputs.get("seed", 12345))
    extra_bond_per_atom = int(inputs.get("extra_bond_per_atom", 0))
    extra_special_per_atom = int(inputs.get("extra_special_per_atom", 0))
    pressure_bar = float(inputs["pressure_bar"])
    pressure_atm = pressure_bar * 0.986923266716
    fugacity_coeff = float(inputs.get("fugacity_coeff", 1.0))
    dump_file = inputs.get("dump_file")
    dump_every_steps = int(inputs.get("dump_every_steps", 1000))

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

    gcmc_fix = (
        f"fix gcmc_{molecule_id} adsorbate gcmc "
        f"{gcmc_every_steps} {exchange_attempts} {move_attempts} 0 {seed} "
        f"{temperature_K:g} {chemical_potential:g} {displacement_A:g} "
        f"mol {molecule_id} group adsorbate full_energy "
        f"pressure {pressure_atm:g} fugacity_coeff {fugacity_coeff:g} "
    )
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
        gcmc_fix,
        "thermo 100",
        "thermo_style custom step atoms temp pe etotal press f_gcmc_co2[3] f_gcmc_co2[4] f_gcmc_co2[5] f_gcmc_co2[6]",
        *(
            [
                f"dump traj all custom {dump_every_steps} {dump_file} id mol type q x y z",
                "dump_modify traj sort id",
            ]
            if dump_file
            else []
        ),
        f"run {run_steps}",
    ]
    return "\n".join(lines) + "\n"


def _format_type_ids(type_ids: Any) -> str:
    if isinstance(type_ids, str):
        return type_ids
    return " ".join(str(type_id) for type_id in type_ids)
