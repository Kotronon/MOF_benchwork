from __future__ import annotations

from pathlib import Path
from typing import Any

from converter.cif_to_lammps_data import convert_cif_to_lammps_data, load_framework_structure
from converter.forcefield_to_lammps import (
    build_lammps_forcefield,
    load_forcefield_parameters,
    write_lammps_forcefield_include,
)
from converter.molecule_to_lammps_template import build_molecule_template, parse_crafted_molecule_def
from pipeline.config import save_benchmark_data
from pipeline.eos import fugacity_coeff_coolprop
from pipeline.lammps_inputs import gcmc_input_builder, render_run0_input
from pipeline.utils import unique


def materialize_benchmark(prepare_plan: dict[str, Any]) -> dict[str, Any]:
    """Materialize a prepare plan into generated benchmark files on disk."""
    working_dir = Path(prepare_plan["working_directory"])
    if working_dir.exists() and not prepare_plan.get("overwrite", True):
        raise FileExistsError(
            f"Working directory already exists: {working_dir}. "
            "Use a new output.run_id or enable output.overwrite."
        )
    working_dir.mkdir(parents=True, exist_ok=True)

    for directory in [
        "data",
        "forcefield",
        "inputs",
        "logs",
        "molecules",
        "source",
        "source/forcefield",
        "source/molecules",
        "source/references",
        "dumps",
    ]:
        (working_dir / directory).mkdir(parents=True, exist_ok=True)

    source_files = _copy_prepare_sources(prepare_plan, working_dir / "source")
    forcefield_config = prepare_plan["planned_files"]["forcefield_include"]["forcefield_config"]
    parameters = load_forcefield_parameters(forcefield_config)
    framework_structure = load_framework_structure(prepare_plan["inputs"]["framework_cif"])
    framework_symbols = unique(framework_structure.symbols)

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
        render_run0_input(
            framework_data_path=framework_data_path,
            molecule_templates=molecule_template_paths,
            forcefield_path=forcefield_path,
            extra_special_per_atom=extra_special_per_atom,
        ),
        encoding="utf-8",
    )

    component = next(iter(molecule_template_paths))
    gcmc_test_path = Path(prepare_plan["planned_files"]["gcmc_test_input"]["path"])
    gcmc_test_path.parent.mkdir(parents=True, exist_ok=True)
    gcmc_test_path.write_text(
        gcmc_input_builder(
            _gcmc_builder_inputs(
                prepare_plan=prepare_plan,
                framework_data_path=framework_data_path,
                molecule_template_paths=molecule_template_paths,
                forcefield_path=forcefield_path,
                framework_atom_type_ids=framework_atom_type_ids,
                adsorbate_atom_type_ids=adsorbate_atom_type_ids,
                component=component,
                pressure_bar=prepare_plan["parameters"]["pressures_bar"][0],
                extra_special_per_atom=extra_special_per_atom,
            )
        ),
        encoding="utf-8",
    )

    for input_script in prepare_plan["planned_files"]["input_scripts"]:
        pressure_bar = input_script["pressure_bar"]
        fugacity_coeff = fugacity_coeff_coolprop(
            component=component,
            temperature_K=prepare_plan["parameters"]["temperature_K"],
            pressure_bar=pressure_bar,
            backend="PR",
        )
        gcmc_input_path = Path(input_script["path"])
        gcmc_input_path.parent.mkdir(parents=True, exist_ok=True)
        gcmc_input_path.write_text(
            gcmc_input_builder(
                _gcmc_builder_inputs(
                    prepare_plan=prepare_plan,
                    framework_data_path=framework_data_path,
                    molecule_template_paths=molecule_template_paths,
                    forcefield_path=forcefield_path,
                    framework_atom_type_ids=framework_atom_type_ids,
                    adsorbate_atom_type_ids=adsorbate_atom_type_ids,
                    component=component,
                    pressure_bar=pressure_bar,
                    extra_special_per_atom=extra_special_per_atom,
                    fugacity_coeff=fugacity_coeff,
                    dump_file=input_script["dump"],
                    dump_every_steps=1000,
                )
            ),
            encoding="utf-8",
        )

    result = {
        "status": "materialized",
        "working_directory": str(working_dir),
        "parameters": prepare_plan["parameters"],
        "files": {
            "framework_data": str(framework_data_path),
            "forcefield_include": str(forcefield_path),
            "molecule_templates": molecule_template_paths,
            "run0_input": str(run0_path),
            "gcmc_test_input": str(gcmc_test_path),
            "summary": prepare_plan["planned_files"]["summary"],
            "source_files": source_files,
            "gcmc_inputs": [script["path"] for script in prepare_plan["planned_files"]["input_scripts"]],
            "gcmc_runs": prepare_plan["planned_files"]["input_scripts"],
            "dump_files": [script["dump"] for script in prepare_plan["planned_files"]["input_scripts"]],
        },
    }
    save_benchmark_data(prepare_plan["planned_files"]["summary"], {"prepare_plan": prepare_plan, "result": result})
    return result


def _gcmc_builder_inputs(
    *,
    prepare_plan: dict[str, Any],
    framework_data_path: Path,
    molecule_template_paths: dict[str, str],
    forcefield_path: Path,
    framework_atom_type_ids: list[int],
    adsorbate_atom_type_ids: list[int],
    component: str,
    pressure_bar: float,
    extra_special_per_atom: int,
    fugacity_coeff: float = 1.0,
    dump_file: str | None = None,
    dump_every_steps: int = 1000,
) -> dict[str, Any]:
    return {
        "framework_data": str(framework_data_path),
        "molecule_templates": molecule_template_paths,
        "forcefield_include": str(forcefield_path),
        "framework_atom_types": framework_atom_type_ids,
        "adsorbate_atom_types": adsorbate_atom_type_ids,
        "component": component,
        "temperature_K": prepare_plan["parameters"]["temperature_K"],
        "pressure_bar": pressure_bar,
        "chemical_potential_kcal_mol": 0.0,
        "displacement_A": 1.0,
        "run_steps": prepare_plan["parameters"].get("run_steps", 10000),
        "gcmc_every_steps": 1,
        "exchange_attempts": 10,
        "move_attempts": 10,
        "seed": 12345,
        "extra_bond_per_atom": extra_special_per_atom,
        "extra_special_per_atom": extra_special_per_atom,
        "fugacity_coeff": fugacity_coeff,
        "dump_file": dump_file,
        "dump_every_steps": dump_every_steps,
    }


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


def _count_unique_bond_types(molecules: Any) -> int:
    return len(unique(bond.bond_type for molecule in molecules for bond in molecule.bonds))


def _max_special_neighbors(molecules: Any) -> int:
    max_neighbors = 0
    for molecule in molecules:
        if molecule.bonds:
            max_neighbors = max(max_neighbors, len(molecule.atoms) - 1)
    return max_neighbors
