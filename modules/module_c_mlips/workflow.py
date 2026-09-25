"""Configuration-driven Module C potential-benchmark workflow."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from analysis.potential_report import create_potential_report
from modules.module_c_mlips.datasets import (
    build_smoke_configurations,
    build_widom_configurations,
)
from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import PotentialBackend
from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)
from modules.module_c_mlips.potential_backends.mace import MaceBackend
from modules.module_c_mlips.runner import run_potential_comparison
from pipeline.config import save_benchmark_data


DEFAULT_BACKENDS = [
    {
        "type": "classical_lammps",
        "name": "uff_ddec_lammps",
        "lammps_command": "lmp",
    },
    {
        "type": "mace_mp",
        "name": "mace_mp_small",
        "model": "small",
        "device": "cpu",
        "default_dtype": "float32",
        "dispersion": False,
    },
]


def run_potential_benchmark(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Build configurations and backends and run a Module C comparison."""
    if run_plan.get("module", {}).get("id") != "C":
        raise ValueError("Potential benchmarks require Module C.")

    components = run_plan["adsorbates"]["components"]
    if len(components) != 1:
        raise ValueError("Module C V1 supports exactly one adsorbate.")

    settings = run_plan["benchmark"].get("potential_benchmark", {})
    if not isinstance(settings, dict):
        raise TypeError("'benchmark.potential_benchmark' must be an object.")
    working_directory = _working_directory(run_plan)
    _initialize_workspace(working_directory, run_plan)
    _configure_runtime_cache(
        Path(run_plan["outputs"]["directory"]) / ".cache" / "plots"
    )
    source_files = _snapshot_sources(run_plan, working_directory / "source")

    component = components[0]
    forcefield = run_plan["resources"]["forcefield"]
    configurations = _build_configurations(
        settings,
        run_plan,
        component,
        forcefield,
    )
    excluded_regions = set(
        settings.get("exclude_regions", ["technical_repulsive_probe"])
    )
    configurations = [
        configuration
        for configuration in configurations
        if configuration.region not in excluded_regions
    ]
    if not configurations:
        raise ValueError("No potential-benchmark configurations remain after filtering.")

    configuration_manifest = _write_configurations(
        configurations,
        working_directory / "configurations",
    )
    backends = _build_backends(
        settings.get("backends", DEFAULT_BACKENDS),
        run_plan,
        working_directory / "backends",
    )
    baseline_backend = str(
        settings.get("baseline_backend", backends[0].name)
    )
    output_path = working_directory / "results" / "potential_comparison.json"

    report = run_potential_comparison(
        configurations,
        backends,
        baseline_backend=baseline_backend,
        output_path=output_path,
    )
    report.update(
        {
            "working_directory": str(working_directory),
            "source_files": source_files,
            "configuration_manifest": configuration_manifest,
        }
    )
    report["analysis"] = create_potential_report(
        report,
        working_directory / "reports",
        save_csv=bool(run_plan["outputs"].get("save_csv", True)),
        save_plots=bool(run_plan["outputs"].get("save_plots", True)),
    )
    save_benchmark_data(output_path, report)
    return report


def _build_configurations(
    settings: dict[str, Any],
    run_plan: dict[str, Any],
    component: str,
    forcefield: dict[str, Any],
) -> list[InteractionConfiguration]:
    configuration_set = str(
        settings.get("configuration_set", "smoke")
    ).strip().casefold()
    common_arguments = {
        "material": run_plan["material"]["material_id"],
        "adsorbate": component,
        "cell_representation": run_plan["simulation"]["cell_representation"],
        "unit_cells": run_plan["simulation"]["unit_cells"],
        "cutoff_A": float(run_plan["simulation"]["cutoff_A"]),
        "minimum_image_policy": run_plan["simulation"]["minimum_image_policy"],
    }
    paths = (
        run_plan["resources"]["cif_path"],
        forcefield["adsorbates"][component],
        forcefield["files"]["pseudo_atoms"],
    )
    if configuration_set == "smoke":
        return build_smoke_configurations(*paths, **common_arguments)
    if configuration_set == "widom":
        dataset = settings.get("dataset", {})
        if not isinstance(dataset, dict):
            raise TypeError("'benchmark.potential_benchmark.dataset' must be an object.")
        seeds = run_plan["simulation"].get("seeds", [12345])
        return build_widom_configurations(
            *paths,
            sample_count=dataset.get("sample_count", 100),
            seed=dataset.get("seed", seeds[0]),
            minimum_distance_A=dataset.get("minimum_distance_A", 0.0),
            maximum_attempts_per_sample=dataset.get(
                "maximum_attempts_per_sample",
                10_000,
            ),
            random_orientations=dataset.get("random_orientations", True),
            **common_arguments,
        )
    raise ValueError(
        "Unsupported Module C configuration_set "
        f"{configuration_set!r}; expected 'smoke' or 'widom'."
    )


def _build_backends(
    backend_specs: Any,
    run_plan: dict[str, Any],
    backend_directory: Path,
) -> list[PotentialBackend]:
    if not isinstance(backend_specs, list) or len(backend_specs) < 2:
        raise ValueError("At least two potential backend definitions are required.")

    forcefield = run_plan["resources"]["forcefield"]
    backends: list[PotentialBackend] = []
    for specification in backend_specs:
        if not isinstance(specification, dict):
            raise TypeError("Each potential backend definition must be an object.")
        backend_type = str(specification.get("type", "")).strip().casefold()
        backend_name = str(specification.get("name", backend_type)).strip()
        if not backend_name:
            raise ValueError("Each potential backend needs a non-empty name.")

        if backend_type == "classical_lammps":
            backends.append(
                ClassicalLAMMPSBackend(
                    lammps_command=str(
                        specification.get("lammps_command", "lmp")
                    ),
                    pseudo_atoms_file=Path(forcefield["files"]["pseudo_atoms"]),
                    mixing_rules_file=Path(forcefield["files"]["mixing_rules"]),
                    backend_name=backend_name,
                    working_directory=backend_directory / backend_name,
                    pair_style=str(
                        specification.get(
                            "pair_style",
                            "lj/cut/coul/long "
                            f"{float(run_plan['simulation']['cutoff_A']):g}",
                        )
                    ),
                    kspace_style=str(
                        specification.get(
                            "kspace_style",
                            f"{run_plan['simulation']['kspace_style']} "
                            f"{float(run_plan['simulation']['kspace_accuracy']):g}",
                        )
                    ),
                )
            )
        elif backend_type == "mace_mp":
            backends.append(
                MaceBackend(
                    backend_name=backend_name,
                    model=str(specification.get("model", "small")),
                    device=str(specification.get("device", "cpu")),
                    default_dtype=str(
                        specification.get("default_dtype", "float32")
                    ),
                    dispersion=bool(specification.get("dispersion", False)),
                )
            )
        else:
            raise ValueError(f"Unsupported potential backend type {backend_type!r}.")

    return backends


def _working_directory(run_plan: dict[str, Any]) -> Path:
    output = run_plan["outputs"]
    directory = Path(output["directory"])
    run_id = output.get("run_id")
    return directory / "runs" / str(run_id) if run_id else directory / "work"


def _initialize_workspace(
    working_directory: Path,
    run_plan: dict[str, Any],
) -> None:
    if working_directory.exists() and not run_plan["outputs"].get("overwrite", True):
        raise FileExistsError(
            f"Working directory already exists: {working_directory}. "
            "Use a new output.run_id or enable output.overwrite."
        )
    for directory in (
        working_directory,
        working_directory / "source" / "framework",
        working_directory / "source" / "forcefield",
        working_directory / "source" / "molecules",
        working_directory / "configurations",
        working_directory / "backends",
        working_directory / "results",
        working_directory / "reports",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    save_benchmark_data(working_directory / "run_plan.json", run_plan)


def _configure_runtime_cache(cache_directory: Path) -> None:
    """Keep plotting/font caches writable and local to the ignored run tree."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory / "matplotlib"))


def _snapshot_sources(
    run_plan: dict[str, Any],
    source_directory: Path,
) -> dict[str, Any]:
    forcefield = run_plan["resources"]["forcefield"]
    framework = _copy_file(
        Path(run_plan["resources"]["cif_path"]),
        source_directory / "framework",
    )
    forcefield_files = {
        name: str(_copy_file(Path(path), source_directory / "forcefield"))
        for name, path in forcefield["files"].items()
    }
    molecule_files = {
        component: str(_copy_file(Path(path), source_directory / "molecules"))
        for component, path in forcefield["adsorbates"].items()
    }
    return {
        "framework_cif": str(framework),
        "forcefield_files": forcefield_files,
        "molecule_definitions": molecule_files,
    }


def _copy_file(source: Path, destination_directory: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {source}")
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    shutil.copy2(source, destination)
    return destination


def _write_configurations(
    configurations: list[InteractionConfiguration],
    output_directory: Path,
) -> list[dict[str, Any]]:
    from ase.io import write

    manifest = []
    for configuration in configurations:
        filename = _safe_name(configuration.configuration_id) + ".extxyz"
        output_path = output_directory / filename
        atoms = configuration.atoms.copy()
        atoms.info.update(
            {
                "configuration_id": configuration.configuration_id,
                "material": configuration.material,
                "adsorbate": configuration.adsorbate,
                "region": configuration.region or "",
                "source": configuration.source,
            }
        )
        write(output_path, atoms, format="extxyz")
        manifest.append(
            {
                "configuration_id": configuration.configuration_id,
                "path": str(output_path),
                "region": configuration.region,
                "atom_count": len(configuration.atoms),
                "framework_indices": configuration.framework_indices,
                "adsorbate_indices": configuration.adsorbate_indices,
                "bonds": [
                    {
                        "atom1_index": bond.atom1_index,
                        "atom2_index": bond.atom2_index,
                        "forcefield_type": bond.forcefield_type,
                    }
                    for bond in configuration.bonds
                ],
                "metadata": configuration.metadata,
            }
        )
    save_benchmark_data(
        output_directory / "configuration_manifest.json",
        {"configurations": manifest},
    )
    return manifest


def _safe_name(value: str) -> str:
    name = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ).strip("._")
    return name or "configuration"
