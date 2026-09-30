"""Configuration-driven Module C potential-benchmark workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.potential_report import create_potential_report
from modules.module_c_mlips.datasets import (
    build_smoke_configurations,
    build_widom_configurations,
    ensure_golddac_dataset,
    load_golddac_configurations,
)
from modules.module_c_mlips.model_assets import ensure_model_asset
from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.potential_backends.base import PotentialBackend
from modules.module_c_mlips.potential_backends.classical_lammps import (
    ClassicalLAMMPSBackend,
)
from modules.module_c_mlips.potential_backends.mace import MaceBackend
from modules.module_c_mlips.potential_backends.nequip import NequipBackend
from modules.module_c_mlips.runner import run_potential_comparison
from modules.module_c_mlips.workspace import (
    configure_runtime_cache as _configure_runtime_cache,
    copy_file as _copy_file,
    initialize_workspace as _initialize_workspace,
    snapshot_sources as _snapshot_sources,
    working_directory as _working_directory,
)
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
    dataset_provenance = _snapshot_dataset_provenance(
        settings,
        working_directory / "source" / "dataset",
    )
    if dataset_provenance is not None:
        source_files["dataset_manifest"] = dataset_provenance["snapshot_path"]
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
    reference_baseline = baseline_backend not in {
        backend.name for backend in backends
    }
    output_path = working_directory / "results" / "potential_comparison.json"

    report = run_potential_comparison(
        configurations,
        backends,
        baseline_backend=baseline_backend,
        reference_baseline=reference_baseline,
        output_path=output_path,
    )
    report.update(
        {
            "working_directory": str(working_directory),
            "source_files": source_files,
            "configuration_manifest": configuration_manifest,
            "dataset_provenance": dataset_provenance,
            "benchmark_protocol": (
                "golddac_dft_reference"
                if reference_baseline
                else "backend_parity"
            ),
        }
    )
    report["analysis"] = create_potential_report(
        report,
        working_directory / "reports",
        save_csv=bool(run_plan["outputs"].get("save_csv", True)),
        save_plots=bool(run_plan["outputs"].get("save_plots", True)),
        acceptance_thresholds=settings.get("acceptance_thresholds"),
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
    if configuration_set == "golddac":
        dataset = settings.get("dataset", {})
        if not isinstance(dataset, dict):
            raise TypeError("'benchmark.potential_benchmark.dataset' must be an object.")
        root = dataset.get("path") or dataset.get(
            "root",
            "external/datasets/golddac",
        )
        dataset_path = Path(root).expanduser()
        if dataset_path.suffix.casefold() != ".xyz":
            ensure_golddac_dataset(dataset_path, download_missing=False)
        return load_golddac_configurations(
            dataset_path,
            split=str(dataset.get("split", "test")),
            adsorbates=dataset.get("adsorbates", [component]),
            materials=dataset.get("materials"),
            regions=dataset.get("regions"),
            max_configurations=dataset.get("max_configurations"),
            seed=int(dataset.get("seed", 12345)),
        )
    raise ValueError(
        "Unsupported Module C configuration_set "
        f"{configuration_set!r}; expected 'smoke', 'widom', or 'golddac'."
    )


def _build_backends(
    backend_specs: Any,
    run_plan: dict[str, Any],
    backend_directory: Path,
) -> list[PotentialBackend]:
    if not isinstance(backend_specs, list) or not backend_specs:
        raise ValueError("At least one potential backend definition is required.")

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
            if isinstance(specification.get("asset"), dict):
                ensure_model_asset(specification, download_missing=False)
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
        elif backend_type == "nequip":
            model = specification.get("model")
            if not model:
                raise ValueError("NequIP backend requires a model path.")
            backends.append(
                NequipBackend(
                    backend_name=backend_name,
                    model=str(model),
                    device=str(specification.get("device", "cpu")),
                    loader=str(specification.get("loader", "auto")),
                    species_to_type_name=specification.get(
                        "species_to_type_name"
                    ),
                    energy_units_to_eV=float(
                        specification.get("energy_units_to_eV", 1.0)
                    ),
                    length_units_to_A=float(
                        specification.get("length_units_to_A", 1.0)
                    ),
                    energy_mode=str(
                        specification.get("energy_mode", "total_energy")
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported potential backend type {backend_type!r}.")

    return backends


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


def _snapshot_dataset_provenance(
    settings: dict[str, Any],
    destination: Path,
) -> dict[str, Any] | None:
    if str(settings.get("configuration_set", "")).casefold() != "golddac":
        return None
    dataset = settings.get("dataset", {})
    if not isinstance(dataset, dict):
        return None
    root = Path(
        dataset.get("path")
        or dataset.get("root", "external/datasets/golddac")
    ).expanduser()
    if root.suffix.casefold() == ".xyz":
        return {
            "dataset": "GoldDAC",
            "version": "v3",
            "doi": "10.6084/m9.figshare.27978474.v3",
            "test_path": str(root),
            "snapshot_path": None,
        }
    manifest_path = root / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    snapshot = _copy_file(manifest_path, destination)
    return {**provenance, "snapshot_path": str(snapshot)}
