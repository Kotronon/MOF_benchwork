"""Shared workspace and provenance helpers for Module C workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from pipeline.config import save_benchmark_data


def working_directory(run_plan: dict[str, Any]) -> Path:
    output = run_plan["outputs"]
    directory = Path(output["directory"])
    run_id = output.get("run_id")
    return directory / "runs" / str(run_id) if run_id else directory / "work"


def initialize_workspace(
    directory: Path,
    run_plan: dict[str, Any],
) -> None:
    if directory.exists() and not run_plan["outputs"].get("overwrite", True):
        raise FileExistsError(
            f"Working directory already exists: {directory}. "
            "Use a new output.run_id or enable output.overwrite."
        )
    for child in (
        directory,
        directory / "source" / "framework",
        directory / "source" / "forcefield",
        directory / "source" / "molecules",
        directory / "inputs",
        directory / "configurations",
        directory / "backends",
        directory / "engine",
        directory / "results",
        directory / "reports",
    ):
        child.mkdir(parents=True, exist_ok=True)
    save_benchmark_data(directory / "run_plan.json", run_plan)


def configure_runtime_cache(cache_directory: Path) -> None:
    """Keep plotting and model caches inside the ignored output tree."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory / "matplotlib"))
    os.environ.setdefault("MLIP_MC_CACHE", str(cache_directory / "mlip_mc"))


def snapshot_sources(
    run_plan: dict[str, Any],
    source_directory: Path,
) -> dict[str, Any]:
    forcefield = run_plan["resources"]["forcefield"]
    framework = copy_file(
        Path(run_plan["resources"]["cif_path"]),
        source_directory / "framework",
    )
    forcefield_files = {
        name: str(copy_file(Path(path), source_directory / "forcefield"))
        for name, path in forcefield["files"].items()
    }
    molecule_files = {
        component: str(copy_file(Path(path), source_directory / "molecules"))
        for component, path in forcefield["adsorbates"].items()
    }
    return {
        "framework_cif": str(framework),
        "forcefield_files": forcefield_files,
        "molecule_definitions": molecule_files,
    }


def copy_file(source: Path, destination_directory: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {source}")
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    shutil.copy2(source, destination)
    return destination
