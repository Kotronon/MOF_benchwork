"""Configuration-driven MLIP-MC workflow for Module C."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from engines.mlip_mc_runner import run_mlip_mc_gcmc, run_mlip_mc_widom
from modules.module_c_mlips.datasets.common import load_host_guest_system
from modules.module_c_mlips.dependencies import ensure_mlip_mc_dependencies
from modules.module_c_mlips.potential_backends.calculators import (
    build_ase_calculator,
)
from modules.module_c_mlips.workspace import (
    configure_runtime_cache,
    initialize_workspace,
    snapshot_sources,
    working_directory,
)
from pipeline.config import save_benchmark_data


SUPPORTED_WORKFLOWS = {"mlip_mc_widom", "mlip_mc_gcmc"}


def run_mlip_mc_benchmark(
    run_plan: dict[str, Any],
    *,
    install_missing: bool = False,
) -> dict[str, Any]:
    """Execute an MLIP-MC Widom or GCMC benchmark from a run plan."""
    if run_plan.get("module", {}).get("id") != "C":
        raise ValueError("MLIP-MC benchmarks require Module C.")
    components = run_plan["adsorbates"]["components"]
    if len(components) != 1:
        raise ValueError("MLIP-MC integration currently supports one adsorbate.")

    potential_settings = run_plan["benchmark"].get("potential_benchmark", {})
    if not isinstance(potential_settings, dict):
        raise TypeError("'benchmark.potential_benchmark' must be an object.")
    workflow = str(
        potential_settings.get("workflow", "")
    ).strip().casefold().replace("-", "_")
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"Unsupported MLIP-MC workflow {workflow!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_WORKFLOWS))
            + "."
        )
    settings = potential_settings.get("mlip_mc", {})
    if not isinstance(settings, dict):
        raise TypeError("'benchmark.potential_benchmark.mlip_mc' must be an object.")
    model = settings.get("model", {})
    if not isinstance(model, dict):
        raise TypeError("'benchmark.potential_benchmark.mlip_mc.model' must be an object.")
    backend = str(model.get("backend", "mace-torch"))
    dependency_status = ensure_mlip_mc_dependencies(
        backend,
        install_missing=(
            install_missing
            or bool(settings.get("auto_install_dependencies", False))
        ),
    )

    directory = working_directory(run_plan)
    initialize_workspace(directory, run_plan)
    configure_runtime_cache(Path(run_plan["outputs"]["directory"]) / ".cache")
    source_files = snapshot_sources(run_plan, directory / "source")
    component = components[0]
    forcefield = run_plan["resources"]["forcefield"]
    framework, adsorbate, _bonds = load_host_guest_system(
        run_plan["resources"]["cif_path"],
        forcefield["adsorbates"][component],
        forcefield["files"]["pseudo_atoms"],
        cell_representation=run_plan["simulation"]["cell_representation"],
        unit_cells=run_plan["simulation"]["unit_cells"],
        cutoff_A=float(run_plan["simulation"]["cutoff_A"]),
        minimum_image_policy=run_plan["simulation"]["minimum_image_policy"],
    )
    adsorbate, removed_virtual_sites = _physical_adsorbate_atoms(adsorbate)
    input_files = _write_engine_inputs(
        framework,
        adsorbate,
        directory / "inputs",
    )
    model_manifest = _build_model_manifest(model, dependency_status.to_dict())
    model_manifest_path = directory / "source" / "model_manifest.json"
    save_benchmark_data(model_manifest_path, model_manifest)
    calculator = build_ase_calculator(model)

    temperature_K = float(run_plan["conditions"]["temperature_K"])
    seeds = run_plan["simulation"].get("seeds", [12345])
    seed = int(settings.get("seed", seeds[0]))
    engine_directory = directory / "engine" / "mlip_mc"
    if workflow == "mlip_mc_widom":
        trials = int(settings.get("trials", 10_000))
        result = run_mlip_mc_widom(
            calculator,
            framework,
            adsorbate,
            temperature_K=temperature_K,
            trial_count=trials,
            seed=seed,
            device=str(model.get("device", "cpu")),
            output_directory=engine_directory / "widom",
            block_size=int(settings.get("block_size", max(1, trials // 10))),
            convergence_checkpoints=settings.get("convergence_checkpoints"),
            write_analysis_csv=bool(run_plan["outputs"].get("save_csv", True)),
            write_analysis_plot=bool(run_plan["outputs"].get("save_plots", True)),
        )
    else:
        result = run_mlip_mc_gcmc(
            calculator,
            framework,
            adsorbate,
            component=component,
            temperature_K=temperature_K,
            pressures_bar=[
                float(value) for value in run_plan["conditions"]["pressures_bar"]
            ],
            equilibration_steps=int(settings.get("equilibration_steps", 10_000)),
            production_steps=int(settings.get("production_steps", 20_000)),
            seed=seed,
            device=str(model.get("device", "cpu")),
            output_directory=engine_directory / "gcmc",
            checkpoint_interval=int(settings.get("checkpoint_interval", 10_000)),
            write_trajectory=bool(
                settings.get(
                    "write_trajectory",
                    run_plan["outputs"].get("save_dumps", False),
                )
            ),
            trajectory_interval=int(
                settings.get(
                    "trajectory_interval",
                    run_plan["outputs"].get("dump_every_steps", 100),
                )
            ),
            overwrite_checkpoints=bool(
                settings.get("overwrite_checkpoints", False)
            ),
            allow_ideal_gas_fallback=bool(
                settings.get("allow_ideal_gas_fallback", False)
            ),
        )

    result.update(
        {
            "working_directory": str(directory),
            "workflow": workflow,
            "material": run_plan["material"]["material_id"],
            "adsorbate": component,
            "model": model_manifest,
            "dependency_status": dependency_status.to_dict(),
            "source_files": source_files,
            "input_files": input_files,
            "removed_virtual_adsorbate_sites": removed_virtual_sites,
            "model_manifest_path": str(model_manifest_path),
        }
    )
    output_path = directory / "results" / "mlip_mc_benchmark.json"
    result["benchmark_output_path"] = str(output_path)
    save_benchmark_data(output_path, result)
    return result


def _physical_adsorbate_atoms(adsorbate: Any) -> tuple[Any, int]:
    """Remove zero-mass force-field sites that are not chemical MLIP atoms."""
    masses = adsorbate.get_masses()
    physical_indices = [
        index for index, mass in enumerate(masses) if float(mass) >= 0.01
    ]
    if not physical_indices:
        raise ValueError("Adsorbate contains no physical atoms for MLIP evaluation.")
    removed = len(adsorbate) - len(physical_indices)
    return adsorbate[physical_indices], removed


def _write_engine_inputs(
    framework: Any,
    adsorbate: Any,
    output_directory: Path,
) -> dict[str, str]:
    try:
        from ase.io import write
    except ImportError as exc:
        raise ImportError("ASE is required to materialize MLIP-MC inputs.") from exc
    output_directory.mkdir(parents=True, exist_ok=True)
    framework_path = output_directory / "framework.extxyz"
    adsorbate_path = output_directory / "adsorbate.extxyz"
    write(framework_path, framework, format="extxyz")
    write(adsorbate_path, adsorbate, format="extxyz")
    return {
        "framework": str(framework_path),
        "adsorbate": str(adsorbate_path),
    }


def _build_model_manifest(
    model: dict[str, Any],
    dependency_status: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        **model,
        "dependency_status": dependency_status,
    }
    model_value = model.get("model")
    if model_value not in (None, "", "small", "medium", "large"):
        model_path = Path(str(model_value)).expanduser()
        if model_path.is_file():
            manifest["resolved_model_path"] = str(model_path.resolve())
            manifest["sha256"] = _file_sha256(model_path)
    return manifest


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
