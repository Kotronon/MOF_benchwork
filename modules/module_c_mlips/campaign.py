"""Declarative campaign execution for extensible Module C benchmarks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from modules.module_c_mlips.active_learning.workflow import run_active_learning
from modules.module_c_mlips.adsorption_engines import (
    build_adsorption_engine,
    compare_engine_results,
)
from modules.module_c_mlips.datasets.common import load_host_guest_system
from modules.module_c_mlips.dependencies import (
    ensure_flames_dependencies,
    ensure_mlip_mc_dependencies,
)
from modules.module_c_mlips.model_assets import ensure_model_asset
from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator
from pipeline.config import normalize_config, save_benchmark_data
from pipeline.planning import build_run_plan


RUNNABLE_STAGES = {
    "screening",
    "active_learning",
    "widom",
    "gcmc_pilot",
    "gcmc_production",
}


def setup_module_c(
    run_plan: dict[str, Any],
    *,
    install_missing: bool = False,
) -> dict[str, Any]:
    """Inspect or install configured Module C runtime dependencies."""
    settings = run_plan["benchmark"]["potential_benchmark"]
    models = _campaign_models(settings)
    dependency_status = []
    for model in models:
        backend = str(model.get("backend", model.get("type", "mace-torch")))
        if backend.casefold().replace("-", "_") in {
            "lammps",
            "classical",
            "classical_lammps",
        }:
            dependency_status.append(
                {
                    "model": model.get("name", "classical_lammps"),
                    "backend": "classical_lammps",
                    "ready": shutil.which("lmp") is not None,
                }
            )
            continue
        try:
            status = ensure_mlip_mc_dependencies(
                backend,
                install_missing=install_missing,
            )
        except ImportError as exc:
            dependency_status.append(
                {
                    "model": model.get(
                        "name",
                        model.get("model"),
                    ),
                    "backend": backend,
                    "ready": False,
                    "error": str(exc),
                    "asset": None,
                }
            )
            continue
        asset = None
        if isinstance(model.get("asset"), dict):
            asset = ensure_model_asset(model, download_missing=install_missing)
        dependency_status.append(
            {"model": model.get("name", model.get("model")), **status.to_dict(), "asset": asset}
        )
    flames = None
    engines = {
        settings.get("adsorption_engine"),
        settings.get("cross_check_engine"),
    }
    if "flames" in engines:
        try:
            flames = ensure_flames_dependencies(
                install_missing=install_missing
            )
        except ImportError as exc:
            flames = {"available": False, "error": str(exc)}
    cp2k = settings["active_learning"].get("cp2k", {})
    cp2k_executable = str(cp2k.get("executable", "cp2k.psmp"))
    cp2k_required = settings["validation"]["mode"] == "active_learning"
    cp2k_available = shutil.which(cp2k_executable) is not None
    dependencies_ready = all(
        item["ready"] for item in dependency_status
    )
    flames_ready = flames is None or bool(flames.get("available"))
    return {
        "status": (
            "ready"
            if dependencies_ready
            and flames_ready
            and (not cp2k_required or cp2k_available)
            else "missing_dependencies"
        ),
        "models": dependency_status,
        "flames": flames,
        "cp2k": {
            "executable": cp2k_executable,
            "scheduler": cp2k.get("scheduler", "local"),
            "required_for_active_learning": cp2k_required,
            "available": cp2k_available,
        },
    }


def run_module_c_campaign(
    raw_config: dict[str, Any],
    *,
    stage: str,
    selected_systems: Sequence[str] = (),
    selected_models: Sequence[str] = (),
    jobs: int = 1,
    submit: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Expand and execute one stage of a Module C campaign."""
    normalized_stage = stage.strip().casefold().replace("-", "_")
    if normalized_stage not in RUNNABLE_STAGES:
        raise ValueError(f"Unsupported Module C stage {stage!r}.")
    base = normalize_config(raw_config)
    settings = base["benchmark"]["potential_benchmark"]
    tasks = build_campaign_tasks(
        base,
        stage=normalized_stage,
        selected_systems=selected_systems,
        selected_models=selected_models,
    )
    if not tasks:
        raise ValueError("No Module C campaign tasks match the selected filters.")

    if jobs < 1:
        raise ValueError("jobs must be positive.")
    if jobs == 1 or normalized_stage == "active_learning":
        results = [
            _run_campaign_task(task, submit=submit, resume=resume)
            for task in tasks
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_run_campaign_task, task, submit=submit, resume=resume): task
                for task in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["task_id"])

    campaign_id = str(settings["campaign"]["id"])
    output = Path(base["output"].get("directory") or "outputs/module_C_potential_benchmark")
    summary_path = output / "campaigns" / campaign_id / f"{normalized_stage}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": _campaign_status(results),
        "campaign_id": campaign_id,
        "stage": normalized_stage,
        "task_count": len(results),
        "results": results,
        "output_path": str(summary_path),
    }
    save_benchmark_data(summary_path, report)
    return report


def build_campaign_tasks(
    normalized_config: dict[str, Any],
    *,
    stage: str,
    selected_systems: Sequence[str] = (),
    selected_models: Sequence[str] = (),
) -> list[dict[str, Any]]:
    settings = normalized_config["benchmark"]["potential_benchmark"]
    campaign = settings["campaign"]
    systems = campaign.get("systems") or [normalized_config["material"]]
    models = _campaign_models(settings)
    system_filter = {value.casefold() for value in selected_systems}
    model_filter = {value.casefold() for value in selected_models}
    seeds = normalized_config["simulation"]["seeds"]
    if stage in {"screening", "active_learning"}:
        seeds = seeds[:1]
    tasks = []
    for system in systems:
        if not isinstance(system, dict):
            raise TypeError("Each campaign system must be an object.")
        system_name = str(system.get("name") or Path(str(system.get("cif_path", "material"))).stem)
        if system_filter and system_name.casefold() not in system_filter:
            continue
        task_models = [None] if stage == "active_learning" else models
        for model in task_models:
            model_name = "mace_mp_0a_active_learning" if model is None else str(model.get("name", model.get("model", "model")))
            if model_filter and model_name.casefold() not in model_filter:
                continue
            for seed in seeds:
                config = deepcopy(normalized_config)
                config["material"] = {**normalized_config["material"], **system}
                if model is not None:
                    config["benchmark"]["potential_benchmark"].setdefault("mlip_mc", {})["model"] = deepcopy(model)
                campaign_id = str(campaign["id"])
                task_id = "__".join(
                    (_safe_id(system_name), _safe_id(model_name), stage, f"seed{seed}")
                )
                output = config["output"]
                output["run_id"] = f"{campaign_id}/{task_id}"
                config["simulation"]["seeds"] = [int(seed)]
                tasks.append(
                    {
                        "task_id": task_id,
                        "system": system_name,
                        "model": model_name,
                        "stage": stage,
                        "seed": int(seed),
                        "config": config,
                    }
                )
    return tasks


def cross_check_campaign_engine(
    raw_config: dict[str, Any],
    *,
    cross_check_engine: str,
    stage: str = "screening",
    selected_systems: Sequence[str] = (),
    selected_models: Sequence[str] = (),
) -> dict[str, Any]:
    base = normalize_config(raw_config)
    task = build_campaign_tasks(
        base,
        stage=stage,
        selected_systems=selected_systems,
        selected_models=selected_models,
    )[0]
    plan = build_run_plan(task["config"])
    primary_name = plan["benchmark"]["potential_benchmark"]["adsorption_engine"]
    primary = _run_adsorption(plan, stage, engine_name=primary_name)
    secondary = _run_adsorption(plan, stage, engine_name=cross_check_engine, output_suffix="cross_check")
    comparison = compare_engine_results(primary, secondary)
    output = Path(plan["outputs"]["directory"]) / "runs" / str(plan["outputs"]["run_id"])
    path = output / "results" / "engine_cross_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison["output_path"] = str(path)
    save_benchmark_data(path, comparison)
    return comparison


def analyze_module_c_campaign(run_plan: dict[str, Any]) -> dict[str, Any]:
    """Aggregate campaign task results without assuming a fixed MOF list."""
    settings = run_plan["benchmark"]["potential_benchmark"]
    campaign_id = str(settings["campaign"]["id"])
    output_root = Path(run_plan["outputs"]["directory"])
    root = output_root / "runs" / campaign_id
    rows = []
    if root.exists():
        for result_path in sorted(root.glob("**/results/*.json")):
            if result_path.name in {"engine_cross_check.json"}:
                continue
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "path": str(result_path),
                    "engine": data.get("engine"),
                    "method": data.get("method"),
                    "status": data.get("status"),
                    "sampling_status": data.get("sampling_status"),
                    "potential_status": data.get("potential_status"),
                    "reference_status": data.get("reference_status"),
                    "runtime_seconds": data.get("runtime_seconds"),
                }
            )
    report_path = output_root / "campaigns" / campaign_id / "module_c_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "completed",
        "campaign_id": campaign_id,
        "result_count": len(rows),
        "results": rows,
        "output_path": str(report_path),
    }
    save_benchmark_data(report_path, report)
    return report


def _run_campaign_task(task: dict[str, Any], *, submit: bool, resume: bool) -> dict[str, Any]:
    plan = build_run_plan(task["config"])
    if plan["benchmark"]["applicability"].get("can_attempt_simulation") is False:
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "reason": plan["benchmark"]["applicability"]["reason"],
        }
    if task["stage"] == "active_learning":
        result = run_active_learning(plan, submit=submit, resume=resume)
    else:
        result = _run_adsorption(plan, task["stage"])
    return {"task_id": task["task_id"], **result}


def _run_adsorption(
    plan: dict[str, Any],
    stage: str,
    *,
    engine_name: str | None = None,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    potential = plan["benchmark"]["potential_benchmark"]
    model = potential.get("mlip_mc", {}).get("model")
    if not isinstance(model, dict):
        raise ValueError("Campaign adsorption tasks require at least one model definition.")
    potential_status = plan["benchmark"]["applicability"].get(
        "potential_status",
        "unassessed",
    )
    if (
        potential["validation"]["mode"] == "active_learning"
        and stage in {"widom", "gcmc_pilot", "gcmc_production"}
    ):
        model, _ = _validated_campaign_model(plan)
        potential_status = "dft_validated"
    forcefield = plan["resources"]["forcefield"]
    component = plan["adsorbates"]["components"][0]
    framework, adsorbate, _ = load_host_guest_system(
        plan["resources"]["cif_path"],
        forcefield["adsorbates"][component],
        forcefield["files"]["pseudo_atoms"],
        cell_representation=plan["simulation"]["cell_representation"],
        unit_cells=plan["simulation"]["unit_cells"],
        cutoff_A=float(plan["simulation"]["cutoff_A"]),
        minimum_image_policy=plan["simulation"]["minimum_image_policy"],
    )
    physical = [index for index, mass in enumerate(adsorbate.get_masses()) if float(mass) >= 0.01]
    adsorbate = adsorbate[physical]
    engine = build_adsorption_engine(engine_name or potential["adsorption_engine"])
    calculator = (
        None
        if engine.name == "lammps"
        else build_ase_calculator(model)
    )
    stage_settings = dict(potential["campaign"]["stages"][stage])
    task_root = Path(plan["outputs"]["directory"]) / "runs" / str(plan["outputs"]["run_id"])
    output_dir = task_root / "engine" / engine.name / stage
    if output_suffix:
        output_dir = task_root / "engine" / output_suffix / engine.name / stage
    settings = {
        **stage_settings,
        "temperature_K": float(plan["conditions"]["temperature_K"]),
        "pressures_bar": _stage_pressures(plan["conditions"]["pressures_bar"], stage_settings),
        "seed": int(plan["simulation"]["seeds"][0]),
        "device": str(model.get("device", "cpu")),
        "output_directory": str(output_dir),
        "cutoff_A": float(plan["simulation"]["cutoff_A"]),
        "component": component,
        "vdw_radius_aliases": model.get("vdw_radius_aliases"),
        "save_csv": plan["outputs"].get("save_csv", True),
        "save_plots": plan["outputs"].get("save_plots", True),
    }
    if engine.name == "lammps":
        if str(stage_settings.get("method", "")).casefold() != "gcmc":
            raise NotImplementedError(
                "The classical LAMMPS campaign engine supports GCMC stages, "
                "not Widom."
            )
        from pipeline.materialization import materialize_benchmark
        from pipeline.prepare import prepare_benchmark

        classical_plan = deepcopy(plan)
        classical_plan["conditions"]["pressures_bar"] = list(
            settings["pressures_bar"]
        )
        classical_plan["simulation"]["initialization_cycles"] = int(
            stage_settings.get("equilibration_steps", 10_000)
        )
        classical_plan["simulation"]["cycles"] = int(
            stage_settings.get("production_steps", 20_000)
        )
        settings["materialized_plan"] = materialize_benchmark(
            prepare_benchmark(classical_plan)
        )
        settings["jobs"] = 1
    system = {"framework": framework, "adsorbate": adsorbate}
    method = str(stage_settings.get("method", "widom")).casefold()
    result = engine.run_widom(system, calculator, settings) if method == "widom" else engine.run_gcmc(system, calculator, settings)
    applicability = plan["benchmark"]["applicability"]
    result.update(
        {
            "material": plan["material"]["material_id"],
            "model": model.get("name", model.get("model")),
            "stage": stage,
            "sampling_status": _sampling_status(result),
            "potential_status": potential_status,
            "reference_status": applicability.get("reference_status", "not_assessed"),
        }
    )
    result_path = task_root / "results" / f"{stage}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result["campaign_result_path"] = str(result_path)
    save_benchmark_data(result_path, result)
    return result


def _campaign_models(settings: dict[str, Any]) -> list[dict[str, Any]]:
    models = settings.get("campaign", {}).get("models", [])
    if not models:
        legacy = settings.get("mlip_mc", {}).get("model")
        models = [legacy] if isinstance(legacy, dict) else []
    if not models:
        models = [
            {
                "backend": "mace-torch",
                "name": "mace_mp_0a_small",
                "model": "small",
                "device": "cpu",
                "default_dtype": "float64",
                "dispersion": False,
            }
        ]
    if not all(isinstance(model, dict) for model in models):
        raise TypeError("Every campaign model must be an object.")
    return models


def _stage_pressures(values: Sequence[float], settings: dict[str, Any]) -> list[float]:
    pressures = sorted({float(value) for value in values})
    if settings.get("pressure_selection") != "low_mid_high" or len(pressures) <= 3:
        return pressures
    return [pressures[0], pressures[len(pressures) // 2], pressures[-1]]


def _sampling_status(result: dict[str, Any]) -> str:
    if result.get("status") != "completed":
        return "failed"
    if result.get("method") == "widom":
        uncertainty = result.get("uncertainty")
        return "converged" if uncertainty else "completed_unassessed"
    return "completed_unassessed"


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_").casefold()


def _campaign_status(results: Sequence[dict[str, Any]]) -> str:
    statuses = {str(item.get("status", "")) for item in results}
    if "failed" in statuses:
        return "failed"
    if statuses & {
        "awaiting_cutoff_test",
        "awaiting_dft",
        "training_prepared",
        "models_ready",
    }:
        return "in_progress"
    if statuses & {"adaptation_not_converged", "cutoff_not_converged"}:
        return "completed_with_failed_validation"
    return "completed"


def _validated_campaign_model(
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = Path(str(plan["outputs"]["run_id"]))
    if len(run_id.parts) < 2:
        raise RuntimeError(
            "Active-learning production requires a campaign-scoped run_id."
        )
    campaign_id = run_id.parts[0]
    root = Path(plan["outputs"]["directory"]) / "runs" / campaign_id
    material_id = str(plan["material"]["material_id"])
    states = []
    for state_path in sorted(
        root.glob("*__active_learning__*/active_learning/state.json")
    ):
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if str(data.get("material_id")) == material_id:
            states.append((state_path, data))
    if not states:
        raise RuntimeError(
            "No active-learning state exists for this material. Run "
            "--run-module-c --stage active-learning first."
        )
    state_path, state = states[-1]
    if state.get("status") != "dft_validated":
        raise RuntimeError(
            "GCMC/Widom production is blocked because active learning is "
            f"{state.get('status')!r}, not 'dft_validated'. State: {state_path}"
        )
    model_paths = state.get("model_paths", [])
    validation = state.get("validation") or {}
    metrics = validation.get("metrics", {})
    best_index = int(metrics.get("best_model_index", 0))
    if best_index >= len(model_paths):
        raise RuntimeError("Validated active-learning state has no usable model path.")
    return (
        {
            "backend": "mace-torch",
            "name": "mace_mp_0a_finetuned",
            "model": str(model_paths[best_index]),
            "device": plan["benchmark"]["potential_benchmark"][
                "active_learning"
            ].get("device", "cuda"),
            "default_dtype": "float64",
            "dispersion": False,
        },
        state,
    )
