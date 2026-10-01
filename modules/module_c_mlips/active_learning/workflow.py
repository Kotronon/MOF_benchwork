"""Persistent orchestration for CP2K-to-MACE active learning."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from modules.module_c_mlips.active_learning.candidates import generate_candidate_pool
from modules.module_c_mlips.active_learning.dft import CP2KLabeler, cutoff_converged
from modules.module_c_mlips.active_learning.models import (
    ActiveLearningSettings,
    ActiveLearningState,
)
from modules.module_c_mlips.active_learning.training import (
    build_mace_finetune_commands,
    classify_validation_metrics,
    evaluate_mace_committee,
    execute_mace_commands,
    load_labels,
    select_committee_candidates,
    write_mace_datasets,
)
from modules.module_c_mlips.datasets.common import load_host_guest_system
from modules.module_c_mlips.workspace import working_directory
from pipeline.config import save_benchmark_data


TERMINAL_STATES = {
    "dft_validated",
    "adaptation_not_converged",
    "cutoff_not_converged",
}


def run_active_learning(
    run_plan: dict[str, Any],
    *,
    submit: bool = False,
    train: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Advance the restartable CP2K-to-MACE workflow as far as possible."""
    potential = run_plan["benchmark"]["potential_benchmark"]
    if potential["validation"]["mode"] != "active_learning":
        raise ValueError("Active learning requires validation.mode='active_learning'.")
    applicability = run_plan["benchmark"].get("applicability", {})
    if applicability.get("dft_configuration_status") == "requires_dft_configuration":
        raise ValueError(
            "This structure requires an explicitly reviewed CP2K charge, multiplicity, "
            "pseudopotential, and dispersion configuration before active learning."
        )

    settings = ActiveLearningSettings.from_dict(potential["active_learning"])
    root = working_directory(run_plan) / "active_learning"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    if state_path.exists():
        if run_plan["outputs"].get("overwrite", False) and not resume:
            shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        else:
            if not resume:
                raise FileExistsError(
                    f"Active-learning state exists: {state_path}. Re-run with "
                    "--resume or --overwrite."
                )
            state = ActiveLearningState.load(state_path)
    if not state_path.exists():
        state = ActiveLearningState(
            campaign_id=str(
                run_plan["outputs"].get("run_id") or "module_c_campaign"
            ),
            material_id=str(run_plan["material"]["material_id"]),
        )

    if state.status in TERMINAL_STATES:
        return _state_report(state, state_path)
    if state.status == "new":
        _initialize(run_plan, settings, state, root)
        state.save(state_path)

    if state.status == "awaiting_cutoff_test":
        waiting = _advance_cutoff_test(settings, state, root, submit=submit)
        state.save(state_path)
        if waiting:
            return _state_report(
                state,
                state_path,
                waiting_for="cp2k_cutoff_test",
            )
        if state.status == "cutoff_not_converged":
            return _state_report(state, state_path)
        _prepare_iteration_labels(settings, state, root, iteration=0)
        state.status = "awaiting_dft"
        state.save(state_path)

    cp2k_settings = {**settings.cp2k}
    if state.selected_cutoff_Ry is not None:
        cp2k_settings["cutoff_Ry"] = state.selected_cutoff_Ry
    labeler = CP2KLabeler(cp2k_settings)
    jobs_path = _jobs_path(root, state.iteration)
    if state.status == "awaiting_dft":
        jobs = _load_jobs(jobs_path)
        labels = labeler.collect(jobs)
        state.labeled_count = len(load_labels(root))
        if len(labels) < len(jobs):
            job_statuses = labeler.status(jobs)
            _raise_failed_jobs(job_statuses)
            if submit:
                jobs = _submit_unfinished_jobs(labeler, jobs)
                save_benchmark_data(jobs_path, {"jobs": jobs})
                job_statuses = labeler.status(jobs)
                state.history.append(
                    {
                        "iteration": state.iteration,
                        "event": "dft_submitted",
                        "job_count": len(jobs),
                    }
                )
                state.save(state_path)
            return _state_report(
                state,
                state_path,
                waiting_for="cp2k_labels",
                extra={"job_statuses": job_statuses},
            )
        state.pending_ids = []
        state.status = "ready_to_train"
        state.history.append(
            {
                "iteration": state.iteration,
                "event": "dft_collected",
                "iteration_label_count": len(labels),
                "total_label_count": state.labeled_count,
            }
        )
        state.save(state_path)

    if state.status == "ready_to_train":
        labels = load_labels(root)
        datasets = write_mace_datasets(
            labels,
            training_ids=set(state.training_ids),
            validation_ids=set(state.validation_ids),
            test_ids=set(state.test_ids),
            output_directory=(
                root / f"iteration_{state.iteration:02d}" / "datasets"
            ),
        )
        commands = build_mace_finetune_commands(
            datasets,
            root / f"iteration_{state.iteration:02d}" / "training",
            committee_size=settings.committee_size,
            base_model=settings.base_model,
            device=settings.device,
        )
        if not train:
            state.status = "training_prepared"
            state.history.append(
                {"iteration": state.iteration, "event": "training_prepared"}
            )
            state.save(state_path)
            return _state_report(
                state,
                state_path,
                waiting_for="mace_training",
                extra={
                    "training_commands": str(
                        _training_commands_path(root, state.iteration)
                    )
                },
            )
        _execute_training(commands, state, state_path)

    if state.status == "training_prepared":
        if not train:
            return _state_report(
                state,
                state_path,
                waiting_for="mace_training",
            )
        commands = json.loads(
            _training_commands_path(root, state.iteration).read_text(
                encoding="utf-8"
            )
        )["commands"]
        _execute_training(commands, state, state_path)

    if state.status == "models_ready":
        labels = load_labels(root)
        metrics = evaluate_mace_committee(
            state.model_paths,
            labels,
            test_ids=set(state.test_ids),
            device=settings.device,
        )
        validation = classify_validation_metrics(metrics, settings)
        validation.update(
            {
                "iteration": state.iteration,
                "test_ids": list(state.test_ids),
                "model_paths": list(state.model_paths),
            }
        )
        validation_path = (
            root
            / f"iteration_{state.iteration:02d}"
            / "validation_metrics.json"
        )
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        save_benchmark_data(validation_path, validation)
        state.validation = validation
        state.stable_rounds = (
            state.stable_rounds + 1
            if validation["status"] == "passed"
            else 0
        )
        state.history.append(
            {
                "iteration": state.iteration,
                "event": "committee_validated",
                "status": validation["status"],
                "stable_rounds": state.stable_rounds,
                "metrics_path": str(validation_path),
            }
        )
        if state.stable_rounds >= settings.stable_rounds_required:
            state.status = "dft_validated"
            state.save(state_path)
            return _state_report(
                state,
                state_path,
                extra={"potential_status": "dft_validated"},
            )
        if not _can_add_round(state, settings):
            state.status = "adaptation_not_converged"
            state.failure_reason = (
                "The independent test and committee-disagreement gates were not "
                "stable for two consecutive rounds before the configured budget "
                "was exhausted."
            )
            state.save(state_path)
            return _state_report(state, state_path)
        _prepare_next_round(run_plan, settings, state, root)
        state.save(state_path)
        return _state_report(
            state,
            state_path,
            waiting_for="cp2k_labels",
        )

    return _state_report(state, state_path)


def _initialize(
    run_plan: dict[str, Any],
    settings: ActiveLearningSettings,
    state: ActiveLearningState,
    root: Path,
) -> None:
    framework, adsorbate = _load_system(run_plan)
    candidates = generate_candidate_pool(
        framework,
        adsorbate,
        root / "iteration_00" / "candidates",
        count=settings.bootstrap_count,
        seed=int(run_plan["simulation"]["seeds"][0]),
        fractions=settings.candidate_fractions,
        iteration=0,
    )
    test_records, initial_training_records = _stratified_split(
        candidates,
        test_count=settings.test_configurations,
        fractions=settings.candidate_fractions,
    )
    validation_records, training_records = _stratified_split(
        initial_training_records,
        test_count=settings.validation_configurations,
        fractions=settings.candidate_fractions,
    )
    state.test_ids = [
        item["candidate_id"] for item in test_records
    ]
    state.training_ids = [
        item["candidate_id"] for item in training_records
    ]
    state.validation_ids = [
        item["candidate_id"] for item in validation_records
    ]
    baselines = _write_baseline_candidates(
        framework,
        adsorbate,
        root / "iteration_00" / "baselines",
    )
    state.baseline_ids = [item["candidate_id"] for item in baselines]
    state.training_ids.extend(state.baseline_ids)
    state.pending_ids = [
        item["candidate_id"] for item in candidates
    ] + state.baseline_ids
    state.candidate_count = len(candidates)
    cutoff_candidate = training_records[0]
    jobs = CP2KLabeler(settings.cp2k).prepare_cutoff_test(
        cutoff_candidate,
        root / "cutoff_test",
    )
    state.status = "awaiting_cutoff_test"
    state.history.append(
        {
            "iteration": 0,
            "event": "initialized",
            "candidate_count": len(candidates),
            "training_count": len(state.training_ids),
            "validation_count": len(state.validation_ids),
            "test_count": len(state.test_ids),
            "baseline_job_count": len(baselines),
            "cutoff_test_job_count": len(jobs),
        }
    )


def _advance_cutoff_test(
    settings: ActiveLearningSettings,
    state: ActiveLearningState,
    root: Path,
    *,
    submit: bool,
) -> bool:
    jobs_path = root / "cutoff_test" / "cp2k_jobs.json"
    jobs = _load_jobs(jobs_path)
    labeler = CP2KLabeler(settings.cp2k)
    labels = labeler.collect(jobs)
    if len(labels) < len(jobs):
        _raise_failed_jobs(labeler.status(jobs))
        if submit:
            jobs = _submit_unfinished_jobs(labeler, jobs)
            save_benchmark_data(jobs_path, {"jobs": jobs})
        return True
    by_id = {str(label["candidate_id"]): label for label in labels}
    results = {
        int(job["cutoff_Ry"]): by_id[str(job["candidate_id"])]
        for job in jobs
    }
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError(
            "ASE is required to assess the CP2K cutoff test."
        ) from exc
    atom_count = len(read(str(jobs[0]["structure_path"])))
    report = cutoff_converged(results, atom_count=atom_count)
    save_benchmark_data(
        root / "cutoff_test" / "cutoff_convergence.json",
        report,
    )
    if report["status"] != "passed":
        state.status = "cutoff_not_converged"
        state.failure_reason = (
            "The CP2K 400/600/800 Ry cutoff convergence gate failed."
        )
        return False
    state.selected_cutoff_Ry = int(report["selected_cutoff_Ry"])
    state.history.append(
        {
            "iteration": 0,
            "event": "cutoff_converged",
            "selected_cutoff_Ry": state.selected_cutoff_Ry,
        }
    )
    return False


def _prepare_iteration_labels(
    settings: ActiveLearningSettings,
    state: ActiveLearningState,
    root: Path,
    *,
    iteration: int,
) -> None:
    if iteration == 0:
        records = _read_manifest(
            root
            / "iteration_00"
            / "candidates"
            / "candidate_manifest.json"
        )
        records += _read_manifest(
            root
            / "iteration_00"
            / "baselines"
            / "candidate_manifest.json"
        )
    else:
        records = _read_manifest(
            root
            / f"iteration_{iteration:02d}"
            / "selection_manifest.json"
        )
    cp2k = {**settings.cp2k, "cutoff_Ry": state.selected_cutoff_Ry}
    CP2KLabeler(cp2k).prepare(
        records,
        root / f"iteration_{iteration:02d}" / "cp2k",
    )


def _prepare_next_round(
    run_plan: dict[str, Any],
    settings: ActiveLearningSettings,
    state: ActiveLearningState,
    root: Path,
) -> None:
    next_iteration = state.iteration + 1
    remaining = settings.maximum_configurations - state.candidate_count
    selection_count = min(settings.configurations_per_round, remaining)
    framework, adsorbate = _load_system(run_plan)
    pool = generate_candidate_pool(
        framework,
        adsorbate,
        root
        / f"iteration_{next_iteration:02d}"
        / "candidate_pool",
        count=max(selection_count, selection_count * 4),
        seed=int(run_plan["simulation"]["seeds"][0]),
        fractions=settings.candidate_fractions,
        iteration=next_iteration,
    )
    selected = select_committee_candidates(
        pool,
        state.model_paths,
        count=selection_count,
        device=settings.device,
    )
    save_benchmark_data(
        root
        / f"iteration_{next_iteration:02d}"
        / "selection_manifest.json",
        {"candidates": selected},
    )
    selected_ids = [str(item["candidate_id"]) for item in selected]
    state.training_ids.extend(selected_ids)
    state.pending_ids = selected_ids
    state.candidate_count += len(selected)
    state.iteration = next_iteration
    state.model_paths = []
    _prepare_iteration_labels(
        settings,
        state,
        root,
        iteration=next_iteration,
    )
    state.status = "awaiting_dft"
    state.history.append(
        {
            "iteration": next_iteration,
            "event": "committee_candidates_selected",
            "selected_count": len(selected),
            "candidate_count": state.candidate_count,
        }
    )


def _load_system(run_plan: dict[str, Any]) -> tuple[Any, Any]:
    forcefield = run_plan["resources"]["forcefield"]
    component = run_plan["adsorbates"]["components"][0]
    framework, adsorbate, _ = load_host_guest_system(
        run_plan["resources"]["cif_path"],
        forcefield["adsorbates"][component],
        forcefield["files"]["pseudo_atoms"],
        cell_representation=run_plan["simulation"]["cell_representation"],
        unit_cells=run_plan["simulation"]["unit_cells"],
        cutoff_A=float(run_plan["simulation"]["cutoff_A"]),
        minimum_image_policy=run_plan["simulation"][
            "minimum_image_policy"
        ],
    )
    physical = [
        index
        for index, mass in enumerate(adsorbate.get_masses())
        if float(mass) >= 0.01
    ]
    return framework, adsorbate[physical]


def _write_baseline_candidates(
    framework: Any,
    adsorbate: Any,
    output_directory: Path,
) -> list[dict[str, Any]]:
    try:
        from ase.io import write
    except ImportError as exc:
        raise ImportError(
            "ASE is required to write active-learning baselines."
        ) from exc
    output_directory.mkdir(parents=True, exist_ok=True)
    isolated = adsorbate.copy()
    isolated.set_cell([30.0, 30.0, 30.0])
    isolated.center()
    isolated.set_pbc(True)
    records = []
    for candidate_id, atoms, candidate_type in (
        (
            "baseline_framework",
            framework,
            "empty_framework_baseline",
        ),
        (
            "baseline_co2",
            isolated,
            "isolated_adsorbate_baseline",
        ),
    ):
        path = output_directory / f"{candidate_id}.extxyz"
        current = atoms.copy()
        current.info.update(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
            }
        )
        write(path, current, format="extxyz")
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "structure_path": str(path),
                "label_status": "pending",
            }
        )
    save_benchmark_data(
        output_directory / "candidate_manifest.json",
        {"candidates": records},
    )
    return records


def _execute_training(
    commands: list[dict[str, Any]],
    state: ActiveLearningState,
    state_path: Path,
) -> None:
    trained = execute_mace_commands(commands)
    state.model_paths = [
        str(item["model_path"]) for item in trained
    ]
    state.status = "models_ready"
    state.history.append(
        {
            "iteration": state.iteration,
            "event": "training_completed",
            "models": state.model_paths,
        }
    )
    state.save(state_path)


def _submit_unfinished_jobs(
    labeler: CP2KLabeler,
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pending = [
        job
        for job in jobs
        if not Path(job["output_path"]).is_file()
        and job.get("status") in {"prepared", "failed"}
    ]
    if not pending:
        return jobs
    submitted = {
        item["candidate_id"]: item
        for item in labeler.submit(pending)
    }
    return [
        submitted.get(job["candidate_id"], job) for job in jobs
    ]


def _raise_failed_jobs(statuses: list[dict[str, Any]]) -> None:
    failed = [
        item for item in statuses if item.get("status") == "failed"
    ]
    if failed:
        identifiers = ", ".join(
            str(item["candidate_id"]) for item in failed
        )
        raise RuntimeError(
            "CP2K failed for the following configurations: "
            f"{identifiers}. Inspect their cp2k.out files, correct the "
            "configuration, and resume."
        )


def _can_add_round(
    state: ActiveLearningState,
    settings: ActiveLearningSettings,
) -> bool:
    return (
        state.iteration < settings.rounds
        and state.candidate_count < settings.maximum_configurations
    )


def _stratified_split(
    candidates: list[dict[str, Any]],
    *,
    test_count: int,
    fractions: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_count <= 0 or test_count >= len(candidates):
        raise ValueError(
            "The held-out test count must leave at least one training candidate."
        )
    categories = tuple(fractions)
    exact = {
        category: test_count * float(fractions[category])
        for category in categories
    }
    quotas = {
        category: int(exact[category])
        for category in categories
    }
    remainder = test_count - sum(quotas.values())
    order = sorted(
        categories,
        key=lambda category: exact[category] - quotas[category],
        reverse=True,
    )
    for category in order[:remainder]:
        quotas[category] += 1

    selected_ids: set[str] = set()
    tests = []
    for category in categories:
        available = [
            item
            for item in candidates
            if item.get("candidate_type") == category
        ]
        if len(available) < quotas[category]:
            raise ValueError(
                f"Not enough {category} candidates for the held-out split."
            )
        chosen = available[: quotas[category]]
        tests.extend(chosen)
        selected_ids.update(str(item["candidate_id"]) for item in chosen)
    training = [
        item
        for item in candidates
        if str(item["candidate_id"]) not in selected_ids
    ]
    return tests, training


def _jobs_path(root: Path, iteration: int) -> Path:
    return (
        root
        / f"iteration_{iteration:02d}"
        / "cp2k"
        / "cp2k_jobs.json"
    )


def _training_commands_path(root: Path, iteration: int) -> Path:
    return (
        root
        / f"iteration_{iteration:02d}"
        / "training"
        / "training_commands.json"
    )


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CP2K job manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["jobs"]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing active-learning manifest: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["candidates"]


def _state_report(
    state: ActiveLearningState,
    state_path: Path,
    *,
    waiting_for: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": state.status,
        "active_learning_state": state.to_dict(),
        "state_path": str(state_path),
        "waiting_for": waiting_for,
        **(extra or {}),
    }
