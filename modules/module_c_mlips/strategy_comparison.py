"""Compare CP2K-local adaptation with CP2K-free pretrained strategies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from modules.module_c_mlips.models import InteractionConfiguration
from modules.module_c_mlips.model_assets import (
    ensure_umof_model_asset,
    is_umof_specification,
)
from modules.module_c_mlips.potential_backends.base import PotentialBackend
from modules.module_c_mlips.workspace import configure_runtime_cache, working_directory
from pipeline.config import save_benchmark_data


STRATEGY_METADATA = {
    "cp2k_local_finetune": {
        "uses_local_cp2k_labels": True,
        "fine_tuned_locally": True,
        "reference_level": "local_cp2k",
    },
    "umof_pretrained": {
        "uses_local_cp2k_labels": False,
        "fine_tuned_locally": False,
        "reference_level": "uMOF r2SCAN-D4 pretraining/fine-tuning",
    },
}


def run_strategy_comparison(
    run_plan: dict[str, Any],
    *,
    download_missing: bool = False,
) -> dict[str, Any]:
    """Run a Module-C strategy comparison without launching CP2K jobs."""
    if run_plan.get("module", {}).get("id") != "C":
        raise ValueError("Strategy comparisons require Module C.")
    potential = run_plan["benchmark"].get("potential_benchmark", {})
    if not isinstance(potential, dict):
        raise TypeError("'benchmark.potential_benchmark' must be an object.")
    comparison = potential.get("comparison", {})
    if not isinstance(comparison, dict):
        raise TypeError("'benchmark.potential_benchmark.comparison' must be an object.")

    strategies = _comparison_strategies(comparison)
    workdir = working_directory(run_plan)
    configure_runtime_cache(Path(run_plan["outputs"]["directory"]) / ".cache" / "plots")
    for child in (workdir / "results", workdir / "reports", workdir / "configurations", workdir / "backends"):
        child.mkdir(parents=True, exist_ok=True)

    configurations, cp2k_reference_available = _load_comparison_configurations(
        run_plan,
        potential,
        comparison,
    )
    backend_specs = _strategy_backend_specs(
        strategies,
        potential,
        comparison,
        download_missing=download_missing,
    )
    from modules.module_c_mlips.workflow import _build_backends

    backends = _build_backends(backend_specs, run_plan, workdir / "backends")
    started = perf_counter()
    evaluations = _evaluate_backends(configurations, backends)
    strategy_rows = _summarize_strategies(
        strategies,
        backend_specs,
        evaluations,
        cp2k_reference_available=cp2k_reference_available,
    )
    report = {
        "status": "completed",
        "comparison_type": "cp2k_local_vs_cp2k_free_pretrained",
        "cp2k_jobs_started": False,
        "cp2k_reference_available": cp2k_reference_available,
        "strategies": strategies,
        "configuration_count": len(configurations),
        "backend_count": len(backends),
        "runtime_seconds": perf_counter() - started,
        "rows": strategy_rows,
        "evaluations": evaluations,
        "outputs": {},
    }

    results_dir = workdir / "results"
    report_path = results_dir / "strategy_comparison.json"
    csv_path = results_dir / "strategy_comparison.csv"
    save_benchmark_data(report_path, report)
    _write_strategy_csv(csv_path, strategy_rows)
    report["outputs"]["json"] = str(report_path)
    report["outputs"]["csv"] = str(csv_path)
    if bool(run_plan["outputs"].get("save_plots", True)):
        plot_path = _write_strategy_plot(
            workdir / "reports" / "cp2k_reference_comparison.png",
            strategy_rows,
        )
        if plot_path is not None:
            report["outputs"]["cp2k_reference_comparison_png"] = str(plot_path)
    save_benchmark_data(report_path, report)
    return report


def _comparison_strategies(comparison: dict[str, Any]) -> list[str]:
    strategies = comparison.get(
        "strategies",
        ["cp2k_local_finetune", "umof_pretrained"],
    )
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("'comparison.strategies' must be a non-empty list.")
    normalized = [str(strategy).strip().casefold() for strategy in strategies]
    unknown = sorted(set(normalized) - set(STRATEGY_METADATA))
    if unknown:
        raise ValueError(f"Unsupported comparison strategies: {', '.join(unknown)}")
    return normalized


def _load_comparison_configurations(
    run_plan: dict[str, Any],
    potential: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[list[InteractionConfiguration], bool]:
    label_root = comparison.get("cp2k_label_root")
    if label_root:
        configurations = load_cp2k_labeled_configurations(label_root)
        if configurations:
            return configurations, True

    from modules.module_c_mlips.workflow import _build_configurations

    components = run_plan["adsorbates"]["components"]
    if len(components) != 1:
        raise ValueError("Strategy comparisons currently support one adsorbate.")
    settings = {**potential, **comparison.get("configuration", {})}
    configurations = _build_configurations(
        settings,
        run_plan,
        components[0],
        run_plan["resources"]["forcefield"],
    )
    return configurations, any(
        configuration.reference_interaction_energy_ev is not None
        or configuration.reference_forces_ev_per_angstrom is not None
        for configuration in configurations
    )


def load_cp2k_labeled_configurations(label_root: str | Path) -> list[InteractionConfiguration]:
    """Build configurations from already completed CP2K label files."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required to load CP2K-labeled configurations.") from exc
    configurations = []
    for label_path in sorted(Path(label_root).glob("iteration_*/cp2k/*/label.json")):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        atoms = read(str(label["structure_path"]))
        framework_atom_count = int(
            atoms.info.get("framework_atom_count")
            or label.get("framework_atom_count")
            or _infer_framework_atom_count(atoms)
        )
        if framework_atom_count <= 0 or framework_atom_count > len(atoms):
            raise ValueError(f"Invalid framework atom count in {label_path}.")
        configurations.append(
            InteractionConfiguration(
                configuration_id=str(label["candidate_id"]),
                material=str(atoms.info.get("material", "unknown")),
                adsorbate=str(atoms.info.get("adsorbate", "CO2")),
                atoms=atoms,
                framework_indices=list(range(framework_atom_count)),
                adsorbate_indices=list(range(framework_atom_count, len(atoms))),
                reference_interaction_energy_ev=float(label["energy_ev"]),
                reference_forces_ev_per_angstrom=label.get("forces_ev_per_A"),
                source="cp2k_active_learning_labels",
                metadata={
                    "cp2k_label_path": str(label_path),
                    "reference_energy_mode": "total_energy",
                    "reference_level": label.get("dft_profile", "cp2k"),
                },
            )
        )
    return configurations


def _strategy_backend_specs(
    strategies: Sequence[str],
    potential: dict[str, Any],
    comparison: dict[str, Any],
    *,
    download_missing: bool = False,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "cp2k_local_finetune" in strategies:
        specs.extend(comparison.get("cp2k_finetuned_backends", []))
    if "umof_pretrained" in strategies:
        umof = comparison.get("umof_backends")
        if umof is None:
            umof = [
                spec
                for spec in potential.get("backends", [])
                if _is_umof_spec(spec)
            ]
        if not umof:
            raise FileNotFoundError(
                "uMOF checkpoint required: define comparison.umof_backends "
                "with local uMOF-MH/uMOF-POLAR model paths."
            )
        specs.extend(umof)
    specs.extend(comparison.get("baseline_backends", []))
    if not specs:
        raise ValueError("No backends configured for strategy comparison.")
    for spec in specs:
        if _is_umof_spec(spec):
            _require_umof_checkpoint(spec, download_missing=download_missing)
    return specs


def _evaluate_backends(
    configurations: Sequence[InteractionConfiguration],
    backends: Sequence[PotentialBackend],
) -> list[dict[str, Any]]:
    evaluations = []
    for configuration in configurations:
        for backend in backends:
            result = backend.evaluate(configuration)
            evaluations.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "backend": backend.name,
                    "energy_ev": result.energy_ev,
                    "forces_ev_per_angstrom": result.forces_ev_per_angstrom,
                    "runtime_seconds": result.runtime_seconds,
                    "reference_energy_ev": configuration.reference_interaction_energy_ev,
                    "reference_forces_ev_per_angstrom": configuration.reference_forces_ev_per_angstrom,
                    "atom_count": len(configuration.atoms),
                    "metadata": result.metadata,
                }
            )
    return evaluations


def _summarize_strategies(
    strategies: Sequence[str],
    backend_specs: Sequence[dict[str, Any]],
    evaluations: Sequence[dict[str, Any]],
    *,
    cp2k_reference_available: bool,
) -> list[dict[str, Any]]:
    rows = []
    for spec in backend_specs:
        name = str(spec.get("name", spec.get("type", "backend")))
        strategy = _strategy_for_backend(spec)
        if strategy not in strategies and not spec.get("strategy"):
            strategy = "baseline"
        current = [item for item in evaluations if item["backend"] == name]
        if not current:
            continue
        energy_errors = []
        force_errors = []
        for item in current:
            if item["reference_energy_ev"] is not None:
                energy_errors.append(abs(item["energy_ev"] - item["reference_energy_ev"]))
            reference_forces = item.get("reference_forces_ev_per_angstrom")
            if reference_forces is not None:
                predicted = np.asarray(item["forces_ev_per_angstrom"], dtype=float)
                reference = np.asarray(reference_forces, dtype=float)
                if predicted.shape == reference.shape:
                    force_errors.extend(np.linalg.norm(predicted - reference, axis=1).tolist())
        metadata = STRATEGY_METADATA.get(
            strategy,
            {
                "uses_local_cp2k_labels": False,
                "fine_tuned_locally": False,
                "reference_level": str(spec.get("reference_level", "baseline")),
            },
        )
        rows.append(
            {
                "strategy": strategy,
                "model": name,
                "uses_local_cp2k_labels": bool(metadata["uses_local_cp2k_labels"]),
                "fine_tuned_locally": bool(metadata["fine_tuned_locally"]),
                "checkpoint": str(spec.get("model", "")),
                "reference_level": str(spec.get("reference_level", metadata["reference_level"])),
                "energy_mae_vs_cp2k": fmean(energy_errors) if energy_errors else None,
                "force_mae_vs_cp2k": fmean(force_errors) if force_errors else None,
                "cp2k_reference_available": cp2k_reference_available,
                "configuration_count": len(current),
                "runtime_seconds": sum(float(item["runtime_seconds"]) for item in current),
            }
        )
    return rows


def _write_strategy_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "model",
        "uses_local_cp2k_labels",
        "fine_tuned_locally",
        "checkpoint",
        "reference_level",
        "energy_mae_vs_cp2k",
        "force_mae_vs_cp2k",
        "cp2k_reference_available",
        "configuration_count",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_strategy_plot(path: Path, rows: Sequence[dict[str, Any]]) -> Path | None:
    values = [
        row for row in rows
        if row.get("energy_mae_vs_cp2k") is not None
        or row.get("force_mae_vs_cp2k") is not None
    ]
    if not values:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in values]
    energy = [row["energy_mae_vs_cp2k"] or 0.0 for row in values]
    force = [row["force_mae_vs_cp2k"] or 0.0 for row in values]
    x = np.arange(len(values))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(max(6, len(values) * 1.8), 4.2))
    ax1.bar(x - width / 2, energy, width, label="Energy MAE / eV")
    ax1.bar(x + width / 2, force, width, label="Force MAE / eV A^-1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("MAE")
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _infer_framework_atom_count(atoms: Any) -> int:
    adsorbate_count = int(atoms.info.get("adsorbate_count", 0) or 0)
    return len(atoms) - 3 * adsorbate_count


def _is_umof_spec(spec: dict[str, Any]) -> bool:
    return is_umof_specification(spec)


def _require_umof_checkpoint(
    spec: dict[str, Any],
    *,
    download_missing: bool = False,
) -> None:
    try:
        ensure_umof_model_asset(spec, download_missing=download_missing)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"uMOF checkpoint required for backend "
            f"{spec.get('name', 'umof')}: {exc}"
        ) from exc


def _strategy_for_backend(spec: dict[str, Any]) -> str:
    explicit = spec.get("strategy")
    if explicit:
        return str(explicit).strip().casefold()
    if _is_umof_spec(spec):
        return "umof_pretrained"
    model = str(spec.get("model", ""))
    if "active_learning" in model or "finetuned" in model:
        return "cp2k_local_finetune"
    return "baseline"
