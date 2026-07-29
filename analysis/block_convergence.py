from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any

from analysis.plotting import legend_outside_right, repeated_seed_labels_needed
from converter.cif_to_lammps_data import load_framework_structure
from parsers.lammps_log_parser import get_log, parse_lammps_log


BLOCK_FIELDS = [
    "pressure_bar",
    "seed",
    "log_file",
    "block_index",
    "start_step",
    "end_step",
    "complete",
    "sample_count",
    "mean_adsorbates",
    "standard_deviation_adsorbates",
    "standard_error_adsorbates",
    "cumulative_mean_adsorbates",
]


def analyze_block_convergence(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    equilibration_step: int | None = None,
    block_size: int = 50_000,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Analyze block and cumulative adsorption means from completed or live logs."""
    run_path = Path(run_dir)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive.")

    parameters = _run_parameters(run_path)
    equilibration = (
        int(equilibration_step)
        if equilibration_step is not None
        else int(parameters["equilibration_steps"])
    )
    production_steps = int(parameters["production_steps"])
    planned_end_step = equilibration + production_steps
    framework_atoms = int(parameters["framework_atoms"])
    adsorbate_atoms = int(parameters["adsorbate_atoms_per_molecule"])

    log_files = sorted((run_path / "logs").glob("gcmc_*bar*.log"))
    if not log_files:
        raise FileNotFoundError(f"No pressure-point logs found in {run_path / 'logs'}.")

    block_rows: list[dict[str, Any]] = []
    run_reports = []
    for log_file in log_files:
        input_file = run_path / "inputs" / f"{log_file.stem}.in"
        metadata = _input_metadata(input_file)
        report, rows = _analyze_log(
            log_file,
            pressure_bar=metadata["pressure_bar"],
            seed=metadata["seed"],
            equilibration_step=equilibration,
            planned_end_step=planned_end_step,
            block_size=block_size,
            framework_atoms=framework_atoms,
            adsorbate_atoms_per_molecule=adsorbate_atoms,
            relative_tolerance=relative_tolerance,
        )
        run_reports.append(report)
        block_rows.extend(rows)

    target = (
        Path(output_dir)
        if output_dir is not None
        else run_path / "analysis" / "block_convergence"
    )
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "block_convergence.csv"
    json_path = target / "block_convergence_summary.json"
    plot_path = target / "block_convergence.png"
    _write_csv(csv_path, block_rows)
    plot_result = _write_plot(plot_path, block_rows, equilibration)

    completed = [report for report in run_reports if report["run_complete"]]
    judged = [report for report in completed if report["converged"] is not None]
    summary = {
        "run_directory": str(run_path),
        "equilibration_step": equilibration,
        "production_steps": production_steps,
        "planned_end_step": planned_end_step,
        "block_size": block_size,
        "relative_tolerance": relative_tolerance,
        "framework_atoms": framework_atoms,
        "adsorbate_atoms_per_molecule": adsorbate_atoms,
        "log_count": len(run_reports),
        "completed_log_count": len(completed),
        "all_completed_runs_converged": (
            bool(judged) and all(report["converged"] for report in judged)
        ),
        "provisional": len(completed) != len(run_reports),
        "runs": run_reports,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_result) if plot_result else None,
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_parameters(run_dir: Path) -> dict[str, int]:
    prepare_path = run_dir / "prepare_summary.json"
    if not prepare_path.exists():
        raise FileNotFoundError(f"Missing prepare summary: {prepare_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    production = _find_key(prepare, "production_steps")
    equilibration = _find_key(prepare, "equilibration_steps")
    adsorbate_atoms = _find_key(prepare, "adsorbate_atoms_per_molecule")
    framework_atoms = _framework_atom_count_for_run(run_dir, prepare)
    if production is None or equilibration is None or adsorbate_atoms is None:
        raise ValueError("prepare_summary.json lacks convergence parameters.")
    return {
        "production_steps": int(production),
        "equilibration_steps": int(equilibration),
        "framework_atoms": framework_atoms,
        "adsorbate_atoms_per_molecule": int(adsorbate_atoms),
    }


def _framework_atom_count_for_run(run_dir: Path, prepare: dict[str, Any]) -> int:
    data_files = sorted((run_dir / "data").glob("*.data"))
    if len(data_files) == 1:
        return _framework_atom_count(data_files[0])
    if len(data_files) > 1:
        raise ValueError(f"Expected at most one data file in {run_dir / 'data'}.")

    framework_plan = (
        prepare.get("prepare_plan", {})
        .get("planned_files", {})
        .get("framework_data", {})
    )
    input_cif = framework_plan.get("input_cif")
    if not input_cif:
        raise ValueError(
            f"No data file found in {run_dir / 'data'} and no CIF conversion plan "
            f"found in {run_dir / 'prepare_summary.json'}."
        )
    structure = load_framework_structure(
        input_cif,
        cell_representation=str(framework_plan.get("cell_representation", "source")),
        unit_cells=framework_plan.get("unit_cells", [1, 1, 1]),
        cutoff_A=framework_plan.get("cutoff_A"),
        minimum_image_policy=str(framework_plan.get("minimum_image_policy", "error")),
    )
    return structure.atom_count


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _framework_atom_count(data_path: Path) -> int:
    for line in data_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == "atoms":
            return int(parts[0])
    raise ValueError(f"Could not read atom count from {data_path}.")


def _input_metadata(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input corresponding to log: {input_path}")
    text = input_path.read_text(encoding="utf-8")
    pressure_match = re.search(
        r"^# pressure_bar metadata:\s*([0-9.eE+-]+)", text, re.MULTILINE
    )
    seed_match = re.search(
        r"^fix\s+\S+\s+\S+\s+gcmc(?:\s+\S+){4}\s+(\d+)",
        text,
        re.MULTILINE,
    )
    if pressure_match is None:
        raise ValueError(f"Missing pressure metadata in {input_path}.")
    return {
        "pressure_bar": float(pressure_match.group(1)),
        "seed": int(seed_match.group(1)) if seed_match else None,
    }


def _analyze_log(
    log_file: Path,
    *,
    pressure_bar: float,
    seed: int | None,
    equilibration_step: int,
    planned_end_step: int,
    block_size: int,
    framework_atoms: int,
    adsorbate_atoms_per_molecule: int,
    relative_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed = parse_lammps_log(get_log(log_file))
    thermo_rows = [
        row
        for row in parsed["rows"]
        if "Step" in row
        and "Atoms" in row
        and equilibration_step <= int(row["Step"]) <= planned_end_step
    ]
    all_steps = [
        int(row["Step"])
        for row in parsed["rows"]
        if "Step" in row
    ]
    latest_step = max(all_steps) if all_steps else None
    run_complete = latest_step is not None and latest_step >= planned_end_step
    block_count = max(
        1, math.ceil((planned_end_step - equilibration_step) / block_size)
    )

    grouped: dict[int, list[float]] = {}
    for row in thermo_rows:
        step = int(row["Step"])
        index = min((step - equilibration_step) // block_size, block_count - 1)
        count = (
            float(row["Atoms"]) - framework_atoms
        ) / adsorbate_atoms_per_molecule
        grouped.setdefault(index, []).append(count)

    output_rows = []
    cumulative_values: list[float] = []
    for index in sorted(grouped):
        values = grouped[index]
        cumulative_values.extend(values)
        start_step = equilibration_step + index * block_size
        end_step = min(start_step + block_size, planned_end_step)
        block_complete = latest_step is not None and latest_step >= end_step
        deviation = stdev(values) if len(values) >= 2 else None
        output_rows.append(
            {
                "pressure_bar": pressure_bar,
                "seed": seed,
                "log_file": str(log_file),
                "block_index": index + 1,
                "start_step": start_step,
                "end_step": end_step,
                "complete": block_complete,
                "sample_count": len(values),
                "mean_adsorbates": mean(values),
                "standard_deviation_adsorbates": deviation,
                "standard_error_adsorbates": (
                    deviation / math.sqrt(len(values))
                    if deviation is not None
                    else None
                ),
                "cumulative_mean_adsorbates": mean(cumulative_values),
            }
        )

    complete_blocks = [row for row in output_rows if row["complete"]]
    last_block_change = None
    if len(complete_blocks) >= 2:
        last_block_change = _relative_change(
            complete_blocks[-1]["mean_adsorbates"],
            complete_blocks[-2]["mean_adsorbates"],
        )
    half_full_change = None
    if run_complete and output_rows:
        halfway = equilibration_step + (
            planned_end_step - equilibration_step
        ) / 2
        halfway_values = [
            (
                float(row["Atoms"]) - framework_atoms
            ) / adsorbate_atoms_per_molecule
            for row in thermo_rows
            if int(row["Step"]) <= halfway
        ]
        all_values = [
            (
                float(row["Atoms"]) - framework_atoms
            ) / adsorbate_atoms_per_molecule
            for row in thermo_rows
        ]
        if halfway_values and all_values:
            half_full_change = _relative_change(
                mean(all_values), mean(halfway_values)
            )

    converged = None
    if run_complete and last_block_change is not None and half_full_change is not None:
        converged = (
            last_block_change <= relative_tolerance
            and half_full_change <= relative_tolerance
        )
    return (
        {
            "pressure_bar": pressure_bar,
            "seed": seed,
            "log_file": str(log_file),
            "latest_step": latest_step,
            "progress_fraction": (
                min(latest_step / planned_end_step, 1.0)
                if latest_step is not None and planned_end_step > 0
                else 0.0
            ),
            "run_complete": run_complete,
            "production_sample_count": len(thermo_rows),
            "complete_block_count": len(complete_blocks),
            "last_two_block_relative_change": last_block_change,
            "half_to_full_cumulative_relative_change": half_full_change,
            "converged": converged,
        },
        output_rows,
    )


def _relative_change(value: float, reference: float) -> float:
    scale = abs(value)
    if scale == 0:
        return 0.0 if reference == 0 else math.inf
    return abs(value - reference) / scale


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLOCK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    equilibration_step: int,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    grouped: dict[tuple[float, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["pressure_bar"], row["seed"]), []).append(row)
    include_seed = repeated_seed_labels_needed(grouped)
    for (pressure, seed), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["block_index"])
        steps = [row["end_step"] for row in values]
        label = f"{pressure:g} bar"
        if include_seed and seed is not None:
            label = f"{label}, seed {seed}"
        axes[0].plot(
            steps,
            [row["mean_adsorbates"] for row in values],
            "o-",
            label=label,
        )
        axes[1].plot(
            steps,
            [row["cumulative_mean_adsorbates"] for row in values],
            "o-",
            label=label,
        )
    axes[0].set_ylabel("Block mean N_CO2")
    axes[1].set_ylabel("Cumulative mean N_CO2")
    axes[1].set_xlabel("LAMMPS step")
    axes[0].axvline(equilibration_step, color="black", linestyle="--", alpha=0.5)
    axes[1].axvline(equilibration_step, color="black", linestyle="--", alpha=0.5)
    for axis in axes:
        axis.grid(alpha=0.3)
        legend_outside_right(axis, fontsize="small")
    figure.tight_layout(rect=(0, 0, 0.78, 1))
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze block convergence in completed or live GCMC logs."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--equilibration-step", type=int)
    parser.add_argument("--block-size", type=int, default=50_000)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = analyze_block_convergence(
        args.run_dir,
        args.output_dir,
        equilibration_step=args.equilibration_step,
        block_size=args.block_size,
        relative_tolerance=args.relative_tolerance,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
