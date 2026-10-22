"""Create held-out CP2K parity plots for ZIF-8 active-learning iterations 1-4."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator


DEFAULT_ROOT = PROJECT_ROOT / (
    "outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/"
    "zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning"
)


def _load_test_set(root: Path, iterations: list[int]):
    from ase.io import read

    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    structures = read(
        root / f"iteration_{iterations[0]:02d}/datasets/test.extxyz", index=":"
    )
    identifiers = [str(atoms.info["candidate_id"]) for atoms in structures]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("The fixed CP2K test set must be nonempty with unique IDs.")
    if set(identifiers) != set(state["test_ids"]):
        raise ValueError("Test structures differ from the frozen campaign test IDs.")
    if set(identifiers) & (set(state["training_ids"]) | set(state["validation_ids"])):
        raise ValueError("The fixed test set overlaps training or validation data.")
    for iteration in iterations:
        iteration_set = read(
            root / f"iteration_{iteration:02d}/datasets/test.extxyz", index=":"
        )
        iteration_ids = [str(atoms.info["candidate_id"]) for atoms in iteration_set]
        if iteration_ids != identifiers:
            raise ValueError(
                f"Iteration {iteration} does not use the same ordered test set."
            )
    for atoms in structures:
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            raise ValueError(
                f"Missing CP2K reference energy/forces for {atoms.info['candidate_id']}"
            )
    return structures, identifiers


def _committee_predictions(
    root: Path, structures: list[Any], iteration: int, device: str
) -> list[dict[str, Any]]:
    from ase.calculators.calculator import Calculator

    model_predictions = []
    for member in range(3):
        stem = f"mace_adsorption_committee_{member}"
        model_path = (
            root
            / f"iteration_{iteration:02d}/training/{stem}/models/{stem}_compiled.model"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing iteration {iteration} model: {model_path}")
        calculator = build_ase_calculator(
            {
                "backend": "mace-torch",
                "model": str(model_path),
                "device": device,
                "default_dtype": "float64",
                "dispersion": False,
            }
        )
        predictions = []
        for atoms in structures:
            current = atoms.copy()
            current.calc = calculator
            predictions.append(
                {
                    "energy_ev": float(current.get_potential_energy()),
                    "forces_ev_per_a": np.asarray(current.get_forces(), dtype=float),
                }
            )
        model_predictions.append(predictions)
        del current, calculator
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    return [
        {
            "energy_ev": float(
                np.mean([model_predictions[member][index]["energy_ev"] for member in range(3)])
            ),
            "forces_ev_per_a": np.mean(
                [model_predictions[member][index]["forces_ev_per_a"] for member in range(3)],
                axis=0,
            ),
        }
        for index in range(len(structures))
    ]


def _collect_rows(
    root: Path,
    structures: list[Any],
    identifiers: list[str],
    iterations: list[int],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    energy_rows = []
    force_rows = []
    for iteration in iterations:
        print(f"Evaluating committee for iteration {iteration}...", flush=True)
        predictions = _committee_predictions(root, structures, iteration, device)
        for atoms, candidate_id, prediction in zip(
            structures, identifiers, predictions, strict=True
        ):
            reference_energy = float(atoms.info["REF_energy"])
            reference_forces = np.asarray(atoms.arrays["REF_forces"], dtype=float)
            predicted_forces = prediction["forces_ev_per_a"]
            if predicted_forces.shape != reference_forces.shape:
                raise ValueError(f"Force shape mismatch for {candidate_id}.")
            atom_count = len(atoms)
            energy_rows.append(
                {
                    "iteration": iteration,
                    "candidate_id": candidate_id,
                    "atom_count": atom_count,
                    "reference_energy_per_atom_ev": reference_energy / atom_count,
                    "predicted_energy_per_atom_ev": prediction["energy_ev"] / atom_count,
                    "reference_total_energy_ev": reference_energy,
                    "predicted_total_energy_ev": prediction["energy_ev"],
                    "absolute_total_energy_error_ev": abs(
                        prediction["energy_ev"] - reference_energy
                    ),
                }
            )
            for atom_index in range(atom_count):
                for component_index, component in enumerate("xyz"):
                    force_rows.append(
                        {
                            "iteration": iteration,
                            "candidate_id": candidate_id,
                            "atom_index": atom_index,
                            "component": component,
                            "reference_force_ev_per_a": reference_forces[
                                atom_index, component_index
                            ],
                            "predicted_force_ev_per_a": predicted_forces[
                                atom_index, component_index
                            ],
                        }
                    )
        del predictions
    return energy_rows, force_rows


def _parity_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    padding = max((high - low) * 0.04, 1e-6)
    return low - padding, high + padding


def _write_parity_plots(
    energy_rows: list[dict[str, Any]],
    force_rows: list[dict[str, Any]],
    iterations: list[int],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy_limits = _parity_limits(
        [
            float(row[key])
            for row in energy_rows
            for key in ("reference_energy_per_atom_ev", "predicted_energy_per_atom_ev")
        ]
    )
    force_limits = _parity_limits(
        [
            float(row[key])
            for row in force_rows
            for key in ("reference_force_ev_per_a", "predicted_force_ev_per_a")
        ]
    )
    colors = ["#176b62", "#d26a38", "#486fa8", "#896b2f"]

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for axis, iteration, color in zip(axes.flat, iterations, colors, strict=True):
        rows = [row for row in energy_rows if row["iteration"] == iteration]
        reference = [row["reference_energy_per_atom_ev"] for row in rows]
        predicted = [row["predicted_energy_per_atom_ev"] for row in rows]
        mae_total = np.mean([row["absolute_total_energy_error_ev"] for row in rows])
        axis.scatter(reference, predicted, s=35, alpha=0.78, color=color, edgecolor="white", linewidth=0.4)
        axis.plot(energy_limits, energy_limits, color="#444444", linestyle="--", linewidth=1)
        axis.set_xlim(energy_limits)
        axis.set_ylim(energy_limits)
        axis.set_title(f"Iteration {iteration} | total-energy MAE {mae_total:.3f} eV")
        axis.grid(True, color="#e2e4e1", linewidth=0.7)
    for axis in axes[-1, :]:
        axis.set_xlabel("CP2K reference energy per atom (eV/atom)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Committee-predicted energy per atom (eV/atom)")
    figure.suptitle("Held-out energy parity across active-learning iterations", y=0.99)
    figure.tight_layout()
    figure.savefig(output_dir / "energy_parity_iterations.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for axis, iteration, color in zip(axes.flat, iterations, colors, strict=True):
        rows = [row for row in force_rows if row["iteration"] == iteration]
        reference = np.asarray([row["reference_force_ev_per_a"] for row in rows])
        predicted = np.asarray([row["predicted_force_ev_per_a"] for row in rows])
        mae = float(np.mean(np.abs(predicted - reference)))
        axis.scatter(reference, predicted, s=5, alpha=0.18, color=color, rasterized=True)
        axis.plot(force_limits, force_limits, color="#444444", linestyle="--", linewidth=1)
        axis.set_xlim(force_limits)
        axis.set_ylim(force_limits)
        axis.set_title(f"Iteration {iteration} | component MAE {mae:.3f} eV/Å")
        axis.grid(True, color="#e2e4e1", linewidth=0.7)
    for axis in axes[-1, :]:
        axis.set_xlabel("CP2K reference force component (eV/Å)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Committee-predicted force component (eV/Å)")
    figure.suptitle("Held-out force-component parity across active-learning iterations", y=0.99)
    figure.tight_layout()
    figure.savefig(output_dir / "force_parity_iterations.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    iterations = args.iterations
    if len(iterations) != 4 or len(set(iterations)) != 4:
        parser.error("Provide exactly four distinct iteration numbers.")
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "evaluation" / "parity_iterations"
    output_dir.mkdir(parents=True, exist_ok=True)
    structures, identifiers = _load_test_set(root, iterations)
    energy_rows, force_rows = _collect_rows(
        root, structures, identifiers, iterations, args.device
    )
    energy_path = output_dir / "energy_parity_data.csv"
    force_path = output_dir / "force_parity_data.csv"
    for path, rows in ((energy_path, energy_rows), (force_path, force_rows)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_parity_plots(energy_rows, force_rows, iterations, output_dir)
    metadata = {
        "iterations": iterations,
        "test_configuration_count": len(identifiers),
        "test_ids": identifiers,
        "prediction": "Arithmetic mean of the three committee members per iteration.",
        "energy_plot": "Total energies are divided by atom count for parity display; per-atom MAE is not the campaign's total-energy MAE.",
        "force_plot": "All Cartesian force components from the fixed held-out CP2K test structures.",
        "scope": "Held-out predictions, not training-set fit residuals; not adsorption-isotherm validation.",
    }
    (output_dir / "parity_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved parity plots and data under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())