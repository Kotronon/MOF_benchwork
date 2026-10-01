"""Common adsorption-engine adapters for Module C."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from engines.mlip_mc_runner import run_mlip_mc_gcmc, run_mlip_mc_widom
from pipeline.config import save_benchmark_data


class AdsorptionEngine(ABC):
    """Uniform interface for Widom and GCMC implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable engine identifier."""

    @abstractmethod
    def run_widom(
        self,
        system: dict[str, Any],
        calculator: Any,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Run and normalize a Widom calculation."""

    @abstractmethod
    def run_gcmc(
        self,
        system: dict[str, Any],
        calculator: Any,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Run and normalize a GCMC calculation."""

    def resume(
        self,
        checkpoint: str | Path,
        system: dict[str, Any],
        calculator: Any,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Resume through the engine-specific checkpoint configuration."""
        resumed = dict(settings)
        resumed["checkpoint"] = str(checkpoint)
        method = str(resumed.get("method", "widom")).casefold()
        return (
            self.run_widom(system, calculator, resumed)
            if method == "widom"
            else self.run_gcmc(system, calculator, resumed)
        )


class MLIPMCAdsorptionEngine(AdsorptionEngine):
    @property
    def name(self) -> str:
        return "mlip_mc"

    def run_widom(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        raw = run_mlip_mc_widom(
            calculator,
            system["framework"],
            system["adsorbate"],
            temperature_K=float(settings["temperature_K"]),
            trial_count=int(settings.get("trials", 10_000)),
            seed=int(settings.get("seed", 12345)),
            device=str(settings.get("device", "cpu")),
            output_directory=settings["output_directory"],
            block_size=settings.get("block_size"),
            convergence_checkpoints=settings.get("convergence_checkpoints"),
            vdw_radius_aliases=settings.get("vdw_radius_aliases"),
            write_analysis_csv=bool(settings.get("save_csv", True)),
            write_analysis_plot=bool(settings.get("save_plots", True)),
        )
        return normalize_adsorption_result(raw)

    def run_gcmc(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        raw = run_mlip_mc_gcmc(
            calculator,
            system["framework"],
            system["adsorbate"],
            component=str(settings.get("component", "CO2")),
            temperature_K=float(settings["temperature_K"]),
            pressures_bar=[float(value) for value in settings["pressures_bar"]],
            equilibration_steps=int(settings.get("equilibration_steps", 10_000)),
            production_steps=int(settings.get("production_steps", 20_000)),
            seed=int(settings.get("seed", 12345)),
            device=str(settings.get("device", "cpu")),
            output_directory=settings["output_directory"],
            checkpoint_interval=int(settings.get("checkpoint_interval", 10_000)),
            write_trajectory=bool(settings.get("write_trajectory", False)),
            trajectory_interval=int(settings.get("trajectory_interval", 100)),
            overwrite_checkpoints=bool(settings.get("overwrite_checkpoints", False)),
            allow_ideal_gas_fallback=bool(settings.get("allow_ideal_gas_fallback", False)),
            vdw_radius_aliases=settings.get("vdw_radius_aliases"),
        )
        framework_mass_amu = float(sum(system["framework"].get_masses()))
        normalized = normalize_adsorption_result(raw)
        normalized["framework_mass_amu"] = framework_mass_amu
        for point in normalized.get("pressure_points", []):
            factor = 1000.0 / framework_mass_amu
            point["loading_mol_per_kg"] = (
                float(point["mean_uptake_molecules_per_unit_cell"]) * factor
            )
            point["loading_std_mol_per_kg"] = (
                float(point["uptake_std_molecules_per_unit_cell"]) * factor
            )
            point["acceptance_rates"] = _acceptance_rates(
                point.get("move_statistics", {})
            )
        save_benchmark_data(normalized["output_path"], normalized)
        return normalized


class FLAMESAdsorptionEngine(AdsorptionEngine):
    """Adapter for the pinned optional FLAMES 0.4.8 API."""

    def __init__(self, *, widom_class: Any | None = None, gcmc_class: Any | None = None) -> None:
        self._widom_class = widom_class
        self._gcmc_class = gcmc_class

    @property
    def name(self) -> str:
        return "flames"

    def run_widom(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        widom_class = self._widom_class or _load_flames_widom()
        output = Path(settings["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        radii = _ase_vdw_radii()
        started = perf_counter()
        simulation = widom_class(
            framework_atoms=system["framework"],
            adsorbate_atoms=system["adsorbate"],
            temperature=float(settings["temperature_K"]),
            model=calculator,
            vdw_radii=radii,
            device=str(settings.get("device", "cpu")),
            save_snapshots=bool(settings.get("write_trajectory", False)),
            output_to_file=True,
            output_folder=str(output),
            random_seed=int(settings.get("seed", 12345)),
            cutoff_radius=float(settings.get("cutoff_A", 6.0)),
            automatic_supercell=False,
        )
        if settings.get("checkpoint"):
            simulation.restart()
        simulation.run(int(settings.get("trials", 10_000)))
        simulation.save_results("Widom_Results.json")
        raw_path = _find_result_file(output, ("Widom_Results.json", "widom_results.json"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        normalized = {
            "status": "completed",
            "engine": "flames",
            "method": "widom",
            "temperature_K": float(settings["temperature_K"]),
            "seed": int(settings.get("seed", 12345)),
            "attempts": int(raw.get("total_insertions", settings.get("trials", 10_000))),
            "runtime_seconds": perf_counter() - started,
            "metrics": {
                "henry_coefficient_mol_kg_pa": {
                    "value": _first_number(raw, "henry_coefficient_mol_kg-1_Pa-1", "henry_coefficient_mol_kg_pa"),
                    "uncertainty": _first_number(raw, "henry_coefficient_std_mol_kg-1_Pa-1", required=False),
                },
                "isosteric_heat_zero_loading_kj_mol": {
                    "value": abs(_first_number(raw, "enthalpy_of_adsorption_kJ_mol-1", "isosteric_heat_zero_loading_kj_mol")),
                    "uncertainty": _first_number(raw, "enthalpy_of_adsorption_std_kJ_mol-1", required=False),
                },
            },
            "raw_result_path": str(raw_path),
        }
        path = output / "widom_normalized.json"
        normalized["output_path"] = str(path)
        save_benchmark_data(path, normalized)
        return normalized

    def run_gcmc(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        gcmc_class = self._gcmc_class or _load_flames_gcmc()
        output = Path(settings["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        points = []
        started = perf_counter()
        equilibration = int(settings.get("equilibration_steps", 10_000))
        production = int(settings.get("production_steps", 20_000))
        for index, pressure in enumerate(settings["pressures_bar"]):
            pressure_bar = float(pressure)
            point_output = output / f"{_pressure_label(pressure_bar)}bar"
            point_output.mkdir(parents=True, exist_ok=True)
            point_started = perf_counter()
            simulation = gcmc_class(
                model=calculator,
                framework_atoms=system["framework"],
                adsorbate_atoms=system["adsorbate"],
                temperature=float(settings["temperature_K"]),
                pressure=pressure_bar * 100_000.0,
                device=str(settings.get("device", "cpu")),
                vdw_radii=_ase_vdw_radii(),
                output_to_file=True,
                output_folder=str(point_output),
                random_seed=int(settings.get("seed", 12345)) + index,
                cutoff_radius=float(settings.get("cutoff_A", 6.0)),
                automatic_supercell=False,
                fugacity_coeff=float(settings.get("fugacity_coefficients", {}).get(str(pressure), 1.0)),
            )
            if settings.get("checkpoint"):
                simulation.restart()
            simulation.run(equilibration + production)
            simulation.save_results()
            raw_path = _find_result_file(point_output, ("results*.json", "Results*.json"))
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            uptake = _flames_gcmc_number(
                raw,
                component=str(settings.get("component", "CO2")),
                section="absolute_uptake",
                unit="mol/kg",
                statistic="mean",
            )
            uptake_sd = _flames_gcmc_number(
                raw,
                component=str(settings.get("component", "CO2")),
                section="absolute_uptake",
                unit="mol/kg",
                statistic="sd",
                required=False,
            )
            heat = _nested_number(raw, ("enthalpy", "kJ_mol", "mean"), required=False)
            points.append(
                {
                    "pressure_bar": pressure_bar,
                    "seed": int(settings.get("seed", 12345)) + index,
                    "equilibration_steps": equilibration,
                    "production_steps": production,
                    "loading_mol_per_kg": uptake,
                    "loading_std_mol_per_kg": uptake_sd,
                    "isosteric_heat_kj_mol": None if heat is None else abs(heat),
                    "runtime_seconds": perf_counter() - point_started,
                    "raw_result_path": str(raw_path),
                    "acceptance_rates": _flames_acceptance_rates(raw),
                    "equilibrated": bool(
                        raw.get("equilibration", {}).get(
                            "equilibrated",
                            False,
                        )
                    ),
                }
            )
        normalized = {
            "status": "completed",
            "engine": "flames",
            "method": "gcmc",
            "temperature_K": float(settings["temperature_K"]),
            "seed": int(settings.get("seed", 12345)),
            "runtime_seconds": perf_counter() - started,
            "pressure_points": points,
        }
        path = output / "isotherm_normalized.json"
        normalized["output_path"] = str(path)
        save_benchmark_data(path, normalized)
        return normalized


class LAMMPSClassicalEngine(AdsorptionEngine):
    """Adapter around the existing materialized Module A LAMMPS workflow."""

    @property
    def name(self) -> str:
        return "lammps"

    def run_widom(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "Classical Widom is not implemented by the LAMMPS pipeline; use the "
            "curated CRAFTED Henry/Qst reference or an MLIP engine."
        )

    def run_gcmc(self, system: dict[str, Any], calculator: Any, settings: dict[str, Any]) -> dict[str, Any]:
        from pipeline.runners import run_isotherm

        materialized = settings.get("materialized_plan")
        if not isinstance(materialized, dict):
            raise ValueError("LAMMPSClassicalEngine requires settings.materialized_plan.")
        raw = run_isotherm(materialized, jobs=int(settings.get("jobs", 1)))
        framework_mass_amu = float(sum(system["framework"].get_masses()))
        conversion = 1000.0 / framework_mass_amu
        pressure_points = []
        for point in raw.get("results", []):
            summary = point.get("summary", {})
            uncertainty = summary.get("ci95_half_width_adsorbates")
            pressure_points.append(
                {
                    "pressure_bar": float(point["pressure_bar"]),
                    "loading_mol_per_kg": (
                        float(summary["mean_adsorbates"]) * conversion
                    ),
                    "loading_ci95_half_width_mol_per_kg": (
                        None
                        if uncertainty is None
                        else float(uncertainty) * conversion
                    ),
                    "replicate_count": int(
                        point.get("replicate_count", 1)
                    ),
                    "seeds": point.get("seeds", []),
                    "converged": bool(point.get("converged", False)),
                }
            )
        normalized = {
            "status": raw["status"],
            "engine": "lammps",
            "method": "gcmc",
            "temperature_K": float(settings["temperature_K"]),
            "runtime_seconds": float(raw.get("wall_time_seconds", 0.0)),
            "framework_mass_amu": framework_mass_amu,
            "pressure_points": pressure_points,
            "checkpoint": raw.get("convergence_report"),
            "raw": raw,
        }
        output = Path(settings["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        path = output / "isotherm_normalized.json"
        normalized["output_path"] = str(path)
        save_benchmark_data(path, normalized)
        return normalized


def build_adsorption_engine(name: str) -> AdsorptionEngine:
    normalized = str(name).strip().casefold().replace("-", "_")
    if normalized == "mlip_mc":
        return MLIPMCAdsorptionEngine()
    if normalized == "flames":
        return FLAMESAdsorptionEngine()
    if normalized in {"lammps", "classical_lammps"}:
        return LAMMPSClassicalEngine()
    raise ValueError(f"Unsupported adsorption engine {name!r}.")


def normalize_adsorption_result(result: dict[str, Any]) -> dict[str, Any]:
    """Add the shared metrics schema without removing engine-native fields."""
    normalized = dict(result)
    if str(result.get("method", "")).casefold() == "widom":
        statistics = result.get("corrected_statistics", {})
        normalized["metrics"] = {
            "henry_coefficient_mol_kg_pa": {
                "value": statistics.get("henry_coefficient_mol_kg_pa"),
                "uncertainty": _uncertainty(result, "henry_coefficient_mol_kg_pa"),
            },
            "isosteric_heat_zero_loading_kj_mol": {
                "value": statistics.get("isosteric_heat_zero_loading_kj_mol"),
                "uncertainty": _uncertainty(result, "isosteric_heat_zero_loading_kj_mol"),
            },
        }
    return normalized


def compare_engine_results(
    primary: dict[str, Any],
    cross_check: dict[str, Any],
    *,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare normalized results using confidence overlap or relative error."""
    if primary.get("method") != cross_check.get("method"):
        raise ValueError("Cross-check results must use the same method.")
    comparisons = {}
    if primary.get("method") == "widom":
        keys = (
            "henry_coefficient_mol_kg_pa",
            "isosteric_heat_zero_loading_kj_mol",
        )
        for key in keys:
            left = primary["metrics"][key]
            right = cross_check["metrics"][key]
            comparisons[key] = _compare_metric(left, right, relative_tolerance)
    else:
        left_points = {float(row["pressure_bar"]): row for row in primary["pressure_points"]}
        right_points = {float(row["pressure_bar"]): row for row in cross_check["pressure_points"]}
        if set(left_points) != set(right_points):
            raise ValueError("GCMC cross-check pressure grids differ.")
        for pressure in sorted(left_points):
            left_value = _loading_value(left_points[pressure])
            right_value = _loading_value(right_points[pressure])
            comparisons[f"loading_{pressure:g}bar"] = _compare_metric(
                {"value": left_value, "uncertainty": _loading_uncertainty(left_points[pressure])},
                {"value": right_value, "uncertainty": _loading_uncertainty(right_points[pressure])},
                relative_tolerance,
            )
    passed = bool(comparisons) and all(item["passed"] for item in comparisons.values())
    return {
        "status": "passed" if passed else "failed",
        "primary_engine": primary.get("engine"),
        "cross_check_engine": cross_check.get("engine"),
        "relative_tolerance": relative_tolerance,
        "comparisons": comparisons,
    }


def _load_flames_widom() -> Any:
    try:
        from flames.widom import Widom
    except ImportError as exc:
        raise ImportError(
            "FLAMES is not installed. Run --setup-module-c --install-missing."
        ) from exc
    return Widom


def _load_flames_gcmc() -> Any:
    try:
        from flames.gcmc import GCMC
    except ImportError as exc:
        raise ImportError(
            "FLAMES is not installed. Run --setup-module-c --install-missing."
        ) from exc
    return GCMC


def _ase_vdw_radii() -> Any:
    try:
        from ase.data import vdw_radii
    except ImportError as exc:
        raise ImportError("ASE is required for adsorption engines.") from exc
    return vdw_radii


def _find_result_file(directory: Path, candidates: tuple[str, ...]) -> Path:
    for pattern in candidates:
        matches = sorted(directory.rglob(pattern))
        if matches:
            return matches[-1]
    raise RuntimeError(f"No FLAMES result JSON was found below {directory}.")


def _first_number(data: dict[str, Any], *keys: str, required: bool = True) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    if required:
        raise ValueError(f"FLAMES result lacks all expected fields: {', '.join(keys)}")
    return None


def _nested_number(data: dict[str, Any], path: tuple[str, ...], *, required: bool = True) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            if required:
                raise ValueError(f"FLAMES result lacks {'.'.join(path)}.")
            return None
        value = value[key]
    return float(value)


def _flames_gcmc_number(
    data: dict[str, Any],
    *,
    component: str,
    section: str,
    unit: str,
    statistic: str,
    required: bool = True,
) -> float | None:
    candidates = [
        (section, unit, statistic),
        (
            "uptake" if section == "absolute_uptake" else section,
            unit,
            statistic,
        ),
        ("results", component, section, unit, statistic),
    ]
    for path in candidates:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return float(value)
    if required:
        raise ValueError(
            "FLAMES result lacks a supported absolute-uptake schema."
        )
    return None


def _uncertainty(result: dict[str, Any], metric: str) -> float | None:
    uncertainty = result.get("uncertainty") or {}
    metrics = uncertainty.get("metrics", {}) if isinstance(uncertainty, dict) else {}
    value = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
    half_width = value.get("ci95_half_width") if isinstance(value, dict) else None
    return None if half_width is None else float(half_width)


def _compare_metric(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> dict[str, Any]:
    left_value = float(left["value"])
    right_value = float(right["value"])
    denominator = max(abs(left_value), abs(right_value), 1.0e-30)
    relative_difference = abs(left_value - right_value) / denominator
    left_u = left.get("uncertainty")
    right_u = right.get("uncertainty")
    intervals_overlap = None
    if left_u is not None and right_u is not None:
        intervals_overlap = not (
            left_value + float(left_u) < right_value - float(right_u)
            or right_value + float(right_u) < left_value - float(left_u)
        )
    return {
        "primary_value": left_value,
        "cross_check_value": right_value,
        "relative_difference": relative_difference,
        "confidence_intervals_overlap": intervals_overlap,
        "passed": bool(intervals_overlap) or relative_difference <= tolerance,
    }


def _loading_value(point: dict[str, Any]) -> float:
    if point.get("loading_mol_per_kg") is not None:
        return float(point["loading_mol_per_kg"])
    raise ValueError("GCMC point has no normalized loading value.")


def _loading_uncertainty(point: dict[str, Any]) -> float | None:
    for key in (
        "loading_std_mol_per_kg",
        "loading_ci95_half_width_mol_per_kg",
    ):
        if point.get(key) is not None:
            return float(point[key])
    return None


def _acceptance_rates(statistics: Any) -> dict[str, float]:
    if not isinstance(statistics, dict):
        return {}
    rates = {}
    for move, values in statistics.items():
        if isinstance(values, dict):
            accepted = values.get("accepted")
            attempted = values.get("attempted", values.get("proposed"))
            if accepted is not None and attempted:
                rates[str(move)] = float(accepted) / float(attempted)
    return rates


def _flames_acceptance_rates(raw: dict[str, Any]) -> dict[str, float]:
    statistics = raw.get("acceptance_rates", raw.get("moves", {}))
    if not isinstance(statistics, dict):
        return {}
    rates = {}
    for key, value in statistics.items():
        if isinstance(value, (int, float)):
            rates[str(key)] = float(value)
    return rates


def _pressure_label(pressure_bar: float) -> str:
    return f"{pressure_bar:g}".replace(".", "p")
