"""Offline planning/preparation for the NbOFFIVE-1-Ni/TIP4P benchmark.

Production execution remains disabled until the coupled engine adapter and
paper-specific assets are validated. Preparation never launches a simulation.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

from modules.base import BenchmarkModule

MODULE_ID = "D"
MODULE_NAME = "Module D"
TASK = "framework_flexibility"
DESCRIPTION = "Framework flexibility via MLIP-MD/GCMC sorption relaxation."
REFERENCE_DIR = Path("data/module_d/mofaff_reference")
CYCLE_COLUMNS = ("case_id", "cycle", "uptake_mol_kg", "framework_rmsd_A",
                 "cell_volume_A3", "potential_energy_eV", "pore_volume_A3",
                 "linker_rotation_deg", "runtime_s")


def metadata() -> dict[str, object]:
    return {"module_id": MODULE_ID, "module_name": MODULE_NAME, "task": TASK,
            "description": DESCRIPTION, "status": "planning_and_analysis"}


def finite(value: Any, name: str, minimum: float = 0, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number) or number < minimum or (positive and number == minimum):
        raise ValueError(f"{name} must be finite and {'>' if positive else '>='} {minimum}.")
    return number


def integer(value: Any, name: str, minimum: int = 1) -> int:
    number = finite(value, name, minimum)
    if number != int(number):
        raise ValueError(f"{name} must be an integer.")
    return int(number)


def asset(path: Any, role: str) -> dict[str, Any]:
    result = {"role": role, "path": str(path) if path else None, "status": "missing"}
    if path:
        source = Path(str(path)).expanduser()
        if source.is_file() and source.stat().st_size:
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result.update(status="present", sha256=digest.hexdigest())
    return result


def build_flexibility_plan(config: dict[str, Any], module: dict[str, str]) -> dict[str, Any]:
    """Plan a paper campaign without falling back to unrelated CRAFTED assets."""
    config = deepcopy(config)
    protocol = config["benchmark"].get("flexibility")
    if not isinstance(protocol, dict):
        raise ValueError("Module D requires benchmark.flexibility; see MODULE_D_FLEXIBILITY.md.")
    if config["material"]["name"].casefold() not in {"nboffive-1-ni", "kaust-7"}:
        raise ValueError("The initial Module D protocol supports NbOFFIVE-1-Ni / KAUST-7.")
    if config["adsorbates"]["components"] != ["H2O"]:
        raise ValueError("The NbOFFIVE-1-Ni protocol requires single-component H2O.")
    conditions = config["conditions"]
    if finite(conditions["temperature_K"], "temperature_K", positive=True) != 300.0:
        raise ValueError("The paper TIP4P saturation pressure is defined at 300 K only.")
    rh_values = conditions.get("relative_humidity")
    if not isinstance(rh_values, list) or not rh_values:
        raise ValueError("relative_humidity must be a non-empty list of fractions in (0, 1].")
    rh_values = [finite(value, "relative_humidity", positive=True) for value in rh_values]
    if any(value > 1 for value in rh_values) or len(set(rh_values)) != len(rh_values):
        raise ValueError("relative_humidity must contain unique fractions in (0, 1], not percent.")
    reservoir = protocol.get("reservoir", {})
    if reservoir.get("water_model") != "TIP4P":
        raise ValueError("The paper profile requires TIP4P, not another water model.")
    p0 = finite(reservoir.get("model_saturation_pressure_bar"), "model_saturation_pressure_bar", positive=True)
    if not math.isclose(p0, 0.05197, rel_tol=0, abs_tol=1e-10):
        raise ValueError("The paper uses TIP4P p0 = 5197 Pa = 0.05197 bar at 300 K.")
    if not reservoir.get("source"):
        raise ValueError("reservoir.source must document the saturation pressure provenance.")
    fugacities = reservoir.get("fugacities_bar")
    if fugacities is not None:
        if not isinstance(fugacities, list) or len(fugacities) != len(rh_values):
            raise ValueError("fugacities_bar must contain one value per RH point.")
        fugacities = [finite(v, "fugacity_bar", positive=True) for v in fugacities]
        if not reservoir.get("fugacity_source"):
            raise ValueError("Explicit fugacities require reservoir.fugacity_source.")
    cycles = integer(protocol.get("sr_cycles", 100), "sr_cycles")
    discard = integer(protocol.get("discard_cycles", 0), "discard_cycles", 0)
    window = integer(protocol.get("convergence_window", 10), "convergence_window", 2)
    if cycles - discard < 2 * window:
        raise ValueError("sr_cycles - discard_cycles must allow two convergence windows.")
    protocol.update(sr_cycles=cycles, discard_cycles=discard, convergence_window=window)
    for key, default in (("uptake_drift_tolerance_mol_kg", 0.05),
                         ("rmsd_drift_tolerance_A", 0.02),
                         ("volume_relative_drift_tolerance", 0.01)):
        protocol[key] = finite(protocol.get(key, default), key, positive=True)
    if protocol.get("md", {}).get("ensemble") != "NVT":
        raise ValueError("The first Module D implementation prepares the NVT branch only.")
    md = protocol["md"]
    md["steps"] = integer(md.get("steps", 1000), "md.steps")
    md["timestep_fs"] = finite(md.get("timestep_fs", 1), "md.timestep_fs", positive=True)
    md["thermostat_damping_ps"] = finite(md.get("thermostat_damping_ps", 0.04), "thermostat_damping_ps", positive=True)
    gcmc = protocol.setdefault("gcmc", {})
    for key, default in (("equilibration_steps", 200000), ("production_steps", 500000)):
        gcmc[key] = integer(gcmc.get(key, default), f"gcmc.{key}")
    assets = [asset(protocol.get("assets", {}).get(role), role) for role in (
        "initial_structure", "rigid_framework_cif", "charges", "mlip_model",
        "mlip_validation", "experimental_reference", "gcmc_parameters_validation")]
    blockers = [f"Missing asset: {a['role']}" for a in assets if a["status"] == "missing"]
    if fugacities is None:
        blockers.append("Missing TMMC pressure-to-fugacity table (paper SI Table S2); p is not silently used as f.")
    blockers.extend([
        "The coupled MD/GCMC engine adapter and TIP4P handoff have not been validated.",
        "The upstream example disables global LJ tail corrections, whereas the paper applies water-water tails; reconcile with SI before execution.",
        "Presence of an MLIP or validation file is not proof of validation for hydration, deformation or stress.",
    ])
    points = [{"point_id": f"rh{i:03d}", "relative_humidity": rh,
               "pressure_bar": rh * p0,
               "fugacity_bar": fugacities[i] if fugacities else None,
               "temperature_K": 300.0} for i, rh in enumerate(rh_values)]
    conditions["pressures_bar"] = [point["pressure_bar"] for point in points]
    output = Path(config["output"].get("directory") or "outputs/module_D_nboffive_water")
    run_id = config["output"].get("run_id")
    if run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(run_id)):
            raise ValueError("run_id must contain only letters, digits, underscores or hyphens.")
        output = output / "runs" / str(run_id)
    cases = []
    for variant in ("rigid", "sr_nvt"):
        for point in points:
            for seed in config["simulation"]["seeds"]:
                cases.append({"case_id": f"{variant}__{point['point_id']}__seed{seed}",
                              "variant": variant, **point, "seed": seed,
                              "expected_cycles": cycles if variant == "sr_nvt" else 1})
    applicability = {
        "status": "planning_only", "current_module_capability": "planning_and_analysis",
        "can_attempt_simulation": False, "requires_user_confirmation": False,
        "recommended_module": "D", "assets": assets, "blockers": blockers,
        "validation_status": "not_validated",
        "reason": "Offline preparation and imported-cycle analysis are implemented; simulation execution is unavailable.",
    }
    return {"status": "planned", "module": module, "material": config["material"],
            "adsorbates": config["adsorbates"], "conditions": conditions,
            "simulation": config["simulation"],
            "benchmark": {**config["benchmark"], "flexibility": protocol, "applicability": applicability},
            "outputs": {**config["output"], "directory": str(output)},
            "state_points": points, "cases": cases}


def prepare_flexibility(plan: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "module_d_preparation", "module": plan["module"],
            "working_directory": plan["outputs"]["directory"], "run_plan": deepcopy(plan)}


def materialize_flexibility(prepared: dict[str, Any]) -> dict[str, Any]:
    """Create an immutable planning bundle, preserving existing calculation outputs."""
    plan = prepared["run_plan"]
    root = Path(prepared["working_directory"])
    manifest = root / "module_d_plan.json"
    if root.exists():
        if plan["outputs"].get("resume") and manifest.is_file():
            previous = json.loads(manifest.read_text())
            comparable = deepcopy(plan)
            for flag in ("resume", "overwrite"):
                comparable["outputs"].pop(flag, None)
                previous["outputs"].pop(flag, None)
            if comparable == previous:
                return {"status": "reused_plan", "working_directory": str(root)}
        raise FileExistsError(f"Module D preserves existing directories: {root}. Choose a new run_id or resume an identical plan.")
    root.mkdir(parents=True)
    manifest.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    with (root / "cases.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan["cases"][0]))
        writer.writeheader()
        writer.writerows(plan["cases"])
    with (root / "cycle_results.template.csv").open("x", newline="") as handle:
        csv.writer(handle).writerow(CYCLE_COLUMNS)
    if REFERENCE_DIR.is_dir():
        shutil.copytree(REFERENCE_DIR, root / "upstream_reference")
    (root / "README.txt").write_text(
        "Planning bundle only; no simulations have been run.\n"
        "Inspect benchmark.applicability in module_d_plan.json for missing assets.\n"
        "upstream_reference contains original inputs for inspection, not runnable prepared cases.\n"
        "Import measured summaries as cycle_results.csv using the template columns.\n"
        "Use case IDs from cases.csv; cycles are 1-based and contiguous.\n"
        "Uptake: mol/kg of dry framework. Structural metrics: framework atoms only.\n")
    return {"status": "prepared_planning_bundle", "working_directory": str(root),
            "simulation_started": False, "applicability": plan["benchmark"]["applicability"]}


class FlexibilityModule(BenchmarkModule):
    def assess(self, config: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
        return build_flexibility_plan(config, self.metadata())["benchmark"]["applicability"]

    def prepare(self, run_plan: dict[str, Any]) -> dict[str, Any]:
        return prepare_flexibility(run_plan)

    def materialize(self, prepare_plan: dict[str, Any]) -> dict[str, Any]:
        return materialize_flexibility(prepare_plan)

    def run_test(self, materialized_plan: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Module D simulation execution is unavailable; use --dry-run, --prepare or --evaluate.")

    def run(self, materialized_plan: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.run_test(materialized_plan)

    def evaluate(self, run_directory: str | Path, **kwargs: Any) -> dict[str, Any]:
        from analysis.module_d_flexibility import evaluate_flexibility
        return evaluate_flexibility(run_directory)
