from __future__ import annotations

import argparse
import json

from pipeline.config import (
    DEFAULT_PRESSURES_BAR,
    data_to_dict,
    load_benchmark_data,
    normalize_config,
    save_benchmark_data,
)
from pipeline.eos import fugacity_coeff_coolprop
from pipeline.evaluate import evaluate_all_references, evaluate_run_directory
from pipeline.lammps_inputs import gcmc_input_builder
from pipeline.materialization import materialize_benchmark
from pipeline.nist_isodb_parser import find_nist_isotherm_candidates, load_nist_isotherm
from pipeline.planning import build_run_plan, resolve_benchmark, select_module
from pipeline.prepare import prepare, prepare_benchmark, timestamp_run_id
from pipeline.registry_generation import write_generated_registries
from pipeline.runners import run_benchmark, run_isotherm
from pipeline.variants import build_variant_plans, run_variant_benchmark
from modules.module_c_mlips.workflow import run_potential_benchmark
from modules.module_c_mlips.dependencies import ensure_mlip_mc_dependencies
from modules.module_c_mlips.mlip_mc_workflow import run_mlip_mc_benchmark
from modules.module_c_mlips.potential_backends.calculators import build_ase_calculator

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a MOF benchmark run from benchmark.json.")
    parser.add_argument("config", nargs="?", default="benchmark.json", help="Path to benchmark JSON input.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without running LAMMPS.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the benchmark environment.")
    parser.add_argument("--run-test", action="store_true", help="Run a test GCMC simulation after preparation.")
    parser.add_argument("--run-isotherm", action="store_true", help="Run the full isotherm after preparation.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate an existing run directory without running LAMMPS.")
    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate an existing run against all references.")
    parser.add_argument("--run-id", help="Write outputs to outputs/<module>/runs/<run-id>.")
    parser.add_argument("--kspace-style", choices=["pppm", "ewald"], help="Override simulation.kspace_style.")
    parser.add_argument("--kspace-accuracy", type=float, help="Override simulation.kspace_accuracy.")
    parser.add_argument(
        "--potentials",
        "--run-potentials",
        dest="run_potentials",
        action="store_true",
        help="Run the Module C potential comparison.",
    )
    parser.add_argument(
        "--run-mlip-mc",
        action="store_true",
        help="Run the configured Module C MLIP-MC Widom or GCMC workflow.",
    )
    parser.add_argument(
        "--setup-mlip-mc",
        action="store_true",
        help="Install and verify the configured MLIP-MC backend, then exit.",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install missing pinned MLIP-MC dependencies in the active Python environment.",
    )
    parser.add_argument(
        "--run-variants",
        "--run-variant",
        dest="run_variants",
        action="store_true",
        help="Run benchmark variants instead of the base plan.",
    )
    parser.add_argument(
        "--registry-output-dir",
        default="data/generated",
        help="Directory for continuously generated capability registries.",
    )
    parser.add_argument(
        "--skip-registry-update",
        action="store_true",
        help="Do not refresh generated capability registries before planning.",
    )
    parser.add_argument("--new-run", action="store_true", help="Write outputs to a timestamped run directory.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the target working directory already exists.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed pressure-point results in an existing run directory.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of pressure-point LAMMPS jobs to run in parallel.")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Run cases marked outside the current validated module capability without an interactive prompt.",
    )
    args = parser.parse_args(argv)

    config = load_benchmark_data(args.config)
    if args.new_run:
        config.setdefault("output", {})["run_id"] = timestamp_run_id()
    if args.run_id:
        config.setdefault("output", {})["run_id"] = args.run_id
    if args.kspace_style:
        config.setdefault("simulation", {})["kspace_style"] = args.kspace_style
    if args.kspace_accuracy is not None:
        config.setdefault("simulation", {})["kspace_accuracy"] = args.kspace_accuracy
    if args.no_overwrite:
        config.setdefault("output", {})["overwrite"] = False
    if args.resume:
        config.setdefault("output", {})["resume"] = True
    if not args.skip_registry_update:
        write_generated_registries(args.registry_output_dir)
    run_plan = build_run_plan(config)
    if args.setup_mlip_mc:
        backend = _configured_mlip_mc_backend(run_plan)
        status = ensure_mlip_mc_dependencies(
            backend,
            install_missing=True,
        )
        model = _configured_mlip_mc_model(run_plan)
        build_ase_calculator(model)
        print(
            json.dumps(
                {
                    "status": "ready",
                    **status.to_dict(),
                    "model": model.get("name", model.get("model")),
                    "model_ready": True,
                },
                indent=2,
            )
        )
        return 0
    if args.dry_run:
        if args.run_variants:
            variant_plans = build_variant_plans(config)
            result = {
                "status": "planned",
                "variant_count": len(variant_plans),
                "variants": variant_plans,
            }
        else:
            result = run_plan
        print(json.dumps(result, indent=2))
        return 0
    if args.prepare:
        prepare_plan = prepare_benchmark(run_plan)
        result = materialize_benchmark(prepare_plan)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_test:
        _confirm_supported_capability(run_plan, allow_unsupported=args.allow_unsupported)
        prepare_plan = prepare_benchmark(run_plan)
        materialized_plan = materialize_benchmark(prepare_plan)
        result = run_benchmark(materialized_plan)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_isotherm:
        _confirm_supported_capability(run_plan, allow_unsupported=args.allow_unsupported)
        prepare_plan = prepare_benchmark(run_plan)
        materialized_plan = materialize_benchmark(prepare_plan)
        result = run_isotherm(materialized_plan, jobs=args.jobs)
        print(json.dumps(result, indent=2))
        return 0
    if args.evaluate:
        prepare_plan = prepare_benchmark(run_plan)
        result = evaluate_run_directory(
            prepare_plan["working_directory"],
            evaluation_config=run_plan.get("evaluation"),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.evaluate_all:
        prepare_plan = prepare_benchmark(run_plan)
        result = evaluate_all_references(
            prepare_plan["working_directory"],
            references=run_plan["resources"]["references"],
            evaluation_config=run_plan.get("evaluation"),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.run_variants:
        _confirm_supported_capability(run_plan, allow_unsupported=args.allow_unsupported)
        result = run_variant_benchmark(config, jobs=args.jobs)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_potentials:
        _confirm_supported_capability(
            run_plan,
            allow_unsupported=args.allow_unsupported,
        )
        if run_plan["module"]["id"] != "C":
            raise ValueError(
                "--potentials requires benchmark.task='potential_benchmark'."
            )
        workflow = str(
            run_plan["benchmark"].get("potential_benchmark", {}).get(
                "workflow",
                "static",
            )
        ).strip().casefold()
        if workflow != "static":
            raise ValueError(
                "--potentials runs only the static comparison. Use "
                "--run-mlip-mc for an MLIP-MC workflow."
            )
        result = run_potential_benchmark(run_plan)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "working_directory": result["working_directory"],
                    "output_path": result["output_path"],
                    "baseline_backend": result["baseline_backend"],
                    "backends": result["backends"],
                    "configuration_count": result["configuration_count"],
                    "runtime_seconds": result["runtime_seconds"],
                    "shared_framework_cache": result[
                        "shared_framework_cache"
                    ],
                    "analysis": result["analysis"],
                },
                indent=2,
            )
        )
        return 0
    if args.run_mlip_mc:
        _confirm_supported_capability(
            run_plan,
            allow_unsupported=args.allow_unsupported,
        )
        result = run_mlip_mc_benchmark(
            run_plan,
            install_missing=args.install_missing,
        )
        print(json.dumps(result, indent=2))
        return 0
    print(json.dumps(run_plan, indent=2))
    return 0


def _confirm_supported_capability(run_plan: dict, *, allow_unsupported: bool = False) -> None:
    applicability = run_plan.get("benchmark", {}).get("applicability", {})
    if applicability.get("can_attempt_simulation") is False:
        module = run_plan.get("module", {})
        reason = applicability.get("reason") or "This benchmark case cannot be simulated by the current pipeline."
        raise RuntimeError(
            f"Cannot run Module {module.get('id', '?')} with the current executable pipeline. "
            f"{reason}"
        )
    if not applicability.get("requires_user_confirmation"):
        return
    if allow_unsupported:
        return

    module = run_plan.get("module", {})
    material = run_plan.get("material", {})
    reason = applicability.get("reason") or "This case is outside the current validated module capability."
    message = (
        f"\nWARNING: {material.get('material_id', 'unknown material')} is marked as "
        f"{applicability.get('current_module_capability', 'outside_validated_scope')} for "
        f"Module {module.get('id', '?')}.\n"
        f"{reason}\n"
        "Continue with the simulation anyway? [y/N]: "
    )
    try:
        answer = input(message)
    except EOFError as exc:
        raise RuntimeError(
            "Simulation requires explicit confirmation because the benchmark case is outside "
            "the current validated module capability. Re-run interactively or pass "
            "--allow-unsupported if this is intentional."
        ) from exc

    if answer.strip().casefold() not in {"y", "yes"}:
        raise RuntimeError("Simulation cancelled by user.")


def _configured_mlip_mc_backend(run_plan: dict) -> str:
    return str(_configured_mlip_mc_model(run_plan).get("backend", "mace-torch"))


def _configured_mlip_mc_model(run_plan: dict) -> dict:
    settings = run_plan.get("benchmark", {}).get("potential_benchmark", {})
    mlip_mc = settings.get("mlip_mc", {}) if isinstance(settings, dict) else {}
    model = mlip_mc.get("model", {}) if isinstance(mlip_mc, dict) else {}
    if not isinstance(model, dict):
        raise TypeError("'benchmark.potential_benchmark.mlip_mc.model' must be an object.")
    return model


if __name__ == "__main__":
    raise SystemExit(main())
