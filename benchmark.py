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
from pipeline.lammps_inputs import gcmc_input_builder
from pipeline.materialization import materialize_benchmark
from pipeline.planning import build_run_plan, resolve_benchmark, select_module
from pipeline.prepare import prepare, prepare_benchmark, timestamp_run_id
from pipeline.runners import run_benchmark, run_isotherm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan a MOF benchmark run from benchmark.json.")
    parser.add_argument("config", nargs="?", default="benchmark.json", help="Path to benchmark JSON input.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without running LAMMPS.")
    parser.add_argument("--prepare", action="store_true", help="Prepare the benchmark environment.")
    parser.add_argument("--run-test", action="store_true", help="Run a test GCMC simulation after preparation.")
    parser.add_argument("--run-isotherm", action="store_true", help="Run the full isotherm after preparation.")
    parser.add_argument("--run-id", help="Write outputs to outputs/<module>/runs/<run-id>.")
    parser.add_argument("--new-run", action="store_true", help="Write outputs to a timestamped run directory.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the target working directory already exists.")
    args = parser.parse_args(argv)

    config = load_benchmark_data(args.config)
    if args.new_run:
        config.setdefault("output", {})["run_id"] = timestamp_run_id()
    if args.run_id:
        config.setdefault("output", {})["run_id"] = args.run_id
    if args.no_overwrite:
        config.setdefault("output", {})["overwrite"] = False
    run_plan = build_run_plan(config)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2))
        return 0
    if args.prepare:
        prepare_plan = prepare_benchmark(run_plan)
        result = materialize_benchmark(prepare_plan)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_test:
        prepare_plan = prepare_benchmark(run_plan)
        materialized_plan = materialize_benchmark(prepare_plan)
        result = run_benchmark(materialized_plan)
        print(json.dumps(result, indent=2))
        return 0
    if args.run_isotherm:
        prepare_plan = prepare_benchmark(run_plan)
        materialized_plan = materialize_benchmark(prepare_plan)
        result = run_isotherm(materialized_plan)
        print(json.dumps(result, indent=2))
        return 0
    print(json.dumps(run_plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
