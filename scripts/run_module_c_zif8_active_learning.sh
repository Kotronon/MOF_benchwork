#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-status}"
CONFIG="${PROJECT_ROOT}/input_json_files/benchmark_zif8_co2_module_c_active_learning.json"
STATE="${PROJECT_ROOT}/outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning/state.json"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command is unavailable in the active environment: %s\n' "$1" >&2
    exit 1
  fi
}

preflight_dft() {
  require_command sbatch
  require_command cp2k.psmp
  "${PYTHON_BIN}" -c 'import ase, mace, mlip_mc'
}

preflight_training() {
  require_command mace_run_train
  "${PYTHON_BIN}" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Run training inside a GPU allocation.")
'
}

run_benchmark() {
  "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
    --run-module-c \
    --stage active-learning \
    "$@" \
    --skip-registry-update
}

case "${MODE}" in
  setup)
    "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
      --setup-module-c \
      --install-missing \
      --skip-registry-update
    preflight_dft
    ;;
  assess)
    "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
      --assess-module-c \
      --skip-registry-update
    ;;
  start)
    preflight_dft
    if [[ -f "${STATE}" ]]; then
      printf 'State already exists: %s\nUse advance, or explicitly use reset to discard it.\n' "${STATE}" >&2
      exit 2
    fi
    run_benchmark --submit
    ;;
  advance)
    preflight_dft
    if [[ ! -f "${STATE}" ]]; then
      printf 'No state exists. Run %s start first.\n' "$0" >&2
      exit 2
    fi
    run_benchmark --submit --resume
    ;;
  prepare-training)
    if [[ ! -f "${STATE}" ]]; then
      printf 'No state exists. Run %s start first.\n' "$0" >&2
      exit 2
    fi
    run_benchmark --resume
    ;;
  train)
    preflight_training
    if [[ ! -f "${STATE}" ]]; then
      printf 'No state exists. Run %s start first.\n' "$0" >&2
      exit 2
    fi
    run_benchmark --train --resume
    ;;
  widom)
    preflight_training
    "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
      --run-module-c \
      --stage widom \
      --jobs "${JOBS:-1}" \
      --resume \
      --skip-registry-update
    ;;
  gcmc-pilot)
    preflight_training
    "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
      --run-module-c \
      --stage gcmc-pilot \
      --jobs "${JOBS:-1}" \
      --resume \
      --skip-registry-update
    ;;
  reset)
    run_benchmark --overwrite
    ;;
  status)
    if [[ ! -f "${STATE}" ]]; then
      printf 'No active-learning state exists yet.\n'
      exit 0
    fi
    "${PYTHON_BIN}" -c '
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
summary = {
    key: state.get(key)
    for key in (
        "schema_version",
        "status",
        "iteration",
        "stable_rounds",
        "candidate_count",
        "labeled_count",
        "selected_cutoff_Ry",
        "failure_reason",
    )
}
summary["training_count"] = len(state.get("training_ids", []))
summary["validation_count"] = len(state.get("validation_ids", []))
summary["test_count"] = len(state.get("test_ids", []))
print(json.dumps(summary, indent=2))
' "${STATE}"
    ;;
  *)
    printf 'Usage: %s {setup|assess|start|advance|prepare-training|train|widom|gcmc-pilot|status|reset}\n' "$0" >&2
    exit 2
    ;;
esac
