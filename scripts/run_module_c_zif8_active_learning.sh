#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-status}"
CP2K_CONDA_SPEC="${CP2K_CONDA_SPEC:-conda-forge::cp2k=2026.1=ha306f6e_2}"
CONFIG="${PROJECT_ROOT}/input_json_files/benchmark_zif8_co2_module_c_active_learning.json"
STATE="${PROJECT_ROOT}/outputs/module_C_potential_benchmark/runs/zif8_co2_active_learning/zif_8__mace_mp_0a_active_learning__active_learning__seed12345/active_learning/state.json"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command is unavailable in the active environment: %s\n' "$1" >&2
    exit 1
  fi
}

install_cp2k() {
  require_command conda
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'Activate MOF_sim before running setup so CP2K is installed into the intended environment.\n' >&2
    exit 1
  fi
  if [[ "$(basename -- "${CONDA_PREFIX}")" != "MOF_sim" ]]; then
    printf 'Refusing to modify %s. Activate the MOF_sim environment first.\n' "${CONDA_PREFIX}" >&2
    exit 1
  fi
  printf 'Installing %s into %s ...\n' "${CP2K_CONDA_SPEC}" "${CONDA_PREFIX}"
  conda install --yes --prefix "${CONDA_PREFIX}" "${CP2K_CONDA_SPEC}"
  hash -r
}

verify_python_stack() {
  "${PYTHON_BIN}" -c '
import importlib
import importlib.util
import json
from importlib import metadata

import ase
import mace
import mlip_mc
import ovito
import torch

versions = {
    "python_packages": {},
    "ovito": ovito.version_string,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
}
for distribution in ("ase", "mace-torch", "mlip-mc", "nequip"):
    try:
        versions["python_packages"][distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        versions["python_packages"][distribution] = None
if importlib.util.find_spec("nequip") is not None:
    importlib.import_module("nequip")
    versions["nequip_import"] = "passed"
else:
    versions["nequip_import"] = "not_installed"
print(json.dumps(versions, indent=2))
'
}

preflight_dft() {
  require_command cp2k
  require_command mpirun
  mpirun --version >/dev/null
  mpirun -np 1 cp2k --version >/dev/null
  "${PYTHON_BIN}" -c 'import ase, mace, mlip_mc'
}

preflight_training() {
  require_command mace_run_train
  "${PYTHON_BIN}" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Run training on a host with an accessible CUDA GPU.")
'
}

run_benchmark() {
  "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
    --run-module-c \
    --stage active-learning \
    "$@" \
    --skip-registry-update
}

state_status() {
  "${PYTHON_BIN}" -c '
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
' "${STATE}"
}

run_production() {
  preflight_dft
  preflight_training
  if [[ ! -f "${STATE}" ]]; then
    printf 'Initializing active learning and running the CP2K cutoff tests.\n'
    run_benchmark --submit
  fi

  for _ in $(seq 1 20); do
    current_status="$(state_status)"
    printf 'Active-learning state: %s\n' "${current_status}"
    case "${current_status}" in
      awaiting_cutoff_test|awaiting_dft)
        run_benchmark --submit --resume
        ;;
      ready_to_train|training_prepared|models_ready)
        run_benchmark --train --resume
        ;;
      dft_validated)
        printf 'Active learning passed the DFT validation gate.\n'
        return
        ;;
      adaptation_not_converged|cutoff_not_converged)
        printf 'Active learning ended without validation: %s\n' "${current_status}" >&2
        return 1
        ;;
      *)
        printf 'Cannot automatically advance state: %s\n' "${current_status}" >&2
        return 2
        ;;
    esac
  done
  printf 'Safety limit reached after 20 workflow transitions. Inspect the state before resuming.\n' >&2
  return 2
}

case "${MODE}" in
  setup)
    install_cp2k
    "${PYTHON_BIN}" benchmark.py "${CONFIG}" \
      --setup-module-c \
      --install-missing \
      --skip-registry-update
    preflight_dft
    verify_python_stack
    conda list --prefix "${CONDA_PREFIX}" cp2k
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
    printf 'Running the CP2K cutoff tests directly on this host. This command blocks until they finish.\n'
    run_benchmark --submit
    ;;
  advance)
    preflight_dft
    if [[ ! -f "${STATE}" ]]; then
      printf 'No state exists. Run %s start first.\n' "$0" >&2
      exit 2
    fi
    printf 'Advancing the workflow and running pending CP2K jobs directly on this host.\n'
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
  production)
    run_production
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
from pathlib import Path
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
root = Path(sys.argv[1]).parent
jobs_path = None
if state.get("status") == "awaiting_cutoff_test":
    jobs_path = root / "cutoff_test" / "cp2k_jobs.json"
elif state.get("status") == "awaiting_dft":
    jobs_path = (
        root
        / f"iteration_{int(state.get('iteration', 0)):02d}"
        / "cp2k"
        / "cp2k_jobs.json"
    )
if jobs_path is not None and jobs_path.is_file():
    jobs = json.load(open(jobs_path, encoding="utf-8"))["jobs"]
    progress = {"total": len(jobs), "pending": 0, "running": 0, "completed": 0, "failed": 0}
    for job in jobs:
        output = Path(job["output_path"])
        if not output.is_file():
            progress["pending"] += 1
            continue
        content = output.read_text(encoding="utf-8", errors="replace")
        if "PROGRAM ENDED AT" in content:
            progress["completed"] += 1
        elif "ABORT" in content or "PROGRAM STOPPED" in content:
            progress["failed"] += 1
        else:
            progress["running"] += 1
    summary["cp2k_progress"] = progress
print(json.dumps(summary, indent=2))
' "${STATE}"
    ;;
  *)
    printf 'Usage: %s {setup|assess|start|advance|prepare-training|train|production|widom|gcmc-pilot|status|reset}\n' "$0" >&2
    exit 2
    ;;
esac
