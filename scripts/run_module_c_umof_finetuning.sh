#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-prepare}"
if (( $# )); then shift; fi
CPU_THREADS="${UMOF_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-${PBS_NP:-${NSLOTS:-${OMP_NUM_THREADS:-1}}}}}"
if ! [[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'UMOF_CPU_THREADS must be a positive integer.\n' >&2
  exit 2
fi
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}"
ARGS=("${MODE}" --matrix)
# Unspecified options inherit campaign.json during resume/evaluate, including custom splits.
add_option() { if [[ -n "${!1:-}" ]]; then ARGS+=("$2" "${!1}"); fi; }
add_option UMOF_MODEL --model
add_option UMOF_OUTPUT_DIRECTORY --output-directory
add_option UMOF_DATA_DIRECTORY --data-directory
add_option UMOF_MH_CHECKPOINT --checkpoint
add_option UMOF_CHECKPOINT --checkpoint
add_option UMOF_REPLAY_SOURCE --replay-source
add_option UMOF_TRAIN_FRACTIONS --fractions
add_option UMOF_SEEDS --seeds
add_option UMOF_LEARNING_RATE --learning-rate
add_option UMOF_REPLAY_SAMPLES --replay-samples
add_option UMOF_REPLAY_VALID_SAMPLES --replay-valid-samples
add_option UMOF_REPLAY_WEIGHT --replay-weight
add_option UMOF_DEVICE --device
add_option UMOF_GASES --gases
add_option UMOF_GAS_DEFINITIONS --gas-definitions
add_option UMOF_VALIDATION_MOFS --validation-mofs
add_option UMOF_TEST_MOFS --test-mofs
add_option UMOF_MAX_EPOCHS --max-epochs
exec "${PYTHON_BIN}" -m modules.module_c_mlips.umof_finetuning "${ARGS[@]}" "$@"
