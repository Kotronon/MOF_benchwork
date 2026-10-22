#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-prepare}"
DATA_DIRECTORY="${UMOF_DATA_DIRECTORY:-external/umof_data}"
CHECKPOINT="${UMOF_MH_CHECKPOINT:-external/umof_models/uMOF-MH.model}"
OUTPUT_DIRECTORY="${UMOF_OUTPUT_DIRECTORY:-outputs/module_C_umof_finetuning_matrix}"
FRACTIONS="${UMOF_TRAIN_FRACTIONS:-0.1,0.25,0.5,1.0}"
SEEDS="${UMOF_SEEDS:-12345}"
MAX_EPOCHS="${UMOF_MAX_EPOCHS:-50}"
LEARNING_RATE="${UMOF_LEARNING_RATE:-0.0001}"
REPLAY_SAMPLES="${UMOF_REPLAY_SAMPLES:-1000}"
REPLAY_WEIGHT="${UMOF_REPLAY_WEIGHT:-0.25}"
DEVICE="${UMOF_DEVICE:-cpu}"
CPU_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"

require_inputs() {
  if [[ ! -d "${DATA_DIRECTORY}" ]]; then
    printf 'uMOF data directory not found: %s\n' "${DATA_DIRECTORY}" >&2
    exit 2
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    printf 'uMOF-MH checkpoint not found: %s\n' "${CHECKPOINT}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -c 'import ase, mace, torch; print("MACE", mace.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
  if [[ "${DEVICE}" == "cpu" ]]; then
    printf 'uMOF fine-tuning will use CPU with %s thread(s).\n' "${CPU_THREADS}"
  fi
  printf 'MACE may download its MatPES r2SCAN replay data on the first training run.\n'
}

run_matrix() {
  local train_flag=()
  if [[ "${MODE}" == "train" ]]; then
    train_flag=(--train)
  fi
  "${PYTHON_BIN}" -m modules.module_c_mlips.umof_finetuning \
    --matrix \
    --data-directory "${DATA_DIRECTORY}" \
    --checkpoint "${CHECKPOINT}" \
    --output-directory "${OUTPUT_DIRECTORY}" \
    --fractions "${FRACTIONS}" \
    --seeds "${SEEDS}" \
    --max-epochs "${MAX_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --replay-weight "${REPLAY_WEIGHT}" \
    --replay-samples "${REPLAY_SAMPLES}" \
    --device "${DEVICE}" \
    "${train_flag[@]}"
}

case "${MODE}" in
  prepare|train)
    require_inputs
    run_matrix
    ;;
  *)
    printf 'Usage: %s {prepare|train}\n' "$0" >&2
    printf 'Defaults: CPU, fractions=%s, seeds=%s, max_epochs=%s\n' \
      "${FRACTIONS}" "${SEEDS}" "${MAX_EPOCHS}" >&2
    exit 2
    ;;
esac
