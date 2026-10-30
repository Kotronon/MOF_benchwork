#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PYTHON_BIN="${PYTHON_BIN:-/home/ge64zuq/miniforge3/envs/MOF_sim/bin/python}"
CPU_THREADS="${UMOF_CPU_THREADS:-1}"
if ! [[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'UMOF_CPU_THREADS must be a positive integer.\n' >&2
  exit 2
fi
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" -m modules.module_c_mlips.umof_adsorption_refinement "$@"
