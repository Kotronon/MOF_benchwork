#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${UMOF_GPU:-2}}"
export OMP_NUM_THREADS="${UMOF_CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/umof-isotherm-mpl}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" -m modules.module_c_mlips.umof_isotherm_comparison "$@"
