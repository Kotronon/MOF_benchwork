#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/umof-stability-mpl}"
export PYTHONUNBUFFERED=1
if (( $# == 0 )); then set -- status; fi
exec "${PYTHON_BIN:-python}" -m modules.module_c_mlips.umof_isotherm_stability "$@"
