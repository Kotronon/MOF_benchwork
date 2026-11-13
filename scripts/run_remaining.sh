#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mof-remaining-mpl}"
export MATPLOTLIBRC="${MATPLOTLIBRC:-$ROOT/data/plotting/matplotlibrc}"
exec "${PYTHON_BIN:-python}" -u -m modules.remaining_runs "$@"
