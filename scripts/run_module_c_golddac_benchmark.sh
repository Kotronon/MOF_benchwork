#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-all}"
CONFIG_DIR="${PROJECT_ROOT}/input_json_files"
SMOKE_CONFIG="${CONFIG_DIR}/benchmark_golddac_co2_mace_models_smoke.json"
PRODUCTION_CONFIG="${CONFIG_DIR}/benchmark_golddac_co2_mace_models_production.json"

run_setup() {
  "${PYTHON_BIN}" benchmark.py "${SMOKE_CONFIG}" \
    --setup-golddac \
    --skip-registry-update
}

run_smoke() {
  "${PYTHON_BIN}" benchmark.py "${SMOKE_CONFIG}" \
    --potentials \
    --skip-registry-update
}

run_production() {
  "${PYTHON_BIN}" benchmark.py "${PRODUCTION_CONFIG}" \
    --potentials \
    --skip-registry-update
}

case "${MODE}" in
  setup)
    run_setup
    ;;
  smoke)
    run_smoke
    ;;
  production)
    run_production
    ;;
  all)
    run_setup
    run_smoke
    run_production
    ;;
  *)
    printf 'Usage: %s {setup|smoke|production|all}\n' "$0" >&2
    exit 2
    ;;
esac
