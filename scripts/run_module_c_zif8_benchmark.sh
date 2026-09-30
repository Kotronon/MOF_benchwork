#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-all}"
SEEDS=(12345 23456 34567)
CONFIG_DIR="${PROJECT_ROOT}/input_json_files"

run_config() {
  local config="$1"
  printf '\nRunning %s\n' "${config}"
  "${PYTHON_BIN}" benchmark.py \
    "${CONFIG_DIR}/${config}" \
    --run-mlip-mc \
    --skip-registry-update
}

run_setup() {
  "${PYTHON_BIN}" benchmark.py \
    "${CONFIG_DIR}/benchmark_zif8_co2_nequip_finetuned_widom_smoke.json" \
    --setup-mlip-mc \
    --skip-registry-update
  "${PYTHON_BIN}" benchmark.py \
    "${CONFIG_DIR}/benchmark_zif8_co2_mlip_mc_widom_smoke.json" \
    --setup-mlip-mc \
    --skip-registry-update
}

run_smoke() {
  run_config "benchmark_zif8_co2_nequip_finetuned_widom_smoke.json"
  "${PYTHON_BIN}" -c 'import json, pathlib; p=pathlib.Path("outputs/module_C_potential_benchmark/runs/zif8_co2_nequip_finetuned_widom_smoke_typed_v1/results/mlip_mc_benchmark.json"); d=json.loads(p.read_text()); assert d["input_contract"]["status"] == "passed"; assert d["adsorbate_species_aliases"]["model_symbols"] == ["Os", "Co", "Os"]; print("Goeminne input contract: passed")'
}

run_production() {
  local seed
  for seed in "${SEEDS[@]}"; do
    run_config "benchmark_zif8_co2_nequip_finetuned_widom_100000_seed${seed}.json"
  done
  for seed in "${SEEDS[@]}"; do
    run_config "benchmark_zif8_co2_mace_mp0a_widom_100000_273K_seed${seed}.json"
  done
  for seed in "${SEEDS[@]}"; do
    run_config "benchmark_zif8_co2_nequip_finetuned_widom_100000_298K_seed${seed}.json"
  done
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
