#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
CONFIG_DIR="${PROJECT_ROOT}/input_json_files"
PREFIX="benchmark_zif8_co2_nequip_finetuned_widom"

case "${MODE}" in
  setup)
    "${PYTHON_BIN}" benchmark.py \
      "${CONFIG_DIR}/${PREFIX}_smoke.json" \
      --setup-mlip-mc \
      --skip-registry-update
    ;;
  smoke)
    "${PYTHON_BIN}" benchmark.py \
      "${CONFIG_DIR}/${PREFIX}_smoke.json" \
      --run-mlip-mc \
      --skip-registry-update
    "${PYTHON_BIN}" -c 'import json, pathlib; p=pathlib.Path("outputs/module_C_potential_benchmark/runs/zif8_co2_nequip_finetuned_widom_smoke_typed_v1/results/mlip_mc_benchmark.json"); d=json.loads(p.read_text()); assert d["input_contract"]["status"] == "passed"; assert d["adsorbate_species_aliases"]["model_symbols"] == ["Os", "Co", "Os"]; print("Goeminne input contract: passed")'
    ;;
  pilot)
    for seed in 12345 23456 34567; do
      "${PYTHON_BIN}" benchmark.py \
        "${CONFIG_DIR}/${PREFIX}_10000_seed${seed}.json" \
        --run-mlip-mc \
        --skip-registry-update
    done
    ;;
  paper)
    for seed in 12345 23456 34567; do
      "${PYTHON_BIN}" benchmark.py \
        "${CONFIG_DIR}/${PREFIX}_100000_seed${seed}.json" \
        --run-mlip-mc \
        --skip-registry-update
    done
    ;;
  *)
    printf 'Usage: %s {setup|smoke|pilot|paper}\n' "$0" >&2
    exit 2
    ;;
esac
