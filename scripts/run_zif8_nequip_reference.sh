#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
CONFIG_DIR="input_json_files"
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
