#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="outputs/module_C_potential_benchmark/runs"
REPORT_ROOT="outputs/module_C_potential_benchmark/benchmarks/zif8_co2"
SEEDS=(12345 23456 34567)

NEQUIP_273=()
MACE_273=()
NEQUIP_298=()
for seed in "${SEEDS[@]}"; do
  NEQUIP_273+=("${RUN_ROOT}/zif8_co2_nequip_finetuned_widom_100000_typed_v1_seed${seed}")
  MACE_273+=("${RUN_ROOT}/zif8_co2_mace_mp0a_widom_100000_273K_seed${seed}")
  NEQUIP_298+=("${RUN_ROOT}/zif8_co2_nequip_finetuned_widom_100000_298K_typed_v1_seed${seed}")
done

for run in "${NEQUIP_273[@]}" "${MACE_273[@]}" "${NEQUIP_298[@]}"; do
  if [[ ! -f "${run}/results/mlip_mc_benchmark.json" ]]; then
    printf 'Missing completed result: %s\n' "${run}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m analysis.mlip_widom_replicates \
  "${NEQUIP_273[@]}" \
  --output-dir "${REPORT_ROOT}/nequip_finetuned_273K" \
  --compare-references

"${PYTHON_BIN}" -m analysis.mlip_widom_replicates \
  "${MACE_273[@]}" \
  --output-dir "${REPORT_ROOT}/mace_mp0a_273K" \
  --compare-references \
  --curated-reference "${REPORT_ROOT}/nequip_finetuned_273K/widom_curated_reference.json"

"${PYTHON_BIN}" -m analysis.mlip_widom_replicates \
  "${NEQUIP_298[@]}" \
  --output-dir "${REPORT_ROOT}/nequip_finetuned_298K" \
  --compare-references

printf '\nAnalysis complete: %s\n' "${REPORT_ROOT}"
