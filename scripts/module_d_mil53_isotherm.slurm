#!/bin/bash
#SBATCH --job-name=mil53-co2
#SBATCH --array=1-90%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lammps_executable="${MIL53_LAMMPS:-/home/ge64zuq/lammps-stable_22Jul2025_update4/src/lmp_serial}"
python3 scripts/run_module_d_mil53.py case \
  --directory outputs/module_D_mil53_co2/runs/classical_isotherm_v1 \
  --case-index "${SLURM_ARRAY_TASK_ID:?}" \
  --command "$lammps_executable"
