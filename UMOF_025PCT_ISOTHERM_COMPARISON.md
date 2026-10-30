# MOF-5/CO2: pretrained UMOF versus 25% fine-tuning

For a short, single-seed comparison on physical GPUs 0 and 3:

```bash
conda activate MOF_sim
bash scripts/run_umof_025pct_isotherm.sh --quick --single-seed --gpus 0,3
```

This uses 1,000 equilibration + 2,000 production moves per pressure, all three
pressures, both models and only seed 104729: **18,000 moves total**, with one
model per GPU (9,000 moves each). The output campaign ends in
`_quick_single_seed`, so it cannot mix with the long comparison. A single seed
provides a descriptive comparison only: `ci95` is null and the assessment makes
no statistical improvement claim. Convergence is still open. Stop an unwanted
older run with Ctrl+C in its terminal before launching another comparison.

For the original three-seed, 540,000-move protocol (no additional training or downloads):

```bash
conda activate MOF_sim
bash scripts/run_umof_025pct_isotherm.sh
```

The wrapper changes into the repository automatically. It selects physical GPU 2,
which is separate from the fine-tuning GPU 1. To select another GPU:

```bash
UMOF_GPU=3 bash scripts/run_umof_025pct_isotherm.sh
```

An existing `CUDA_VISIBLE_DEVICES` takes precedence. Within a process, the selected
physical GPU appears as CUDA device 0. `PYTHON_BIN` can select the environment's
Python explicitly. Run under your normal persistent terminal/session for a long job.

The independent comparison uses:

- MOF-5 (= IRMOF-1), CRAFTED DDEC source structure, rigid framework, automatic
  cell replication for a 6-Angstrom cutoff;
- CO2, 298 K, 0.1 / 1 / 10 bar, Peng–Robinson fugacity;
- original UMOF-MH (`r2scan+d4`) versus the selected 25% model (`Default`), both
  float64 with no additional dispersion correction;
- three matching seed identifiers, 104729 / 130363 / 155921;
- 10,000 equilibration and 20,000 production Monte Carlo moves per pressure;
- six isolated model/seed processes, each covering three pressures, executed
  sequentially to avoid competing for GPU memory.

The existing engine evaluates each model's isolated framework and guest energy
references within its GCMC energy convention. No raw total-energy offset from
one model is reused for the other. This test does not retrain either model.

Input model checksums, CIF, pseudo-atoms, guest definition, reference CSV and
configuration are archived in `comparison_inputs.json`. A second invocation reuses
completed model/seed results only for the same archived inputs. The existing engine
also writes restart checkpoints during incomplete tasks. A lock prevents two
invocations from writing the same comparison simultaneously. After an interruption,
rerun the same command. Model/seed logs are named after their task IDs.

Check inputs without starting simulations:

```bash
bash scripts/run_umof_025pct_isotherm.sh --check
```

Repeat the short technical check (two equilibration + four production moves, one
pressure, one seed per model, separate output; not an accuracy result):

```bash
bash scripts/run_umof_025pct_isotherm.sh --smoke
```

Regenerate the report from all six completed runs:

```bash
bash scripts/run_umof_025pct_isotherm.sh --analyze-only
```

Results and logs:

```text
outputs/module_C_potential_benchmark/campaigns/mof5_co2_umof_mh_025pct_comparison_v1/
```

Simulation data:

```text
outputs/module_C_potential_benchmark/runs/mof5_co2_umof_mh_025pct_comparison_v1/
```

The `comparison/` directory contains:

- `isotherm_comparison.png` / `.svg`: both model means, sample standard deviations
  across seeds, and CRAFTED reference;
- `isotherm_seed_comparison.png` / `.svg`: individual seed curves;
- `reference_deviations.png` / `.svg` and `reference_metrics.csv`: deviations and
  descriptive MAE/RMSE against CRAFTED;
- `isotherm_summary.csv` and `isotherm_seeds.csv`: aggregate and individual values;
- `pilot_assessment.json`: mean paired-seed RMSE difference and a Student-t 95%
  interval (negative favors fine-tuning), with a cautious pilot-trend label.

The reference is `DDEC_IRMOF-1_UFF_CO2_298.csv`: **a classical CRAFTED simulation,
not experiment or DFT**. Absolute uptake is compared in mol/kg (= mmol/g), with
pressure converted from Pa to bar; reference values are interpolated only within
its pressure range. Closer agreement does not establish experimental accuracy.

The error bars in the figures are sample standard deviations, not confidence
intervals. The separate Student-t interval uses three independent seed pairs;
it excludes reference uncertainty and systematic errors. Neither completed
simulations nor agreement between seeds establishes equilibration. The existing
GCMC adapter does not establish convergence for these runs. Treat a favorable
result as a pilot trend and inspect the pressure-point histories before claiming
an improvement. A six-move technical smoke result must never support that claim.

Prepared checks: input resolution and hashes; eight targeted unit tests; both
models loaded and completed the separate six-move GPU-2 technical test, including
plot and reference report generation. The scientific comparison is not started
by preparation.

Parallel execution is optional for either protocol: `--gpus 0,1,3` assigns one
process per physical GPU. The explicit list takes precedence over the wrapper's
`CUDA_VISIBLE_DEVICES`. Use GPUs available to your job. Tasks on the same GPU run
sequentially; this parallelizes independent model/seed simulations, not one MC
trajectory. If a worker fails or the launcher is interrupted, its other active
workers are stopped too. A new invocation reuses completed matching model/seed
results. `--quick` alone keeps three seeds (54,000 moves); `--single-seed` alone
keeps the original 30,000 moves per pressure (180,000 total).

Check the short configuration without simulation:

```bash
bash scripts/run_umof_025pct_isotherm.sh --quick --single-seed --gpus 0,3 --check
```

The matching `--smoke` flag also tests the parallel launch separately, with
only six moves per model. `--analyze-only` must use the same `--quick` and
`--single-seed` flags as the comparison being analyzed.

## Longer check at 10 bar only

```bash
conda activate MOF_sim
bash scripts/run_umof_025pct_isotherm.sh --ten-bar --gpus 0,3
```

This runs one seed and only 10 bar: each model receives 10,000 equilibration
moves plus 20,000 production moves. Total work is 60,000 moves; with two GPUs,
each GPU performs 30,000. Outputs are separate under campaign suffix
`_10bar_single_seed`. The seed is 104731, matching the actual 10-bar point seed
from the earlier three-pressure pilot (base 104729 plus pressure index 2).
These are fresh longer simulations, not a continuation of the short pilot.

The `comparison/` directory additionally contains `10bar_history.png` / `.svg`
with binned loading and the cumulative production mean, `10bar_history.csv`,
and `10bar_sampling_diagnostics.json` with four consecutive production-block
means and first/second-half means. These diagnose drift; they do not prove
convergence and are not independent replicates. This single-pressure comparison
retains descriptive accuracy metrics against the classical reference.

`--ten-bar` implies a single seed and cannot be combined with `--quick`.
For input validation append `--check`; for a separate six-move technical test
at 10 bar append `--smoke`. To regenerate only the report, use
`--ten-bar --analyze-only`. The longer scientific run is not started by preparation.
