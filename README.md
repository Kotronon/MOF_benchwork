# MOF_benchwork

Use one project environment for planning, LAMMPS, analysis, and MLIP-MC:

```bash
conda activate MOF_sim
```

## Module C with MLIP-MC

The MLIP-MC integration uses the active Python environment. It does not create
or switch to another environment. The selected backend is read explicitly from
the benchmark JSON, so installing multiple supported calculators does not
change which model is executed.

Prepare the pinned MLIP-MC package, the selected backend, and its model
checkpoint on a workstation or cluster login node:

```bash
python benchmark.py \
  input_json_files/benchmark_zif8_co2_mlip_mc_widom_smoke.json \
  --setup-mlip-mc \
  --skip-registry-update
```

Inspect the resolved run without performing a calculation:

```bash
python benchmark.py \
  input_json_files/benchmark_zif8_co2_mlip_mc_widom_smoke.json \
  --dry-run \
  --skip-registry-update
```

Run the Widom smoke benchmark:

```bash
python benchmark.py \
  input_json_files/benchmark_zif8_co2_mlip_mc_widom_smoke.json \
  --run-mlip-mc \
  --skip-registry-update
```

If setup and execution must happen in one command, add `--install-missing`.
This invokes pip through the same interpreter that runs `benchmark.py`:

```bash
python benchmark.py CONFIG.json --run-mlip-mc --install-missing
```

For offline compute nodes, always run `--setup-mlip-mc` on a network-enabled
login node first. MLIP-MC is pinned in `requirements-mlip-mc.txt`; model and
dependency metadata are recorded in every Module C run directory.

### Widom convergence runs

Three production templates use independent seeds and 10,000 attempts each.
Each run reports cumulative results after 1,000, 5,000, and 10,000 attempts,
plus non-overlapping block SEM and normal 95% intervals:

```bash
python benchmark.py \
  input_json_files/benchmark_zif8_co2_mlip_mc_widom_10000_seed12345.json \
  --run-mlip-mc --skip-registry-update
```

Repeat with the `seed23456` and `seed34567` input files. Every run writes
`widom_analysis.json`, `widom_convergence.csv`, and
`widom_convergence.png` below its `engine/mlip_mc/widom` directory. Henry
coefficients are reported in both `mol kg^-1 Pa^-1` and
`mmol g^-1 bar^-1`; zero-loading isosteric heat is reported in `kJ mol^-1`.

Aggregate the three independent seeds without manually copying values:

```bash
python -m analysis.mlip_widom_replicates \
  outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_seed12345 \
  outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_seed23456 \
  outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_seed34567 \
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_replicates
```

The aggregate uses a Student-t confidence interval across seeds and marks the
result converged only when all configured criteria are met.
