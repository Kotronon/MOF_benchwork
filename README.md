# MOF_benchwork

Use one project environment for planning, LAMMPS, analysis, and MLIP-MC:

```bash
conda activate MOF_sim
```

Install the core Python dependencies with pip inside the active environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` covers Python packages only. LAMMPS, OVITO, and CP2K are
external executables or Conda packages and must be installed separately. Module
C MLIP-MC dependencies are pinned in `requirements-mlip-mc.txt` and can be set
up through the Module-C commands below.

## Pipeline architecture

`benchmark.py` is the CLI orchestrator. It loads and normalizes the input JSON,
builds a run plan, and delegates module-specific behavior through the explicit
registry in `modules/registry.py`.

Registered modules share the same public contract from `modules/base.py`:

- metadata: `id`, `name`, `slug`, `task_aliases`, `default_metrics`
- selection: `selection_score(config)`
- execution hooks: `assess`, `prepare`, `materialize`, `run_test`, `run`, `evaluate`

Module A, B, and C reuse the shared planning/materialization pipeline where
possible. Module D is registered as a clear placeholder for flexible-framework
benchmarks; it can be selected and reported, but execution fails explicitly
until its workflow is implemented.

To add a new module, add a `BenchmarkModule` entry to `MODULE_REGISTRY`, define
unique task aliases and default metrics, implement module-specific hooks only
where the shared pipeline is insufficient, and add selection/dry-run tests.

## Module selection

Automatic selection is property-based:

- Module A: single-component rigid-framework GCMC.
- Module B: true multi-component mixture/selectivity GCMC.
- Module C: Widom, Henry, or potential-benchmark workflows.
- Module D: flexible-framework or MD/GCMC-flexibility workflows.

A single component with `mixture: {"CO2": 1.0}` is normalized to a normal
single-gas run and therefore stays in Module A. Mixture mole fractions are only
kept when at least two configured components are present and the fractions sum
to one.

Inspect any resolved plan without running a simulation:

```bash
python benchmark.py CONFIG.json --dry-run --skip-registry-update
```

## Module A and B with LAMMPS GCMC

Module A covers single-component adsorption. The default `benchmark.json`
resolves MOF-5 to IRMOF-1, prepares the framework, force field, molecule
template, and LAMMPS inputs, and can run a short technical smoke test:

```bash
python benchmark.py benchmark.json --prepare --skip-registry-update
python benchmark.py benchmark.json --run-test --skip-registry-update
```

Module B extends the same LAMMPS path to mixtures. A valid mixture config
contains at least two components and gas-phase mole fractions:

```json
{
  "adsorbates": {
    "components": ["CO2", "N2"],
    "mixture": {"CO2": 0.15, "N2": 0.85}
  }
}
```

For each pressure point, the input builder creates one `fix gcmc_*` per
component, uses partial pressures `P_i = y_i * P_total`, writes component
count variables such as `v_nco2` and `v_nn2`, and evaluates selectivity as:

```text
S_CO2/N2 = (x_CO2 / x_N2) / (y_CO2 / y_N2)
```

Run the local CO2/N2 smoke benchmark with:

```bash
python benchmark.py \
  input_json_files/benchmark_mof5_co2_n2_mixture_smoke_test.json \
  --run-test --skip-registry-update --overwrite
```

The full mixture path writes `mixture_results.csv` and
`selectivity_surface.csv` in the run directory.

## Module C with MLIP-MC

### Extensible campaign workflow

The `campaign` workflow accepts a direct periodic CIF and evaluates
applicability from structure, model element coverage, minimum-image cell size,
pore accessibility, adsorbate definitions, and any required charge/DFT
metadata. An unregistered but technically valid MOF is run as
`screening_only`; it is not presented as quantitatively validated.

```bash
# Property-based assessment without simulation
python benchmark.py \
  input_json_files/benchmark_mof5_co2_module_c_campaign_screening.json \
  --assess-module-c --skip-registry-update

# Optional dependencies are installed only with --install-missing
python benchmark.py CONFIG.json \
  --setup-module-c --install-missing --skip-registry-update

# Zero-shot screening, production stages, and an optional FLAMES cross-check
python benchmark.py CONFIG.json --run-module-c --stage screening
python benchmark.py CONFIG.json --run-module-c --stage widom --resume
python benchmark.py CONFIG.json --run-module-c --stage gcmc-pilot --resume
python benchmark.py CONFIG.json --cross-check-engine flames --stage screening
python benchmark.py CONFIG.json --analyze-module-c
```

### uMOF pretrained strategy comparison

The uMOF comparison is treated as a CP2K-free pretrained strategy. It does not
launch CP2K and does not fine-tune locally. The configured uMOF-MH and
uMOF-POLAR checkpoints are stored under `external/models/umof/` or under the
explicit paths in the JSON. The release stores both checkpoints inside
`umof_models.zip`; setup extracts the corresponding `umof_models/uMOF-*.model`
member. The comparison selects the `r2scan+d4` head. uMOF-POLAR uses the
PolarMACE loader and requires `graph_longrange==0.4.0`, which setup checks and
installs with `--install-missing`.
When the files are missing, setup can download the
required model assets from the uMOF Figshare release, but only when
`--install-missing` is passed. If CP2K labels already exist from the local
active-learning workflow, the report includes energy and force MAE against
those labels. Without labels, the CP2K error columns are left empty and the run
still records model outputs and runtime.

Prepare/check the plan:

```bash
python benchmark.py \
  input_json_files/benchmark_mof5_co2_umof_strategy_comparison.json \
  --dry-run --skip-registry-update
```

Download/check the uMOF checkpoints and runtime dependencies:

```bash
python benchmark.py \
  input_json_files/benchmark_mof5_co2_umof_strategy_comparison.json \
  --setup-module-c --install-missing --skip-registry-update
```

For a machine without a free CUDA GPU, use
`input_json_files/benchmark_mof5_co2_umof_strategy_comparison_cpu.json` instead.
It runs all three models on CPU and writes to a separate run directory; its
runtime measurements are CPU timings.

Run the comparison after setup:

```bash
python benchmark.py \
  input_json_files/benchmark_mof5_co2_umof_strategy_comparison.json \
  --compare-strategies --skip-registry-update
```

Expected outputs are written to the run directory:

- `results/strategy_comparison.json`
- `results/strategy_comparison.csv`
- `results/strategy_comparison_progress.json` saves completed evaluations during the run
- `reports/cp2k_reference_comparison.png` when CP2K reference labels exist

Configured `exclude_regions` filters are applied before model evaluation, including
the synthetic overlapping structures in the smoke set.

The intended scientific comparison is:

```text
local CP2K-labeled MACE fine-tuning
vs.
uMOF pretrained MOF-specific model without new CP2K labels
```

The optional FLAMES adapter is pinned to release 0.4.8 and an immutable Git
commit because 0.4.9 is not an available release. The setup command reports a
missing dependency without changing the environment unless
`--install-missing` is supplied.

The three paper systems are available as one regression campaign. Select a
single system with `--systems` when running on a workstation:

```bash
python benchmark.py \
  input_json_files/benchmark_module_c_reference_campaign.json \
  --assess-module-c --skip-registry-update

python benchmark.py \
  input_json_files/benchmark_module_c_reference_campaign.json \
  --run-module-c --stage screening --systems ZIF-8
```

For `validation.mode: active_learning`, the first call creates separate,
stratified fit, validation, and permanently held-out test splits, plus
empty-framework and isolated-CO2 baselines and CP2K cutoff-test jobs.
`--submit` submits CP2K jobs when a scheduler is configured, or executes them
directly on the current host when `scheduler: local` is configured. MACE
training requires the separate `--train` flag so it cannot start accidentally
on a login node.

```bash
python benchmark.py \
  input_json_files/benchmark_mof5_co2_module_c_active_learning.json \
  --run-module-c --stage active-learning

python benchmark.py CONFIG.json \
  --run-module-c --stage active-learning --submit --resume
```

The ZIF-8 workflow is wrapped in a helper for a directly accessed cluster
without Slurm. Its `setup` mode installs the pinned CP2K package into the
currently active Conda environment. The compatible one-environment pin is
`conda-forge::cp2k=2026.1=ha306f6e_2` together with
`conda-forge::dbcsr=2.9.1=mpi_openmpi_hcf4b7bf_3`. This CP2K build requires
`libdbcsr.so.2.9`, which DBCSR 2.10 does not provide. LibXC and libxc-c are
also pinned to CPU builds of 7.0.0 because 7.1.2 fails during initialization
with this CP2K build. Setup verifies CP2K through MPI and re-imports ASE,
Torch, MACE, MLIP-MC, and optional NequIP. The OVITO application and optional
Python module are reported separately; the Python module is not required
for active learning. The CP2K
configuration uses one local MPI job with 16 ranks; adjust
`mpi_processes_per_job` and `max_parallel_jobs` to the actual node before
production. `start` and `advance` block while their CP2K jobs run, so launch
them in `tmux` or another persistent shell. When the state reaches
`training_prepared`, run `train` on a host with CUDA:

```bash
bash scripts/run_module_c_zif8_active_learning.sh setup
bash scripts/run_module_c_zif8_active_learning.sh assess
bash scripts/run_module_c_zif8_active_learning.sh start
bash scripts/run_module_c_zif8_active_learning.sh status
bash scripts/run_module_c_zif8_active_learning.sh advance

# Only on a host with an accessible CUDA GPU:
bash scripts/run_module_c_zif8_active_learning.sh train

# Only after the state reports dft_validated, also on a GPU node:
bash scripts/run_module_c_zif8_active_learning.sh widom
bash scripts/run_module_c_zif8_active_learning.sh gcmc-pilot
```

If a cutoff test ends with `cutoff_not_converged`, append higher values to
`active_learning.cp2k.cutoff_test_Ry` in the input configuration and resume with
`advance` or `production`. The existing cutoff jobs and data splits are preserved;
only the added cutoffs run. The ZIF-8 configuration includes 1000 and 1200 Ry.
Physical and numerical CP2K settings must remain unchanged when reusing results.

When individual DFT jobs abort with `SCF run NOT converged`, use
`bash scripts/run_module_c_zif8_active_learning.sh retry-scf`, then resume
`production`. The preparation archives failed inputs, outputs, logs and restart
wavefunctions, raises `MAX_SCF` to the configured `max_scf` (1000 for ZIF-8),
and uses `SCF_GUESS RESTART`. Successful jobs and the SCF tolerance are preserved.
Only failures explicitly caused by SCF convergence are prepared for retry.

If CP2K and CUDA are available on the same directly accessed node, the full
adaptation loop can instead be run with one blocking command:

```bash
bash scripts/run_module_c_zif8_active_learning.sh production
```

This resumes an existing state, executes every pending CP2K batch, trains the
three-model committee, and stops at either `dft_validated` or a reproducible
failure state. It does not automatically start Widom or GCMC production.

For example, start a persistent direct-host session with:

```bash
conda activate MOF_sim
tmux new -s zif8-active-learning
bash scripts/run_module_c_zif8_active_learning.sh setup
bash scripts/run_module_c_zif8_active_learning.sh assess
bash scripts/run_module_c_zif8_active_learning.sh production
```

Do not run `start` or `advance` concurrently on two hosts against the same
state directory. The active-learning state updates are intentionally serial.
`assess` checks scientific and structural applicability only. `setup` reports
the actual CP2K, MPI, Python-backend, and CUDA availability on the current
host. During a direct CP2K batch, `status` can be called from a second shell to
count pending, running, completed, and failed calculations.

The 50 initial host/guest configurations are partitioned into 24 fitting,
6 validation, and 20 test configurations. The two isolated baselines are
training-only. The test split is never passed to `mace_run_train`.

The state machine performs CP2K cutoff validation, MACE-MP-0a committee
fine-tuning, held-out energy/force evaluation, and disagreement-based candidate
selection. It records `dft_validated` only after two consecutive passing
rounds. Budget exhaustion produces `adaptation_not_converged` and blocks
automatic Widom/GCMC production. CP2K and MACE are intentionally not launched
by setup or assessment commands.

### GoldDAC DFT validation gate

Before running long Widom or GCMC calculations, validate the potential on the
held-out GoldDAC test set used by MOFSimBench. The integrated task compares
MACE-MP and MACE-DAC-1 against DFT interaction energies and combined-system
forces. It never uses the GoldDAC train or validation splits as benchmark
references.

On a network-enabled cluster login node, install the active-environment
dependencies and download the checksummed GoldDAC v3 data and MACE-DAC-1
checkpoint:

```bash
conda activate MOF_sim
bash scripts/run_module_c_golddac_benchmark.sh setup
```

Then run the balanced six-configuration gate and inspect its automatically
generated JSON, CSV, and plot:

```bash
bash scripts/run_module_c_golddac_benchmark.sh smoke
```

Only after the smoke result is technically sound should the complete CO2
test subset be evaluated:

```bash
bash scripts/run_module_c_golddac_benchmark.sh production
```

To isolate the effect of the external D3 correction, run the four-way
ablation matrix (MACE-MP and MACE-DAC, each with and without D3) on identical
configurations. The second script argument selects this profile while the
original one-argument commands remain unchanged:

```bash
bash scripts/run_module_c_golddac_benchmark.sh setup dispersion
bash scripts/run_module_c_golddac_benchmark.sh smoke dispersion
bash scripts/run_module_c_golddac_benchmark.sh production dispersion
```

The smoke profile uses six region-balanced test configurations. The production
profile evaluates all 156 CO2 configurations from the held-out GoldDAC test
split. Its aggregate report includes signed mean and median energy differences
in addition to MAE/RMSE, so systematic overbinding or underestimated repulsion
is not hidden by absolute errors.

Results are written below
`outputs/module_C_potential_benchmark/runs/golddac_co2_*`. The aggregate JSON
reports MAE/RMSE globally, by potential-energy region, and by MOF. GoldDAC v3
is downloaded from [DAC-SIM](https://doi.org/10.6084/m9.figshare.27978474.v3),
and the protocol follows
[MOFSimBench](https://doi.org/10.1038/s41524-025-01872-3). The model checkpoint
is pinned to a DAC-SIM Git commit and verified by SHA-256.

This static DFT test diagnoses potential quality. Passing it does not by
itself validate Henry coefficients, adsorption heats, or isotherms; those
remain separate downstream sampling tests.

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
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_replicates \
  --compare-references
```

The aggregate uses a Student-t confidence interval across seeds and marks the
result converged only when all configured criteria are met. With
`--compare-references`, it also creates the layered reference report below
`reference_comparison/`.

Compare the pooled result with the matching CRAFTED baselines and the curated
298 K experimental ZIF-8/CO2 isotherm already stored in NIST ISODB:

```bash
python -m analysis.mlip_widom_reference \
  outputs/module_C_potential_benchmark/widom_zif8_co2_replicates/widom_replicate_summary.json \
  --crafted-root CRAFTED-2.0.0 \
  --isodb-root isodb-library \
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_replicates/reference_comparison
```

The comparison writes a layered JSON/CSV report and one plot. CRAFTED is
labelled as a classical simulation baseline, while the Hwang et al. isotherm
is fitted only over the configured low-pressure interval. A locally curated
Goeminne or other DFT-derived Widom result can be added with a repeatable
`--curated-reference reference.json` option. Material, adsorbate, and
temperature are validated before the values are accepted.

The curated reference format is intentionally small and provenance-aware:

```json
{
  "reference_id": "goeminne_zif8_finetuned",
  "label": "Goeminne finetuned NequIP",
  "reference_class": "dft_finetuned",
  "material": "ZIF-8",
  "adsorbate": "CO2",
  "temperature_K": 298.15,
  "metrics": {
    "isosteric_heat_zero_loading_kj_mol": {
      "value": 14.0,
      "uncertainty": 0.3
    }
  },
  "provenance": {
    "doi": "10.5281/zenodo.7904959",
    "model": "model_ZIF8_N=1000.pth"
  }
}
```

The numeric values above only demonstrate the schema and must be replaced by
values obtained at the same temperature and with a documented protocol.

### D3 energy diagnostics

When matching no-D3 and D3 runs use the same seeds, diagnose the potential
change configuration by configuration from their binary Widom traces:

```bash
python -m analysis.mlip_widom_energy_diagnostics \
  --baseline-run outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_seed12345 \
  --baseline-run outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_seed23456 \
  --dispersion-run outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_with_dispersion_seed12345 \
  --dispersion-run outputs/module_C_potential_benchmark/runs/zif8_co2_mlip_mc_widom_10000_with_dispersion_seed23456 \
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_d3_energy_diagnostics
```

The diagnostic validates that paired records contain identical atom positions,
then reports the D3 interaction-energy contribution, its distance dependence,
effective sample size, and concentration of Boltzmann weight. It also exports
the highest-weight configurations as an extended XYZ trajectory for inspection
in OVITO. This comparison uses energies already present in the completed runs
and therefore does not require another MACE calculation.

### Fine-tuned NequIP reference

The Goeminne ZIF-8 model is treated as an interaction-energy model, not a
general total-energy model. Its configurations therefore set
`energy_mode: interaction_direct`: isolated framework and CO2 baseline calls
return zero, while combined host-guest structures are evaluated by NequIP.
This matches the reference workflow and avoids subtracting unrelated baseline
energies a second time. No additional D3 correction is applied to this model.
The published checkpoint also distinguishes framework atoms from guest atoms
through model labels: physical CO2 is stored as O-C-O but evaluated as
Os-Co-Os. These are type labels, not chemical substitutions. The workflow
preserves the physical masses, uses physical O/C van der Waals exclusion
radii, and rejects a run unless the 276-atom ZIF-8 cell, cell dimensions, and
guest labels satisfy the configured model input contract.

The reproducibility archive is about 1 GB, but setup uses HTTP byte ranges to
read its ZIP directory and transfer only the roughly 1 MB
`model_ZIF8_N=1000.pth` member. The member CRC is verified and the resulting
SHA-256 is recorded below the ignored `external/` directory. If byte ranges are
unavailable, setup falls back to the complete archive and verifies its MD5.
It also installs the pinned legacy NequIP loader in the active environment:

```bash
bash scripts/run_zif8_nequip_reference.sh setup
```

The supplied configurations use CPU for local execution. On a GPU cluster,
set the model `device` to `cuda` in the configurations before running. The
legacy model mapping includes all seven exported types (H, C, N, O, Co, Zn, Os).
Run the gates in order. The smoke run checks model loading;
the 10,000-trial pilot checks sign, magnitude, sampling stability, and restart
files. Only then start the paper-length protocol (273 K, 100,000 insertions per
seed):

```bash
bash scripts/run_zif8_nequip_reference.sh smoke
bash scripts/run_zif8_nequip_reference.sh pilot
bash scripts/run_zif8_nequip_reference.sh paper
```

Three seeds are included for an uncertainty estimate; the cited MLIP-MC paper
uses 100,000 Widom insertion attempts for its reported protocol. Aggregate the
paper runs after synchronization:

```bash
python -m analysis.mlip_widom_replicates \
  outputs/module_C_potential_benchmark/runs/zif8_co2_nequip_finetuned_widom_100000_typed_v1_seed12345 \
  outputs/module_C_potential_benchmark/runs/zif8_co2_nequip_finetuned_widom_100000_typed_v1_seed23456 \
  outputs/module_C_potential_benchmark/runs/zif8_co2_nequip_finetuned_widom_100000_typed_v1_seed34567 \
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_nequip_reference
```

For a cheaper failure-mode check, compare the fine-tuned model against the
stored MACE no-D3/D3 energies only on the exported high-weight structures:

```bash
python -m analysis.mlip_reference_configuration_benchmark \
  --structures outputs/module_C_potential_benchmark/widom_zif8_co2_d3_energy_diagnostics/widom_d3_top_weight_configurations.extxyz \
  --model external/models/goeminne_zif8/model_ZIF8_N=1000.pth \
  --framework-atoms 276 \
  --temperature-k 273 \
  --device cuda \
  --loader legacy \
  --species-map identity \
  --supercell 2 2 2 \
  --supercell-count 5 \
  --output-dir outputs/module_C_potential_benchmark/widom_zif8_co2_reference_configuration_check
```

The supercell check detects a finite-cell or periodic-neighbourhood artefact.
The exported structures are deliberately selected from the dominant D3 tail,
so their MAE diagnoses that failure mode but is not an unbiased test-set MAE.

### Cluster-ready ZIF-8 benchmark

The complete corrected validation matrix can be started from the repository
root. Setup downloads and verifies the Goeminne and MACE-DAC checkpoints in
the active environment; smoke checks the model-specific atom typing;
production executes the original nine independent 100,000-insertion runs
sequentially on one GPU:

```bash
conda activate MOF_sim
bash scripts/run_module_c_zif8_benchmark.sh setup
bash scripts/run_module_c_zif8_benchmark.sh smoke
bash scripts/run_module_c_zif8_benchmark.sh production
```

After the GoldDAC screening, run the selected MACE-DAC model without an
additional D3 correction first as a smoke test and then with three independent
100,000-insertion seeds:

```bash
bash scripts/run_module_c_zif8_benchmark.sh dac-setup
bash scripts/run_module_c_zif8_benchmark.sh dac-smoke
bash scripts/run_module_c_zif8_benchmark.sh dac-production
```

`bash scripts/run_module_c_zif8_benchmark.sh dac` combines MACE-DAC setup,
smoke, and production. The `all` mode additionally includes this MACE-DAC
matrix.

After synchronizing the run directories back to the local repository, create
all replicate, convergence, and layered reference comparisons with:

```bash
bash scripts/analyze_module_c_zif8_benchmark.sh
```

The matrix contains three matched-seed runs for fine-tuned NequIP, MACE-MP-0a,
and MACE-DAC at 273 K, plus three fine-tuned NequIP runs at 298.15 K for the
experimental low-pressure comparison. The analysis script includes MACE-DAC
automatically once all three results are present. The source hierarchy and DOI
provenance are stored in `data/references/module_c_zif8_co2.json`.

The fine-tuning study by Kaur et al. supports the later method of starting from
MACE-MP-0 and adapting it with a small, high-quality data set. It does not
provide ZIF-8/CO2 adsorption labels. An adsorption-specific MACE model must
therefore be trained on framework-guest interaction energies and validated on
held-out configurations before it can replace the Goeminne reference model.
