# MOF_benchwork

Use one project environment for planning, LAMMPS, analysis, and MLIP-MC:

```bash
conda activate MOF_sim
```

## Module C with MLIP-MC

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
root. Setup downloads and verifies the Goeminne checkpoint in the active
environment; smoke checks the model-specific atom typing; production executes
the nine independent 100,000-insertion runs sequentially on one GPU:

```bash
conda activate MOF_sim
bash scripts/run_module_c_zif8_benchmark.sh setup
bash scripts/run_module_c_zif8_benchmark.sh smoke
bash scripts/run_module_c_zif8_benchmark.sh production
```

After synchronizing the run directories back to the local repository, create
all replicate, convergence, and layered reference comparisons with:

```bash
bash scripts/analyze_module_c_zif8_benchmark.sh
```

The matrix contains three matched-seed runs for fine-tuned NequIP and
MACE-MP-0a at 273 K, plus three fine-tuned NequIP runs at 298.15 K for the
experimental low-pressure comparison. The source hierarchy and DOI provenance
are stored in `data/references/module_c_zif8_co2.json`.

The fine-tuning study by Kaur et al. supports the later method of starting from
MACE-MP-0 and adapting it with a small, high-quality data set. It does not
provide ZIF-8/CO2 adsorption labels. An adsorption-specific MACE model must
therefore be trained on framework-guest interaction energies and validated on
held-out configurations before it can replace the Goeminne reference model.
