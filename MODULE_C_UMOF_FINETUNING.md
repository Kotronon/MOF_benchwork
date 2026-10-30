# uMOF MH / POLAR fine-tuning on the CPU cluster

## Select the model

Every lifecycle command accepts `--model mh` or `--model polar`. With no model
option, existing MH commands keep their previous defaults. An explicit custom
output directory with an existing campaign inherits its archived model; an
explicit conflicting model is rejected, including for status/setup/report/smoke.
`UMOF_MODEL=polar` is an alternative to the flag. `UMOF_CHECKPOINT` is the generic
checkpoint override; the old `UMOF_MH_CHECKPOINT` remains supported. Prefer explicit
flags when old `UMOF_*` shell variables are present.

```bash
export PYTHON_BIN=python
export UMOF_CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# On a network-enabled environment host, if dependencies are missing:
python -m pip install -r requirements-module-c-umof.txt

# Choose mh or polar. Only setup --download-missing may download workflow assets.
bash scripts/run_module_c_umof_finetuning.sh setup --model polar --download-missing
bash scripts/run_module_c_umof_finetuning.sh prepare --model polar
bash scripts/run_module_c_umof_finetuning.sh status --model polar

# Run production only inside the cluster allocation:
bash scripts/run_module_c_umof_finetuning.sh train --model polar
bash scripts/run_module_c_umof_finetuning.sh resume --model polar
bash scripts/run_module_c_umof_finetuning.sh evaluate --model polar
bash scripts/run_module_c_umof_finetuning.sh report --model polar
```

For MH, replace `polar` with `mh`. Both profiles use CPU/float64, batch size 1,
learning rate 1e-4, seed 12345, a maximum of 50 epochs and SWA start 40. Each profile
has four sequential runs. Running both profiles means eight training runs.

### CUDA training with CPU RAM offload

For a new GPU campaign use `--device cuda --output-directory outputs/umof_polar_cuda`.
The adapter automatically offloads saved autograd tensors to CPU RAM during CUDA
training, including the derivative graph required by force training. Model
parameters and optimizer state stay on the GPU. Float64, labels and the training
protocol are unchanged. This trades host RAM and CPU/GPU transfer time for lower
GPU memory use; it is not disk swap and cannot guarantee that every structure fits.
CPU training is unaffected. Set `UMOF_CPU_OFFLOAD=0` to disable offload when there
is sufficient GPU memory. Offload activation is logged and recorded in checkpoints.
For a measured small-graph fast path, `UMOF_GPU_ONLY_MAX_EDGES=8000` keeps batches
with at most 128 atoms and 8000 edges entirely on the GPU; larger batches still
offload. The default is zero (offload every CUDA training batch). This threshold
is hardware/workload dependent; it is not a GPU memory reservation.

```bash
CUDA_VISIBLE_DEVICES=0 UMOF_CPU_OFFLOAD=1 PYTHON_BIN=python \
  bash scripts/run_module_c_umof_finetuning.sh resume --model polar \
  --device cuda --output-directory outputs/umof_polar_cuda
```

### Short Float32 pilot

`--dtype float32` is archived with the protocol and used consistently for
training, candidate selection, both baseline and tuned test predictions, cache
identities and the exported model specification. Existing campaigns default to
Float64. A precision change requires a separate output directory. `--start-swa`
sets the archived stage-two epoch for a short schedule; it cannot change on resume.

For first results rather than a complete four-fraction convergence study:

```bash
CUDA_VISIBLE_DEVICES=0 UMOF_CPU_OFFLOAD=1 UMOF_GPU_ONLY_MAX_EDGES=8000 PYTHON_BIN=python \
  bash scripts/run_module_c_umof_finetuning.sh train --model polar \
  --device cuda --dtype float32 --fractions 0.25 --max-epochs 5 --start-swa 3 \
  --replay-samples 200 --output-directory outputs/umof_polar_24h_float32
```

This uses 211 adsorption training configurations, 200 replay configurations,
and the original adsorption validation/test sets. It runs one seed and fraction.
The smaller replay pool and shorter schedule change the training protocol;
results are preliminary and do not establish convergence or retention on the
full replay distribution. Baseline improvement is measured, not assumed.
The workflow automatically writes selection, held-out baseline/tuned metrics,
parity plots and the model specification after training. The actual wall time
depends on graph sizes, offload transfers and concurrent GPU use.

| Profile | Default campaign directory | Replay reference |
|---|---|---|
| `mh` | `outputs/module_C_umof_finetuning_matrix_v2` | MatPES r2SCAN |
| `polar` | `outputs/module_C_umof_finetuning_matrix_polar_v2` | Published uMOF r2SCAN-D4 |

The adsorption geometries, labels, splits and nested selections are shared:
109/211/398/774 training configurations, 46 validation and 38 test configurations.
POLAR adds explicit charge/spin conditioning metadata without changing DFT labels.
No new DFT calculations or mixture labels are generated. Model checkpoints,
prediction caches, replay selections, training state and exports remain separate.
Existing MH production results and checkpoint identities are preserved.

## POLAR replay and conditioning

The inspected released `uMOF-POLAR.model` is a `PolarMACE` with **one** head,
`r2scan+d4`; it does not contain the original OMol replay head. The user-approved
POLAR recipe therefore creates a new `pt_head` for retention of published uMOF
knowledge. It does **not** claim to restore OMol replay or reproduce the original
paper's complete training recipe.

Its source is `qmof_r2scan_gaussian_noise_train.extxyz` from `umof_data.zip`, SHA-256
`310d9267720430280f2cf2a9448ec96d87e9df970e1908fa23925971eb517463`, checked against
the local published archive member. Both heads use r2SCAN-D4 energy/force labels
and foundation atomic energy references. Select 1,000 distinct replay training
configurations and 100 disjoint replay validation configurations, weight 0.25,
once per seed for all fractions. Unsupported elements are excluded. Reduced
framework compositions occurring in any prepared adsorption split are excluded
conservatively, including supercells. This filter uses geometry/composition only,
never test energy/force labels. It may exclude unrelated isomers too and does not
establish independence from foundation pretraining. Replay manifests archive the
source checksum, selected offsets, geometry IDs and excluded compositions.

POLAR requires `graph_longrange==0.4.0`. Source `total_charge`/`charge` and
`total_spin`/`spin` values must agree and be finite; spin denotes positive integer
multiplicity. Missing values explicitly use MACE defaults: charge 0, multiplicity
1. Their provenance is stored as an assumption, **not** inferred quantum-chemical
spin labels. Charged/open-shell transfer accuracy remains unvalidated. Training
and calculator aliases carry the same values. No dipole, charge, stress or virial
targets are added to this energy/force training.

Exports include `model_type`, actual adsorption head, dtype and checksum, so the
Module-C backend automatically uses the appropriate calculator. Stage-1/SWA
selection uses adsorption validation; final test predictions compare with the
unchanged foundation of the **same** profile. Resume across profiles is forbidden.

## Technical checks and transfer

```bash
bash scripts/run_module_c_umof_finetuning.sh smoke --model polar
bash scripts/run_module_c_umof_finetuning.sh smoke --model mh \
  --smoke-directory outputs/module_C_umof_finetuning_smoke_profiles_mh_v3
```

Each smoke uses 9/3/3 real adsorption configurations, 8/2 real replay
configurations, two epochs followed by resume to three, with a fixed SWA schedule.
It verifies both heads, weight changes, finite losses, exports, baseline evaluation
and Module-C backend loading. `smoke_result.json` is written only after all checks
pass. Use a new smoke directory when the source campaign changes.

Transfer the full selected campaign directory, including `assets/`, plus the
updated repository code. `START.txt` contains model-specific commands relative to
the repository. An already prepared campaign can train/resume/evaluate offline
without the full external data archives. Its environment and requirements lock
include POLAR's electrostatics dependency when installed. Do not copy the local
temporary test dependency path into cluster commands.

The following sections document the original MH protocol and its compatibility
behavior; MatPES-specific statements apply to **MH only**.

The prepared v2 campaign fine-tunes uMOF-MH on published CO2, N2 and CH4 adsorption configurations. No CP2K calculations, classical mixture energies or flexible Module-D simulations are used. Production training is intentionally left for the cluster. Existing results are retained, including the previous v2 campaign in `outputs/module_C_umof_finetuning_matrix_v2_archive_20261023`.

## Offline transfer and launch

Transfer the repository with the new Python modules, the shell wrapper and the **entire** `outputs/module_C_umof_finetuning_matrix_v2` directory. This includes its foundation model and selected replay files under `assets/`; the full external archives are unnecessary for training an already prepared campaign. Do not transfer only the generated command JSON. It records arguments for inspection; the wrapper reconstructs runnable absolute paths in each run's own working directory.

Activate the cluster environment containing the versions recorded in `environment.json` and `requirements-lock.txt`. The lock file records the directly used packages; `environment.json` also records the full installed Python package inventory. Build the environment on a network-enabled login node before entering an offline compute allocation. Set `PYTHON_BIN=python` inside that environment; generated commands contain no developer-specific Python path.

```bash
export PYTHON_BIN=python
export UMOF_OUTPUT_DIRECTORY=outputs/module_C_umof_finetuning_matrix_v2
export UMOF_CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"
bash scripts/run_module_c_umof_finetuning.sh status
bash scripts/run_module_c_umof_finetuning.sh train
```

The four runs execute sequentially. Run the command inside the site's scheduler allocation. This project does not assume a partition, account or walltime. To continue after interruption, or rerun only missing evaluation work:

```bash
bash scripts/run_module_c_umof_finetuning.sh resume
bash scripts/run_module_c_umof_finetuning.sh evaluate
bash scripts/run_module_c_umof_finetuning.sh report
```

`resume` skips finished training and resumes unfinished exports/selection/evaluation. `evaluate` never launches training. An increased epoch limit is accepted only with unchanged archived learning-rate and SWA settings, e.g. `resume --max-epochs 60`; the prescribed first matrix remains capped at 50. Explicit conflicting options are rejected. With no override, custom gas definitions and MOF splits are inherited from the manifest. Do not leave unrelated `UMOF_*` overrides in the shell environment.

The wrapper limits OMP/MKL/OpenBLAS threads using `UMOF_CPU_THREADS`, then `SLURM_CPUS_PER_TASK`, `PBS_NP`, `NSLOTS`, `OMP_NUM_THREADS`, then 1. A campaign-wide advisory writer lock prevents simultaneous writers. Status and checkpoint sidecars are written atomically. Filesystem permissions are respected; training requires writable campaign/run directories. Prepared source files can be read-only. A checkpoint with missing or changed identity/hash metadata is rejected rather than silently restarting.

## Recreating assets and preparation

Only setup can download. It can extract local release archives without networking. Preparation and training never resolve remote model names or replay aliases.

```bash
bash scripts/run_module_c_umof_finetuning.sh setup --download-missing
bash scripts/run_module_c_umof_finetuning.sh setup
bash scripts/run_module_c_umof_finetuning.sh prepare
bash scripts/run_module_c_umof_finetuning.sh smoke \
  --smoke-directory outputs/module_C_umof_finetuning_smoke_v2_verified
```

`setup` checks the foundation heads, the audited MatPES release SHA-256, adsorption source files and Python environment. `prepare` archives all inputs required by the campaign and creates commands without training. Preparing an existing campaign verifies its manifest; incompatible settings require a new output directory. The old API functions and flag-only CLI remain available.

## Fixed first-run protocol

| Setting | Value |
|---|---|
| Foundation / head | uMOF-MH / `r2scan+d4` |
| Adapted adsorption / replay heads | `Default` / `pt_head` |
| Compute / dtype / batch | CPU / float64 / 1 |
| Learning rate / seed | 0.0001 / 12345 |
| Requested nested fractions | 10 / 25 / 50 / 100% |
| Actual selected configurations | 109 / 211 / 398 / 774 |
| Actual fractions | 14.08 / 27.26 / 51.42 / 100% |
| Validation / test configurations | 46 / 38 |
| Replay train / separate validation | 1000 / 100 |
| Replay configuration weight | 0.25 |
| Epoch cap / SWA start | 50 / 40 |
| LR scheduler | ReduceLROnPlateau, factor 0.8, patience 50 |
| MACE stopping patience | 20 |

The actual fractions exceed small requested fractions because every populated MOF/component/loading stratum retains at least one configuration. Validation MOFs are HKUST-1 and MIL-53-Al; test MOFs are MOF-303 and MgMOF-74. Test labels are never passed to MACE training or checkpoint selection. This is held-out fine-tuning transfer evaluation: these MOFs may already have appeared in foundation pretraining.

MatPES retains its own **r2SCAN DFT** labels from `dft_energy`/`dft_forces`; stored predictor outputs are ignored. Adsorption labels are **r2SCAN-D4**. The two references remain on separate heads. For local replay files MACE uses the per-configuration `config_weight=0.25`; `weight_pt_head` alone only controls the built-in selection path. No duplicate padding or implicit adsorption replication is used. The adapter preserves both heads and checks strict restart loading, checkpoint hashes, RNG and SWA state. It resumes after the completed epoch. This does not claim bitwise equivalence to an uninterrupted MACE run (MACE recreates its validation/early-stopping bookkeeping).

The [MACE training implementation](https://github.com/ACEsuit/mace/blob/develop/mace/cli/run_train.py) and the installed 0.3.16 code were checked. The local `prepare_pt_head` short-name branch replaces MatPES aliases by `mp`, which is why explicit local replay files are mandatory here. No installed MACE package was modified.

## Future gas and mixture inputs

`--gases CO2,N2,CH4` controls allowed stable component IDs. Additional definitions are supplied with `--gas-definitions gases.json`, a mapping from component ID to elemental counts, for example `{"Ar":{"Ar":1}}`. The checkpoint must support every element in each selected definition and frame. Element support does **not** establish model accuracy. Isomers require distinct IDs and explicit atom assignments; matching elemental formulas alone are ambiguous.

Each future co-loaded frame needs its own consistent DFT energy and complete force labels. There is no construction of mixture labels by adding pure-gas energies. Prefer explicit, zero-based atom metadata:

```json
{
  "umof_framework_indices": [0, 1, 2],
  "umof_guest_assignments": [
    {"component_id": "CO2", "atom_indices": [3, 4, 5]},
    {"component_id": "N2", "atom_indices": [6, 7]}
  ]
}
```

The example defines the metadata schema, not a production structure. Every atom must occur exactly once, either in the framework or in one molecule. Molecule identities must match their declared formulas. Legacy connectivity recognition remains available for released data; unrecognized or ambiguous fragments are listed in the exclusion manifest. Contradictory counts, compositions, MOF IDs or reference labels fail preparation. A configuration containing an excluded component is excluded as a whole.

Prepared frames record component IDs, integer molecule counts, adsorbed molecule fractions, pure/mixture kind, MOF ID, assignment provenance, source frame index and geometry ID. Adsorbed fractions are never interpreted as gas-phase mole fractions. Sampling uses MOF, sorted component set and total loading classes (1, 2–3, 4+). Mixtures add 20-percentage-point composition bins, with 100% in the final bin. Component order does not affect selection.

Existing stress arrays and convention metadata are retained. An unspecified source convention stays unspecified. Stress and virial labels are explicitly disabled in this energy/force training.

## Reports and Module D boundary

`coverage.json` records available gases, complete component sets, loading counts and distinct source cells per MOF. All currently selected frames are pure-gas configurations. Different cells demonstrate structural variation but do not identify or validate a framework deformation pathway. Mixture validation and flexibility validation both remain open.

Per-run `results/selection.json` compares Stage 1 and, when exported, SWA/Stage 2 on adsorption-validation force RMSE, then energy RMSE, then prefers Stage 1 on ties. Only after selection does `results/evaluation.json` compare the chosen model with the unchanged foundation on the test split. Prediction caches are keyed by dataset/model checksums, head, dtype, device and MACE version. Metrics cover whole configurations, including mixed-component groups as separate groups, in eV/atom and eV/Å.

CSV/JSON summaries, PNG/SVG parity plots, learning curves, data-fraction plots and runtime comparisons are created by evaluation/report. Logs are `training.log`, `export.log`, `selection.log` and `evaluation.log`. The exported `module_c_model.json` contains a relative checkpoint path, SHA-256, actual head and dtype; resolve and verify it with `load_model_specification` before using the Module-C calculator. No production model registry is changed automatically.

The next Module-D step is independent validation on genuinely co-loaded CO2/N2 configurations and relevant framework deformations, followed where necessary by training on consistent quantum-chemical labels. Neither the technical smoke test nor pure-gas transfer errors replace those validations.
