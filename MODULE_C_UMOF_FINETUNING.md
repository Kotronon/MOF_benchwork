# uMOF fine-tuning on the CPU cluster

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
