# Module D: flexible frameworks

The new [general profile](docs/module_d_general.md) supports arbitrary material
inputs with a common classical/MLIP potential interface. Classical NVT/NPT MD/GCMC
is executable, with a [MIL-53(Al)/CO₂ technical pilot](docs/module_d_mil53_cluster.md).
MLIP execution remains gated. See that guide for templates, required force-field
inputs, smoke tests and cluster commands. This is not universal MOF parameterization.

The [full MIL-53/CO₂ isotherm campaign](docs/module_d_mil53_isotherm_cluster.md)
is configured for 300 K, ten pressures from 0.01 to 10 bar, three variants and
three seeds (90 independent cases). Each case uses a 50 ps warmup and a 100 ps
observation budget. Execution uses the existing repository checkout and a Slurm
array template; no distribution archive is needed. Parameter assets are shared
between jobs through relative links. No production jobs have been submitted.

The [MIL-53 cluster pilot status and next checks](docs/module_d_mil53_cluster.md#stand-nach-dem-clusterpilot-14112026)
record nine technically complete runs, with only one passing the stationarity
diagnostic. Mean pilot loadings are 2.00 mol/kg (rigid), 2.03 mol/kg (flexible NVT)
and 1.46 mol/kg (flexible NPT); these are not established equilibrium values.
The [NPT control review](reports/module_d_npt_review_v2_2026-11-15/README.md)
now covers four complete 5.5 ps runs for one seed: CO₂/dry × 0.25/0.125 fs.
The dry framework also relaxes substantially (mean volume increase about 3.4–3.5%);
all four runs fail the RMSD stationarity diagnostic. Timestep convergence remains
unproven. Higher-precision CO₂ output was included in the subsequent longer
controls below; numerical validation and agreement across additional seeds remain open.

The subsequent [20 ps control review](reports/module_d_npt_relaxation_review_2026-11-16/README.md)
covers two complete LAMMPS runs at 0.25 fs with seed 12345. Mean volume changes
over 5–20 ps are +3.362% with CO₂ and +3.506% dry; mean CO₂ loading is
1.666 mol/kg. Both cases still fail the RMSD window diagnostic, and the CO₂
case also fails the loading diagnostic. Precise dumps retain linear CO₂ but
show pair-distance deviations up to 5.22×10⁻⁵ Å in saved frames. These remain
exploratory results; no Module-D FLAMES run has been performed.

## NbOFFIVE-1-Ni water paper profile

The separate paper profile plans the NbOFFIVE-1-Ni (KAUST-7)/H2O comparison and evaluates
imported sorption-relaxation (SR) cycle measurements. It prepares an independent
workspace while Module A–C calculations continue. Both MD and GCMC target
LAMMPS; RASPA/gRASPA is not an execution dependency. It does **not** yet run
the coupled simulation, reproduce an isotherm, or validate a potential.

The selected starting point is Wang et al., *J. Phys. Chem. C* **130** (2026),
2833–2846, [DOI 10.1021/acs.jpcc.6c00023](https://doi.org/10.1021/acs.jpcc.6c00023).
This follows the Module D section of
[the project literature review](<guidelines_for_project_developement/Literature research for MOF simulations.docx>).
The alternative MOF-5 entry in the progress report is deferred.

## Prepare the campaign

From the repository root, using the existing environment:

```bash
conda activate MOF_sim
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --dry-run
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare
```

The default directory is
`outputs/module_D_nboffive_water/runs/planning_lammps_v1/`. The plan contains 60 cases:
10 relative-humidity points × 3 independent seeds × rigid/SR-NVT branches.
The rigid branch expects one GCMC summary per case; the flexible branch expects
100 SR cycle summaries per case. These are planned computations, not results.

The bundle contains:

- `module_d_plan.json`: conditions, cases, parameters, file hashes and outstanding prerequisites.
- `cases.csv`: stable case IDs and their thermodynamic conditions.
- `cycle_results.template.csv`: a header-only measurement schema.
- `upstream_reference/`: original, checksum-verified MOFAFF example inputs and their license.
- `README.txt`: import instructions and the current execution boundary.

All paths in the input JSON are resolved relative to the working directory,
consistent with the existing benchmark CLI. `--dry-run` creates no output.
Module D commands skip global registry regeneration, including when
`--skip-registry-update` is omitted. No command here downloads assets, installs
packages, trains a model, or starts MD/GCMC.

An existing directory is never overwritten, including with `--overwrite`.
To verify and reuse the same preparation:

```bash
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare --resume
```

Resume compares the plan, asset hashes, archived upstream inputs and case table.
For changed settings or a new campaign, use a new run ID:

```bash
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare --run-id planning_v2
```

## Current cluster entry point

For the paper profile, only preparation and imported-result analysis can currently be started on the
cluster. There is no working paper-profile production submission script. A scheduler
allocation cannot enable the missing simulation adapter.

Transfer the current repository files, including `data/module_d/`, to the cluster.
From that checkout, activate the project's Python environment (`MOF_sim` if
installed there) and run:

```bash
conda activate MOF_sim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --dry-run
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare --run-id cluster_lammps_v1
```

This creates `outputs/module_D_nboffive_water/runs/cluster_lammps_v1/`.
Use the same `--run-id cluster_lammps_v1` with `--prepare --resume` to verify
and reuse that bundle. Preparation needs no GPU or LAMMPS installation; use a
small CPU allocation if required by the cluster's login-node policy. No jobs
are submitted by these commands.

The earlier `planning_v1` bundle retains the old paper-engine plan. The new
LAMMPS plan uses a different run ID so that existing bundles remain immutable.

## Paper protocol and project choices

The paper profile's scope is the NVT branch at 300 K. Its coupled adapter, NPT,
mixtures and arbitrary structural-state ensembles are not implemented. Classical
NVT execution is provided separately by the general profile above.

| Parameter | Initial setting | Basis |
| --- | --- | --- |
| GCMC water model | Rigid four-site TIP4P | Paper §2.4 |
| Framework charges | DDEC06 | Paper §2.4–2.5; atom-resolved values still required |
| MD potential | DeePMD model trained across water loadings | Paper §3.4; checkpoint still required |
| MD per SR cycle | 1000 steps × 1 fs = 1 ps, NVT | Paper §2.3, §2.5 |
| Thermostat damping | 0.04 ps | Paper §2.3 |
| GCMC per SR cycle | 200,000 equilibration + 500,000 production steps | Paper §2.4–2.5 |
| SR iterations | 100 | Paper §2.5 |
| Discarded SR iterations | 0 | Paper average over all iterations |
| LJ cutoff | 9.7 Å | Original GCMC input / paper §2.4 |
| Electrostatics | Ewald, precision 1e-6 | Original GCMC input / paper §2.4 |
| Seeds | 12345, 23456, 34567 | Project replication choice |
| RH grid | 1, 2.5, 5, 7.5, 10, 20, 40, 60, 80, 100% | Project starting grid, not SI Table S2 |
| Stationarity windows | Last two non-overlapping windows of 10 cycles | Project diagnostic |
| Drift thresholds | 0.05 mol/kg uptake; 0.02 Å RMSD; 1% cell volume | Project diagnostic, not paper acceptance criteria |

The intended SR handoff is loaded structure → MLIP-MD → final framework and
water configuration → TIP4P geometry projection → classical GCMC → loaded
configuration for the next MD segment. Both framework and guest configuration
must be transferred with stable atom identities. Independently generated cases
must not inherit another RH point's final configuration without an explicit
continuation protocol.

The shared `simulation` block also receives legacy pipeline defaults during
normalization. Module D does not render those defaults as a flexible LAMMPS
potential. The intended implementation uses separate LAMMPS stages: DeePMD
for MD and classical TIP4P/UFF interactions for GCMC. Unit conversion must be
explicit: the original MD input uses `metal` units; the source gRASPA pressure
is in Pa, while LAMMPS `real` units use atm. Original gRASPA files remain
literature references only. The initial reference
structure is already a supercell; `[2, 2, 2]` describes its provenance and is
**not** an instruction to replicate that file again.

### Relative humidity and fugacity

Use RH as a fraction, e.g. `0.8` for 80%. The paper's TIP4P saturation pressure
at 300 K is 5197 Pa = 0.05197 bar, so `pressure_bar = RH * 0.05197`.
The experimental saturation reference is 3600 Pa = 0.036 bar; it must not be
substituted into the simulated reservoir. The current profile rejects other
temperatures and other water models because this conversion is specific to the
paper's parameterization. This is a reference conversion, not a validated
LAMMPS calibration. Reuse of the paper reservoir requires validation of the
LAMMPS water Hamiltonian, cutoff and tail-correction treatment.

The paper obtains the pressure-to-fugacity relation from TMMC. The planning
config leaves `reservoir.fugacities_bar` unset until that relation/SI Table S2 is
available. Pressure is never silently substituted for fugacity. Explicit
fugacities require one value per RH point and a `fugacity_source` citation.
Neither the upstream example's `Pressure` value nor its coefficient of one is
sufficient to reconstruct the complete experimental RH grid or reservoir table.

### Original inputs and unresolved assets

Original example files are retained under `data/module_d/mofaff_reference`,
from [MOFAFF](https://github.com/xwsci/mofaff), commit
`7c8ddb138b5cda0a53c55b38906261d998397e47`. `provenance.json` records SHA-256
checksums; the upstream BSD-3-Clause license is included. The example files are
reference material, not prepared executable cases.

The global LJ tail setting is `no`, but `force_field.def` explicitly overrides
`Ow Ow` to `truncated yes`. Thus the initial suspicion that the input omitted
water–water tails was incorrect. All pair overrides must be retained by the
future adapter.

The supplied `conf.lmp.initial` contains 1184 atoms. Using the upstream type
mapping and its count of 32 framework O atoms gives 96 water molecules in the
supercell (12 per cell for the stated 2×2×2 construction). Its structure and atom
ordering still require scientific validation against the intended starting
state; this count does not prove that it is the exact published optimized cell.

Required next inputs are represented explicitly in `benchmark.flexibility.assets`:

| Asset | Purpose and remaining work |
| --- | --- |
| `initial_structure` | Upstream hydrated supercell is present; validate exact provenance and atom mapping. |
| `rigid_framework_cif` | Obtain the intended rigid reference structure with consistent cell and atom identities. A guest-stripped hydrated structure is a different structural baseline from the dry optimized framework. |
| `charges` | Obtain the atom-resolved DDEC06 charges and map them to framework IDs. The zero framework charges in `pseudo_atoms.def` are not replacements; the example reads charges from CIF. |
| `mlip_model` | Select the variable-hydration DeePMD checkpoint used for SR. Do not substitute a dry-only or fixed-12-water model. |
| `mlip_validation` | Record model identity, element/type order, hydration and deformation coverage and validation evidence. File presence alone is not validation. |
| `experimental_reference` | Obtain numerical water uptake at 300 K with units, adsorption basis, RH definition and source. |
| `gcmc_parameters_validation` | Reconcile the complete force-field settings and reservoir conversion with the paper and SI. |

The paper links its model/data archives at
[Zenodo record 15750252](https://zenodo.org/records/15750252). The variable-loading
archive is `DP_NbOFFIVE-1-Ni-xH2O.zip` (published MD5
`f3dccedc2d4258dffd0f5aaca5fb914c`, about 2.5 GB). It has not been downloaded by
this preparation. The other archives target dry and fixed-hydration systems.
After retrieval, inspect checkpoint metadata and training coverage before use.

The future execution adapter must check LAMMPS with DeePMD, MC and the required
electrostatics support,
apply per-case seeds, check subprocess exit codes and preserve failure/restart
state. It must validate water geometry projection, charge mapping, guest counts,
cell transfer and units before a real short test. `--run-test` and
`--run-isotherm` currently fail explicitly, including with `--allow-unsupported`.
Module-C pure-gas transfer results do not validate this flexible water system.

The GCMC target is LAMMPS `fix gcmc` with full-system energy evaluation. Water
geometry and molecule IDs must remain consistent through insertion/deletion and
the MD/GCMC handoff. The paper's configurational-bias moves cannot be equated
with the native LAMMPS move set; sampling efficiency must be checked separately.
See the official [GCMC documentation](https://docs.lammps.org/fix_gcmc.html) and
[TIP4P guidance](https://docs.lammps.org/Howto_tip4p.html). The local capability
check found MC and Ewald support but no `pair_style deepmd`; the cluster build
must be checked independently. These are implementation prerequisites, not a
claim that the LAMMPS runner already exists.

## Analyze measured cycles

Create `cycle_results.csv` in the prepared directory using the template header.
Use the `case_id` values from `cases.csv`. Each case must have consecutive,
1-based cycle numbers; duplicate, unknown, skipped, excessive or non-finite
records are rejected. A partial prefix can be analyzed, but is reported as
partial and is excluded from the rigid/flexible comparison until complete.

Required columns are `case_id`, `cycle`, `uptake_mol_kg`,
`framework_rmsd_A`, and `cell_volume_A3`. Loading is absolute mol/kg of **dry
framework**, not kg of the hydrated system. Structural measurements refer to
framework atoms only. A rigid-case record should contain its measured loading,
zero framework displacement relative to its own reference, and its actual cell
volume; do not invent missing SR geometry measurements.

Optional fields are potential energy in eV, accessible pore volume in Å³,
a consistently defined linker angle change in degrees, and elapsed time in
seconds. Empty optional fields remain missing. Energy and pore-volume means
are reported only when measured for every retained cycle. Linker angles are
retained in the raw records; no linear average is formed across periodic angles.

```bash
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --evaluate
```

The command uses the archived plan and writes `evaluation/module_d_report.json`
and `evaluation/rigid_vs_flexible.csv`. Raw measurements and the archived plan
are not modified. The comparisons pair the same RH and seed and report
`flexible - rigid`; relative change is undefined (`null`) for zero rigid uptake.
The stationarity result accompanies each comparison, including failed checks.

`structural_metrics` in `analysis/module_d_flexibility.py` computes framework
RMSD using minimum-image displacements in the reference cell, removes affine
cell deformation, and reports the cell-volume change separately. It does not
fit a rotation or remove global translation. Supply a stable framework atom
mapping; species equality cannot verify the order of identical elements.
Explicit four-atom linker definitions yield signed periodic dihedral changes.
Accessible pore volume requires a separate pore analysis and is never inferred
from cell volume.

A complete report means that all planned rows were imported. It does not mean
the simulation is equilibrated or the model validated. SR cycle standard
deviation is not a confidence interval for independent replicates. The two-window
check is only a stationarity diagnostic. Experimental-reference scoring,
replicate confidence intervals, plots, trajectory ingestion and simulation
execution remain future work.

## Fast checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
  tests.test_module_d_flexibility tests.test_module_registry -v
```

These checks use analytical structures and temporary measurement fixtures. They
launch no MD, GCMC or training jobs and do not create scientific benchmark data.
