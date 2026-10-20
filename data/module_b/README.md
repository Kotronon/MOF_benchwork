# Module B: general mixtures in rigid Module-A frameworks

See [completed checks and remaining scientific validation](VALIDATION.md).

Module B shares `MaterialResolver`, forcefield resolution, `prepare_benchmark`
and `materialize_benchmark` with Module A. It accepts the same supplied CIFs,
aliases, available charge schemes, cell representations and framework forcefields.
Existing geometry, cutoff and parameter checks still apply. Flexible frameworks
and changing cells belong to Module D and are outside this implementation.
Catalog parity is a preparation-path check, not 1,357 completed simulations.
For example, the local `NEUTRAL/ZIF-8.cif` has no `_atom_site_charge` records
and is rejected by the common converter; Module B does not silently invent charges.

## Gas models and reservoir

`adsorbates.components` contains stable, distinct IDs (letters, digits,
underscores; normalized to uppercase). `adsorbates.mixture` assigns every ID a
finite, nonnegative mole fraction summing to one. For example:

```json
{"adsorbates": {"components": ["CO2", "CH4", "N2", "AR"],
  "mixture": {"CO2": 0.4, "CH4": 0.3, "N2": 0.3, "AR": 0.0}}}
```

Definitions can be selected explicitly with
`simulation.forcefield.definitions: {"MY_GAS": "path/model.json"}`.
Legacy RASPA `.def` files and JSON molecule definitions are supported; see the
JSON definitions in `models/`. Geometry uses Å, masses u, charges elementary
charge, critical pressure Pa and critical temperature K. Pseudoatom bundles
provide masses/charges and LJ parameters (epsilon K, sigma Å). Bonded terms
use LAMMPS `real` units. `simulation.forcefield.parameter_files` supplies
additional `{source, files: {pseudo_atoms, mixing_rules}, atom_types}` bundles.
Only explicitly scoped or required types are merged; conflicting definitions
fail. Use distinct type IDs for distinct models. Adding another gas must not
silently replace an existing gas's parameters. Missing parameters stop the run.

Component-specific LAMMPS type IDs keep molecule counts separate even when gases
share a pseudoatom. Source assets and SHA-256 checksums are archived with each
prepared run; campaign states also record physical identity and package versions.
Use model `schema_version` and source descriptions in custom model files.

Multicomponent Peng–Robinson computes `f_i = y_i P phi_i(T,P,y)` using each
definition's critical parameters. Optional symmetric binary interactions are
passed as `simulation.reservoir.binary_interactions` (pair keys such as `"CO2/CH4": 0.0`).
The vapor root is the default; `root: "liquid"` must be explicit. Root selection
does not perform phase-stability/flash calculations. Explicit component
fugacities in bar can be supplied as `simulation.reservoir.fugacities_bar`; all
components must be mapped, with zero fugacity exactly for zero gas fraction.
No componentwise pure-gas EOS fallback is used for mixtures.

## Flexible guests

Map flexible IDs to model files through `adsorbates.models`. Molecule definitions
provide atom positions/types; model files provide a rooted growth tree,
reference springs, physical bonded terms, intramolecular exclusions and the
ideal Rosenbluth normalization at each requested temperature. Supported bonded
styles are harmonic bonds/angles/impropers and OPLS dihedrals. Atom indices in
model topology are zero-based. Flexible rings and constrained flexible models
are rejected. The proposal interface (`draw`, `reference_energy`, `log_ideal`)
is the extension point for future topologies.

The multiple-try CBMC implementation proposes entire linear or branched trees
from normalized Gaussian reference bonds `U_ref = sum(k r²)`. It includes
insertion, deletion and regrowth with consistent reverse trial sets. The supplied
dimensionless normalization is
`log E_ref[exp(-beta (U_intra - U_ref))]`; a documented value is mandatory at
the actual temperature. This is not an arbitrary tuning parameter. The physical
intramolecular Hamiltonian and exclusions must match its derivation.

LAMMPS computes physical energies and forces for every trial. Python only
orchestrates proposals and acceptance. This correctness-oriented backend rebuilds
topology per trial and is substantially slower than the native rigid GCMC path.
It currently requires one common `special_bonds` convention across flexible
species. Mixed multisite rigid guests require a zero 1–2 weight. MPI/threaded
CBMC pressure-point execution is not supported.

Optional `simulation.sampling.hmc` contains `steps`, `every` and `step_size`.
Leapfrog updates flexible guest coordinates only and uses a Metropolis correction.
`step_size` is a fictitious integration parameter, **not a time in femtoseconds**.
Framework coordinates/cell and rigid guest internal coordinates never move in
these steps. CBMC checkpoints include molecules, RNG state, counters and samples.

`gaussian_linear` and `gaussian_branched` are synthetic analytic test models.
They deliberately are not claimed to represent physical gases. Their normalization
is exactly known because physical and reference intramolecular energies coincide.

## General campaign workflow

Run commands from the repository root with the core requirements, the LAMMPS
executable/Python API and `requirements-module-b-al.txt` installed.

```bash
# Configuration and requests only; no simulation starts here.
python -m modules.module_b_campaign init CONFIG.json CAMPAIGN/state.json --budget 24
python -m modules.module_b_campaign propose CAMPAIGN/state.json

# Explicit bounded execution, only when a simulation campaign is authorized:
python -m modules.module_b_campaign run CAMPAIGN/state.json --max-requests 1

# Alternatively run a generated request through the existing benchmark CLI,
# then import its completed summary:
python -m modules.module_b_campaign ingest CAMPAIGN/state.json RUN/isotherm_summary.json
python -m modules.module_b_campaign report CAMPAIGN/state.json
```

The default pool is 4,096 scrambled Sobol points (seed 2026): logarithmic pressure,
uniform Dirichlet(1) composition coverage and optional linear temperature.
Pressure bounds come from `conditions.pressures_bar`; optional temperature bounds
come from `conditions.temperatures_K`, otherwise `temperature_K` stays fixed.
Explicit point lists are accepted by the Python `initialize(..., candidates=...)`
API. Each point has `pressure_bar`, `temperature_K` and full `mixture`.

Noise-aware GPs use log1p component loadings and seed standard errors. Acquisition
maximizes the largest component-relative latent interval, with a molecule-count
floor near zero. Zero-fraction components are excluded from that score and their
predicted loading is zero. Selectivity involving a zero fraction is undefined.
The total training budget counts distinct states; precision retries (one by
default, with four times the steps) replace an observation and remain in history.
`--max-requests` separately bounds executions, including retries. A pending request
is idempotent; changed assets or inconsistent imports fail before reuse.

Holdout coordinates are reserved at initialization. `propose --holdout` and
`run --holdout --max-requests N` process these separately. Their labels never
enter fitting, acquisition or stopping decisions. Reports separate surrogate
errors against LAMMPS holdouts from external physical validation. The statuses
`pac_target_met`, `surrogate_uncertainty_target_met`, `budget_exhausted`,
`execution_budget_exhausted`, `simulation_precision_unmet` and `technical_error`
have distinct meanings; none implies paper reproduction or equilibration.

Reports contain component/total molecule counts, all unordered selectivity pairs,
seed intervals, surrogate CSVs, projection plots and a holdout learning curve.
Higher-dimensional projections do not represent complete adsorption surfaces.
Cross-component GP covariance is not modeled; ratios of GP predictions have no
claimed joint confidence interval.

## Six paper cases

The algorithm profiles are pinned to
[upstream commit 0415e03](https://github.com/mukherjee07/Active-Learning-for-multicomponent-adsorption-in-a-MOF/tree/0415e03f2a67657cf88d8964ad665177c54bbde8).
Each of CO2/CH4, Xe/Kr and H2S/CO2 has P–X and P–X–T profiles under
`paper_profiles/`. Paper pressure grids are **linear in physical pressure**,
with 2,499 P–X or 2,541 P–X–T points. Initial designs contain 54 or 90 coordinates;
64 or 96 independent holdout coordinates are reserved before fitting.

```bash
python -m modules.module_b_campaign init VERIFIED_CUBTC_CONFIG.json CAMPAIGN/state.json \
  --case CO2-CH4 --space P-X --mode paper --budget 100
```

`paper` uses published kernels and a 90% PAC target with beta 0.02 (P–X) or 0.05
(P–X–T). `upstream_compat` reproduces repository kernel/selection/stopping choices,
including the H2S/CO2 P–X–T kernel discrepancy and the maximum relative uncertainty
threshold 0.005. Both select the component with the worst relative log-space
uncertainty, then its largest absolute sigma. Deliberate differences remain:
fixed optimizer seed, no duplicate acquisition, locked holdouts and LAMMPS labels.
`noise_aware` uses the general seed-noise model on these coordinates.

Training and holdout labels in live campaigns must come from new LAMMPS runs.
The original CO2/CH4 training file in `fixtures/` is used only for an independent
algorithm regression test; its checksum, source and unverified loading unit are
documented alongside it. It is not a physical calibration/reference import.

The profiles explicitly leave unresolved physical parameters open. Verify the
Cu-BTC structure, charges, guest parameters, interactions, reservoir, adsorption
basis and source loading unit before claiming physical reproduction. Set
`benchmark.paper_uptake_unit` explicitly to the verified `mol/kg` or
`molecules/cell`; log-space PAC is unit-dependent. Automated paper execution also
requires an explicit `benchmark.paper_model_verified: true` declaration in the
user-supplied physical configuration; this declaration is not an automatic
verification of the literature. Source assets/checksums are still enforced.
No production campaign or dense reference grid was run in this implementation.

## Statistics, pilot and short checks

Components use 95% Student-t intervals across independent seeds. Selectivities
are ratios of component means with paired-seed Fieller intervals; unbounded and
undefined intervals are explicitly identified. At least three seeds and at most
20% relative interval half-width are required for each active component and each
defined, configured pair. This measures statistical precision, not complete
equilibrium convergence. Zero-fraction pairs are reported undefined and do not
prevent precision assessment of the active subsystem.

For an existing Module-B run with raw LAMMPS logs, generate a time-series report
without rerunning or modifying the simulation:

```bash
python -m analysis.module_b_convergence_report \
  outputs/module_B_mixture_selectivity/runs/RUN_NAME
```

The report writes JSON, CSV and Markdown under
`RUN_NAME/analysis/module_b_convergence/`. Per-seed uncertainties use
autocorrelation-informed batch means combined with independent-seed variation;
selectivity intervals use paired-seed Fieller inference with the same uncertainty
propagation. By
default, it compares the first 20,000 production steps with the remaining last
30,000 steps, targets 20% relative interval half-width and uses 0.05
molecules/cell as the small-loading absolute-difference floor. These operational
thresholds are recorded in the report and do not prove equilibrium.

References must match material, component set, composition, temperature, pressure
range and known adsorption basis. Total mixture uptake is never compared as CO2
uptake to a pure-gas isotherm. The corrected MOF-5 pilot stays in its original
`evaluation_corrected` folder with archived framework mass 6158.936 u and the
missing-raw-log limitation. Its status remains:
**Pilot ausgewertet; quantitative Referenzvalidierung offen**.

```bash
python -m unittest discover -s tests -p 'test_module_b*.py'
python benchmark.py input_json_files/benchmark_module_b_co2_ch4_smoke.json \
  --run-isotherm --skip-registry-update
python benchmark.py input_json_files/benchmark_module_b_flexible_guests_smoke.json \
  --run-isotherm --skip-registry-update
```

The last two commands are deliberately short technical tests with three independent
seeds. Existing run directories are not overwritten by default. Tests cover shared
material preparation across the local catalog, representative charge schemes/cells,
binary/ternary/quaternary mixtures, component order and zero fractions, reference
exclusions, t/Fieller statistics, analytic ideal-gas/inner-coordinate distributions,
nontrivial Rosenbluth normalization, HMC, fixed frameworks, paper transforms,
holdouts, budgets and resume/import behavior. They do not demonstrate production
sampling precision or scientific reproduction of the paper.
