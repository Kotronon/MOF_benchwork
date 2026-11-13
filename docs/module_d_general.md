# Module D: general classical and MLIP profiles

Module D accepts arbitrary material names through `benchmark.flexibility.profile = "general"`.
It has a common potential definition, input archive and rigid/flexible analysis.
**This does not automatically parameterize arbitrary MOFs.** A CIF or an adsorption
Lennard-Jones table alone is insufficient for flexible classical MD.

| Capability | Current implementation |
| --- | --- |
| Classical flexible framework, fixed periodic cell | Executable native LAMMPS MD/GCMC, NVT |
| Neutral atomic guest | Executable, GCMC exchanges/moves and guest NVT |
| Neutral rigid molecular guest | Executable, GCMC exchanges and rigid-body NVT |
| Classical vs MLIP potential definitions | Common schema; separate run directories |
| MLIP execution | Blocked pending a tested native adapter and model coverage checks |
| Classical NPT with variable cell | Executable with explicit mechanical pressure; breathing is not established by execution |
| Mixtures, flexible guests, implicit TIP4P | Not supported by this general execution adapter |
| NbOFFIVE-1-Ni paper reproduction | Separate existing paper profile, still planning/analysis only |

The real-material [MIL-53(Al)/CO₂ cluster pilot](module_d_mil53_cluster.md) includes
published host parameters and an explicitly unvalidated CO₂ cross model.

NPT requires `md.cell_mode` (`aniso` or `tri`), `external_pressure_bar` and
`barostat_damping_ps`. It prepares rigid, flexible NVT and flexible NPT cases.
`aniso` changes lengths; `tri` also changes tilts. Guest rigid-body integration
precedes box changes; the barostat uses global pressure and dilates all atoms.

NVT allows framework atoms and linkers to move but keeps cell vectors fixed. It
cannot establish an adsorption-induced change of lattice volume or a breathing
transition. Broad material input support is therefore distinct from support for
every flexibility mechanism. No RASPA/gRASPA installation is needed.

## First verify the technical installation

The included example is an **invented bonded system, not a MOF force field or
scientific isotherm**. It exercises a charged, neutral framework and a neutral
rigid triatomic guest, long-range electrostatics, insertion/deletion and the
rigid/flexible comparison. Do not use its coefficients for a real material.

From the repository root, in an environment with Python and a `lmp` executable:

```bash
python benchmark.py input_json_files/benchmark_module_d_technical_demo.json --dry-run
python benchmark.py input_json_files/benchmark_module_d_technical_demo.json --prepare
python -m modules.module_d_general check --directory outputs/module_D_technical_demo/runs/classical_v1
python -m modules.module_d_general smoke --directory outputs/module_D_technical_demo/runs/classical_v1
python -m modules.module_d_general run --directory outputs/module_D_technical_demo/runs/classical_v1
python benchmark.py input_json_files/benchmark_module_d_technical_demo.json --evaluate
```

Run `smoke` and `run` inside a cluster allocation. These commands submit no jobs
and do not require a GPU. The initial tested implementation uses CPU LAMMPS;
GPU availability alone does not accelerate GCMC or guarantee adapter compatibility.
LAMMPS needs MC, MOLECULE and, for this example, KSPACE and RIGID support.
Local integration testing used LAMMPS 22 Jul 2025 Update 4.

`--command /path/to/lmp` selects another executable. It also accepts an argument
list for a launcher (for example `--command srun lmp`). Use the same command for
smoke and production. Scheduler account, partition and resources remain specific
to your cluster. Run the smoke test on the cluster build, even after local tests.

Alternatively, `benchmark.py CONFIG --run-test --resume` and then
`benchmark.py CONFIG --run-isotherm --resume` use `lmp` from PATH. `--resume`
verifies identical preparation; it is **not** checkpoint/RNG continuation.

## Supply a real material

Copy `input_json_files/benchmark_module_d_general_classical.template.json` and
replace its placeholders. Its missing assets intentionally prevent execution.
The listed styles/cutoff/exclusions are examples, not recommended universal
settings. All pair, bond, angle, dihedral and improper terms required by the
chosen force field must be defined, either in the data file or in the explicit
coefficient commands. Add the matching styles before reading data.

Required inputs:

1. `assets.initial_structure`: a **dry, neutral, already parameterized periodic
   LAMMPS data file**, `atom_style full`, with masses, charges and topology.
   Guest atom/bond/angle types and their masses/coefficients must already be
   reserved in the data file. Guest atom types must be separate from host types.
   The file is the complete simulation cell; no automatic replication occurs.
2. `potential`: `kind` (`classical` or `mlip`), name, parameter source, unit
   system (`real` or `metal`), `styles` before `read_data`, and `coefficients`
   afterwards. Supported commands are limited to force-field definitions; no
   arbitrary `run`, `fix`, `shell` or LAMMPS variable expansion is accepted here.
   Source files in `potential.files` are copied to `assets/potential/NAME`;
   coefficient commands must reference that bundle-relative location.
3. `guest`: a neutral atomic type, or a single rigid molecule template with
   explicit `Coords`, `Types` and `Charges` and matching `atoms_per_molecule`.
   Reserve per-atom topology through `extra_bond_per_atom`, etc., as needed.
   Rigid molecules require `intra_energy` in potential units and its source;
   establish that reference with the chosen exclusions/bonded terms rather than
   assuming it is zero. Implicit four-site TIP4P needs a separate adapter.
4. Temperature, positive pressures and **explicit fugacities**, with their
   source or a stated ideal-gas approximation. Pressure is never silently used
   as fugacity. The general profile uses pressure, not paper-specific humidity.
5. Sampling length, independent seeds and validation of the chosen force field
   for dry and loaded framework deformations. `assets.validation` can archive
   this evidence; presence of a file does not certify it.

`potential.units` controls all supplied force-field coefficients and guest
intramolecular energy. Conditions use bar/K; timestep is fs and thermostat
damping is ps. The runner converts pressure to atm for `real`, timestep to ps
for `metal`, and reported potential energy to eV. Uptake is absolute mol/kg of
the **dry framework**, obtained from guest count and host masses in the data.
Legacy normalized `simulation.forcefield`, cutoff and unit-cell defaults do not
generate this profile's potential; the explicit potential and data file do.

## Sampling and comparison

This is native **same-potential hybrid MD/GCMC**, not a reproduction of the paper's
alternating MLIP/classical sorption-relaxation procedure. A single LAMMPS process
retains the loaded configuration; the MC acceptance uses `full_energy`. All
branches start from the same dry cell and use the same potential, pressure,
fugacity, temperature, sampling length and guest treatment. The flexible branches integrate framework motion; only the NPT branch changes
the cell. The rigid branch still integrates guest MD.

For rigid molecular guests, native `fix gcmc ... rigid` requires
`move_attempts=0`. MD provides molecular translation and rotation; GCMC provides
insertion/deletion. This is not configurational-bias MC. See the official
[GCMC](https://docs.lammps.org/fix_gcmc.html) and
[rigid-body](https://docs.lammps.org/fix_rigid.html) documentation.

`hybrid.equilibration_steps` is followed by `sr_cycles` measurement blocks of
`hybrid.steps_per_cycle` MD steps. The legacy `sr_cycles` field is reused for
analysis compatibility; these are hybrid sampling blocks, not paper SR cycles.
Equilibration must be a multiple of block length, and block length a multiple
of exchange interval. Discarded blocks and stationarity diagnostics apply to
**both** rigid and flexible branches. Blocks can be correlated; no naive
confidence interval is computed from them.

Measurements are block means of uptake, framework RMSD, cell volume and potential
energy. RMSD is the mean of `sqrt(MSD)` using unwrapped host displacement with
center-of-mass drift removed, relative to the initial dry geometry. This differs
from the separate minimum-image `structural_metrics` helper. The report records
the convention. In NPT this includes affine cell strain; `cell_history.txt`
separately records cell lengths, tilts and pressure tensor components in native
LAMMPS units. See [LAMMPS MSD](https://docs.lammps.org/compute_msd.html).

`mc_attempts.txt` stores cumulative attempted/accepted MC counts. Trajectories
include IDs, molecule IDs, types, charges and image flags. Final data and binary
restart files are saved for inspection; automatic continuation is not implemented.

## Archives, gates and outputs

Preparation copies declared assets and writes a SHA-256 manifest. Execution
verifies the archive and does not depend on the original source paths; transfer
the bundle plus the current repository runner to the cluster. Existing bundles,
partial runs and results are never overwritten. After a failed run, keep its
logs and choose a fresh `output.run_id` for a corrected configuration.

`check` returns a nonzero exit code for a blocked profile. It checks archived
inputs and declared capabilities, not the physical validity of a potential.
`smoke` performs short rigid and flexible runs at the first pressure/seed and
writes only under `smoke/`. It is not a convergence study or a test of every
loading. `run` requires a passing smoke test of the same bundle, runner code
and LAMMPS build description. Production runs all planned cases sequentially.
Use separate bundles/jobs for independent materials or potential profiles.

Results are under `results/CASE_ID/`; completed campaign measurements become
`cycle_results.csv`. `--evaluate` writes `evaluation/module_d_report.json` and
`rigid_vs_flexible.csv`, including pressure, variant and stationarity of both
branches. Completion and passing stationarity do not prove physical validation
or equilibration. All current reports retain `validation_status=not_validated`.

## MLIP extension

`input_json_files/benchmark_module_d_general_mlip.template.json` demonstrates the
same interface for a future MLIP backend. Element/type mapping, checkpoint,
framework/gas interactions, changing guest number, deformation and (for NPT)
stress must be validated for each model. Arbitrarily combining classical MD and
MLIP GCMC is not enabled. At present, MLIP profiles can be prepared and their
imported measurements analyzed; executing them raises an explicit blocker.

## Re-run tests

```bash
python -m unittest tests.test_module_d_flexibility tests.test_module_d_general
MODULE_D_LAMMPS=/path/to/lmp python -m unittest tests.test_module_d_general
```

The opt-in integration test runs atomic `real`, molecular `real` and atomic
`metal` cases through preparation, smoke, production and evaluation. These are
technical fixtures and must not be interpreted as MOF validation.
