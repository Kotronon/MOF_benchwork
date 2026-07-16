# MOF Benchwork: From Benchmark Planning to LAMMPS-Ready Inputs

---

## Project Goal

- Build a reproducible benchmark workflow for MOF adsorption simulations.
- Focus on Module A first: single-gas CO2 adsorption in MOF-5 / IRMOF-1.
- Use CRAFTED data as the source for:
  - CIF framework structures
  - force field parameters
  - adsorbate molecule definitions
  - reference isotherm data
- Start with validated input generation before running full GCMC simulations.

---

## Current Architecture

- `benchmark.py` acts as the central pipeline orchestrator.
- It does not run simulations directly.
- It builds a reproducible run plan from `benchmark.json`.
- Main pipeline steps:
  - load configuration
  - normalize defaults
  - resolve material and force field data
  - select benchmark module
  - build a dry-run plan
  - prepare LAMMPS input artifacts

---

## Benchmark Configuration

- `benchmark.json` follows a structured schema:
  - `material`
  - `adsorbates`
  - `conditions`
  - `benchmark`
  - `simulation`
  - `output`
- Missing values are filled with sensible defaults.
- Example defaults:
  - adsorbate: `CO2`
  - task: `auto`
  - simulation method: `GCMC`
  - force field: `UFF`

---

## Material and Data Resolution

- Implemented `MaterialResolver`.
- Supports aliases such as:
  - `MOF-5`
  - `IRMOF-1`
- Resolves the correct CIF path from CRAFTED.
- Charge scheme selection is currently handled during material resolution.
- `charge_resolver.py` is intentionally not implemented yet, because separate charge logic is not needed at this stage.

---

## Force Field Resolution

- Implemented `ForcefieldResolver`.
- Resolves required UFF files from CRAFTED:
  - `force_field.def`
  - `force_field_mixing_rules.def`
  - `pseudo_atoms.def`
  - `CO2.def`
  - `N2.def`
- Missing adsorbate definitions produce clear errors.
- This keeps benchmark planning independent from hard-coded file paths.

---

## CIF to LAMMPS Data Conversion

- Implemented `converter/cif_to_lammps_data.py`.
- ASE is used to read geometry and cell vectors.
- CIF charges are parsed manually from `_atom_site_charge`.
- This is necessary because ASE does not automatically preserve the CRAFTED charge column.
- Output:
  - framework-only LAMMPS `.data`
  - `Masses`
  - `Atoms # full`
  - restricted triclinic box parameters

---

## Restricted Triclinic Box Support

- IRMOF-1 is converted into LAMMPS restricted triclinic form.
- The converter computes:
  - `lx`, `ly`, `lz`
  - `xy`, `xz`, `yz`
- Fractional CIF coordinates are transformed into LAMMPS-compatible Cartesian coordinates.
- LAMMPS can successfully read the generated structure with `read_data`.

---

## Molecule Template Conversion

- Implemented `converter/molecule_to_lammps_template.py`.
- Parses CRAFTED / RASPA molecule definition files.
- Currently supports rigid linear molecules such as:
  - CO2
  - N2
- Generates LAMMPS molecule templates with:
  - coordinates
  - atom types
  - bonds
  - charges
- CO2 now uses global atom type IDs consistent with the framework data file.

---

## Force Field to LAMMPS Conversion

- Implemented `converter/forcefield_to_lammps.py`.
- Parses:
  - pseudo atom masses and charges
  - Lennard-Jones parameters
  - Lorentz-Berthelot mixing rule
- Builds global atom type assignments:
  - framework atoms: `Zn`, `H`, `C`, `O`
  - CO2 atoms: `O_co2`, `C_co2`
- Generates explicit LAMMPS `pair_coeff` lines.

---

## Unit Conversion and Mixing

- CRAFTED stores Lennard-Jones epsilon in Kelvin.
- LAMMPS `units real` expects epsilon in kcal/mol.
- Conversion:

```text
epsilon_kcal_mol = epsilon_K * 0.0019872041
```

- Lorentz-Berthelot mixing:

```text
sigma_ij = (sigma_i + sigma_j) / 2
epsilon_ij = sqrt(epsilon_i * epsilon_j)
```

---

## Generated LAMMPS Artifacts

The current workflow can generate:

- `IRMOF-1.data`
  - framework atoms
  - framework charges
  - extra CO2 atom types
  - bond type capacity for molecule templates
- `CO2.template`
  - CO2 geometry
  - global atom type IDs
  - CO2 charges
  - rigid bonds
- `forcefield.in`
  - `pair_style`
  - `kspace_style`
  - explicit `pair_coeff` values

---

## LAMMPS Smoke Test

- A LAMMPS `run 0` smoke test was added.
- It verifies that LAMMPS can read:
  - generated `.data`
  - generated CO2 molecule template
  - generated force field include file
- Example input:

```lammps
units real
atom_style full
boundary p p p
read_data IRMOF-1.data extra/special/per/atom 2
molecule co2 CO2.template
include forcefield.in
run 0
```

---

## Why `extra/special/per/atom 2` Is Needed

- The framework `.data` file initially contains no CO2 molecules.
- CO2 is loaded later through a molecule template.
- The CO2 template contains bonds.
- LAMMPS needs reserved memory for special bonded neighbors.
- `extra/special/per/atom 2` provides enough capacity for bonded CO2 molecules.

---

## Testing Status

- Implemented unit and smoke tests in `test_benchmark.py`.
- Current coverage includes:
  - configuration loading and defaults
  - material alias resolution
  - module selection
  - CRAFTED molecule parsing
  - CIF charge parsing
  - restricted triclinic `.data` generation
  - force field parsing
  - pair coefficient generation
  - LAMMPS `read_data` and `run 0`
- Current result:

```text
28 tests OK
```

---

## Development Environment

- Conda environment: `MOF_sim`
- Installed and tested:
  - ASE
  - LAMMPS
  - OVITO
- LAMMPS version supports:
  - `fix gcmc`
  - `fix widom`
- System Python tests still pass with expected skips for ASE/LAMMPS-dependent tests.

---

## What Has Been Achieved

- The project now has a reproducible planning layer.
- CRAFTED data can be resolved instead of hard-coded.
- IRMOF-1 CIF data can be converted into LAMMPS `.data`.
- CO2 molecule definitions can be converted into LAMMPS molecule templates.
- UFF force field parameters can be converted into LAMMPS pair coefficients.
- LAMMPS can successfully validate the generated input artifacts with `run 0`.

---

## Next Steps

1. Add a real `--prepare` mode to `benchmark.py`.
2. Materialize the prepare plan into output directories.
3. Generate complete Module A LAMMPS input scripts.
4. Run a very short one-pressure CO2 GCMC test.
5. Parse basic LAMMPS output.
6. Only then scale to a full pressure grid and compare against reference isotherms.

---

## Scientific Roadmap

- Short-term:
  - one pressure point
  - short GCMC validation run
  - verify molecule insertion works
- Medium-term:
  - full CO2 isotherm for IRMOF-1
  - compare with CRAFTED reference data
- Long-term:
  - Module B: mixtures
  - Module C: Henry constants / Widom insertion
  - Module D: flexible frameworks

