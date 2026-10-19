# Implementation validation — 2026-10-19

The code/configuration/short-test implementation step is complete. Scientific
reproduction of the paper remains open; no production campaign or complete
reference grid was executed.

## Automated checks

- Full existing test discovery: **304 tests, OK, 13 skipped** (150 s), using
  `/Users/kathi/miniforge3/envs/MOF_sim/bin/python -m unittest discover -s tests -p 'test_*.py'`.
  Skips are environment/dependency-conditional tests; they are not counted as passes.
- Focused Module B discovery before the final small regression addition:
  **59 tests, all passed**. The extra zero-fraction precision regression and the
  existing RASPA molecule-resolution regression also passed separately.
- `git diff --check` passed. Component/interval and ternary surrogate-projection
  figures were generated and visually inspected.
- Shared preparation checked for **1,357 local material catalog entries**.
  Representative full materializations: MOF-5/DDEC, IRMOF-7/EQeq and ZIF-8/PACMOF.
  This does not claim 1,357 validated adsorption simulations.

The test suite covers the pinned paper-coordinate/GP-transform fixture, all six
profile definitions, binary/ternary/quaternary components, zero fractions,
component-order invariance in the EOS/candidate pool, source identity, holdout
isolation, distinct PAC/budget/precision statuses, archived-input tampering,
component t intervals, paired Fieller intervals and reference exclusions.
Flexible sampling checks include ideal molecule counts, physical/reference
partition functions with unequal springs, inner-coordinate distributions,
Metropolis HMC, exact stochastic checkpoint restoration and LAMMPS energy/forces
with a fixed framework.

## Actual LAMMPS short runs

Artifacts are below `outputs/module_B_mixture_selectivity/runs/`:

| Run directory | Seeds | Scope | Result |
| --- | --- | --- | --- |
| `general_co2_ch4_technical_smoke_verified` | 12345, 23456, 34567 | MOF-5, 300 K, 1 bar, 100 equilibration + 100 production steps | All completed |
| `general_flexible_guests_technical_smoke_verified` | 12345, 23456, 34567 | MOF-5, two synthetic Gaussian-tree guests, 10 equilibration + 20 production moves, two CBMC trials, optional HMC | All completed |

Native CO2/CH4 dump checks found **0 Å framework displacement** and identical
cells across saved frames. Flexible runs exercised accepted insertion, deletion,
regrowth and HMC moves; all three checkpoints resumed without new moves and
retained identical component summaries. Corrected CSVs, uncertainty plots and
reports are in each run's `evaluation_corrected/` subdirectory.

These deliberately short runs **do not meet the statistical precision target**
and are not production adsorption data. The flexible Gaussian models are
technical fixtures, not forcefields for real chemical species.

## Corrections and remaining scientific work

Adding methane previously merged an entire fallback bundle and could replace
CRAFTED CO2 parameters. Fallback types are now scoped to their owning components;
conflicting explicit parameters fail. The `_verified` CO2/CH4 run uses this
corrected assignment. Older technical results must not be used in its place.

The local neutral ZIF-8 CIF lacks atom-site charge records. It fails the same
shared converter checks as in Module A; no neutral-charge model is silently
invented. A charged, available ZIF-8/PACMOF asset was used for the positive test.

Remaining before a scientific reproduction claim:

1. Establish the paper's exact structure, charges, forcefield/reservoir details,
   adsorption basis and source loading units, with source provenance.
2. Run explicitly budgeted LAMMPS training and independent held-out points at
   adequate precision, investigating equilibration from retained time series.
3. Report surrogate errors against those LAMMPS holdouts separately from
   LAMMPS deviations against compatible paper data, including learning curves.

The corrected historical MOF-5 pilot is preserved with its archived **6158.936 u**
framework mass and missing-raw-log limitation:
**Pilot ausgewertet; quantitative Referenzvalidierung offen**.
Flexible MOFs remain reserved for Module D.
