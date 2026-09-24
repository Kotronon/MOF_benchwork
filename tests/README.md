# Test suite

The tests are grouped by responsibility:

- `test_planning.py`: configuration, preparation, CLI, and run-plan tests
- `test_resolvers_and_registry.py`: material, adsorbate, reference, and registry tests
- `test_forcefields.py`: force-field parsing and LAMMPS coefficient generation
- `test_converters_and_lammps.py`: CIF/molecule conversion and LAMMPS integration
- `test_runners_and_resume.py`: restart, resume, and replicate aggregation
- `test_evaluation.py`: log parsing, isotherm tables, and reference evaluation
- `test_analysis.py`: convergence, reproducibility, cell, and k-space analysis
- `module_c/`: potential results, interaction energies, classical LAMMPS,
  Module-A parity, smoke dataset, and LAMMPS structure tests

Run the fast Module C tests:

```bash
conda run --no-capture-output -n MOF_sim \
  python -m unittest discover -s tests/module_c -t . -v
```

Run one thematic test file:

```bash
conda run --no-capture-output -n MOF_sim \
  python -m unittest -v tests.test_forcefields
```

Run only the classical Module-A/Module-C parity checks:

```bash
conda run --no-capture-output -n MOF_sim \
  python -m unittest -v tests.module_c.test_classical_parity
```

The parity file contains fast unit tests plus two real LAMMPS integration
test groups. The integration tests use identical UFF/DDEC settings (`12.8 A`
cutoff and PPPM accuracy `1e-4`) and compare potential-energy components and
atomwise forces for framework-only and fixed MOF+CO2 configurations. The
positive deterministic positions, the host-guest interaction energy, and a
sampled frame from an existing Module-A GCMC dump are covered. The intentional
atom-overlap probe must fail with a non-finite energy. Environment-dependent
tests are skipped unless ASE, CoolProp, CRAFTED, and the `lmp` executable are
available; the sampled-frame test is additionally skipped when the local dump
is absent. These checks validate the shared classical baseline, not equality
of stochastic GCMC trajectories or adsorption averages.

Run the complete suite, including available LAMMPS integration tests:

```bash
conda run --no-capture-output -n MOF_sim \
  python -m unittest discover -s tests -t . -v
```

The complete suite can take substantially longer because some tests launch
LAMMPS. Tests requiring unavailable optional dependencies are skipped.
