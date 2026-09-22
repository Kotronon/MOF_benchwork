# Test suite

The tests are grouped by responsibility:

- `test_planning.py`: configuration, preparation, CLI, and run-plan tests
- `test_resolvers_and_registry.py`: material, adsorbate, reference, and registry tests
- `test_forcefields.py`: force-field parsing and LAMMPS coefficient generation
- `test_converters_and_lammps.py`: CIF/molecule conversion and LAMMPS integration
- `test_runners_and_resume.py`: restart, resume, and replicate aggregation
- `test_evaluation.py`: log parsing, isotherm tables, and reference evaluation
- `test_analysis.py`: convergence, reproducibility, cell, and k-space analysis
- `module_c/`: potential result, smoke dataset, and LAMMPS structure tests

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

Run the complete suite, including available LAMMPS integration tests:

```bash
conda run --no-capture-output -n MOF_sim \
  python -m unittest discover -s tests -t . -v
```

The complete suite can take substantially longer because some tests launch
LAMMPS. Tests requiring unavailable optional dependencies are skipped.
