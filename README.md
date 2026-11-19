# MOF Simulation Benchmark

This repository benchmarks adsorption, gas-mixture selectivity, interatomic
potentials, and framework flexibility in metal-organic frameworks. `benchmark.py`
is the command-line entry point. Run commands from the repository root.

## Start here

```bash
python -m pip install -r requirements.txt
python benchmark.py input_json_files/benchmark.json --dry-run --skip-registry-update
```

The dry run resolves and checks the example configuration without starting a
simulation. Running a LAMMPS calculation also requires the configured assets
and an `lmp` executable. See [Getting started](docs/getting_started.md) and
[Installation](docs/installation.md).

## Documentation

| Topic | Guide |
| --- | --- |
| Code layout and execution flow | [Architecture](docs/architecture.md) and [module architecture](docs/architecture_modules.md) |
| Inputs and generated files | [Configuration](docs/configuration.md) and [outputs](docs/outputs.md) |
| Scientific checks and automated tests | [Validation](docs/validation.md) and [testing](docs/testing.md) |
| Problems and diagnostics | [Troubleshooting](docs/troubleshooting.md) |

| Module | Scope | Guide |
| --- | --- | --- |
| A | Single-component rigid-framework adsorption | [Module A](docs/modules/module_a.md) |
| B | Mixtures and GCMC active learning | [Module B](docs/modules/module_b.md) |
| C | Potential benchmarks, MLIP adsorption, and fine-tuning | [Module C](docs/modules/module_c.md) |
| D | Framework flexibility | [Module D](docs/modules/module_d.md) |

## Repository layout

```text
benchmark.py                  CLI entry point
docs/                         Versioned guides and reviews
modules/                      Module A file and Module B, C, D packages
pipeline/                     Shared planning, preparation, and execution
engines/                      Simulation engine adapters
resolvers/                    Material, gas, force-field, and reference lookup
analysis/                     Result analysis and diagnostic scripts
tests/                        Automated tests
data/ and input_json_files/   Reference assets and example configurations
outputs/ and reports/         Generated run artifacts
```

The [module architecture guide](docs/architecture_modules.md) maps each package
to its entry points and dependencies. Scientific conclusions require matching
reference conditions and independent sampling checks; see [Validation](docs/validation.md).
