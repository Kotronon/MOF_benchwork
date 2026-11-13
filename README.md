# MOF Simulation Benchmark

## Overview

This project provides a modular benchmarking framework for evaluating molecular simulations of metal-organic frameworks (MOFs).

The benchmark supports adsorption simulations, gas-mixture selectivity, machine-learning interatomic potentials (MLIPs), and framework flexibility.

LAMMPS is the main engine for the classical adsorption workflows. Some Module C
workflows also use MLIP-MC, CP2K, or FLAMES; the selected workflow determines the
required tools.

## Benchmark Modules

| Module | Description | Main metrics |
| :---:  | :---: | :---: |
| [A](docs/module_a.md) | Single-component CO_2 adsorption | Uptake, isotherms, reference error, runtime |
| [B](docs/module_b.md) | Gas-mixture adsorption and active learning | Selectivity, prediction error, simulation savings, uncertainty |
| [C](docs/module_c.md) | Potential benchmarks and MLIP adsorption | Energy/force errors, adsorption properties, runtime |
| [D](MODULE_D_FLEXIBILITY.md) | Framework flexibility | Structural deformation, pore-volume changes, adsorption differences |

Shared guides: [getting started](docs/getting_started.md),
[installation](docs/installation.md), [configuration](docs/configuration.md),
[architecture](docs/architecture.md), [validation](docs/validation.md),
[outputs](docs/outputs.md), and [troubleshooting](docs/troubleshooting.md).

### Module D starting point

The [general Module D profile](docs/module_d_general.md) now provides a shared
classical/MLIP potential interface and executable classical NVT/NPT MD/GCMC, with
rigid/flexible comparisons and a [MIL-53(Al)/CO₂ pilot](docs/module_d_mil53_cluster.md).
MLIP execution remains blocked. Each real MOF still needs its own parameterized flexible potential.

The NbOFFIVE-1-Ni/H2O campaign targets LAMMPS for both MD and GCMC, without a
RASPA/gRASPA dependency. It supports offline planning, preparation and
analysis of imported sorption-relaxation cycles. This paper profile's MD/GCMC
execution is not yet implemented. See [the Module D guide](MODULE_D_FLEXIBILITY.md) for the paper
protocol, missing assets, measurement schema and validation boundaries.

```bash
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --dry-run
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare
```

The preparation creates an isolated bundle under
`outputs/module_D_nboffive_water/runs/planning_lammps_v1/` without starting simulations
or updating shared registries. Use `--prepare --resume` to verify an existing
identical bundle, or choose a new `--run-id` for changed settings.


## Installation

See [the installation guide](docs/installation.md) for Python dependencies,
external executables, and optional Module B/C tools. A short setup checklist is
in [Getting started](docs/getting_started.md).

## Quickstart

From the repository root, inspect the supplied MOF-5 configuration, prepare its
LAMMPS inputs, then run a short technical test:

```bash
python benchmark.py input_json_files/benchmark.json --dry-run --skip-registry-update
python benchmark.py input_json_files/benchmark.json --prepare --skip-registry-update
python benchmark.py input_json_files/benchmark.json --run-test --skip-registry-update
```

The test requires a working `lmp` executable and the configured material,
force-field, and molecule assets. It is a technical smoke test, not a
scientifically converged isotherm. See [Module A](docs/module_a.md) for context
and [outputs](docs/outputs.md) for the generated files.

## Project structure

| Path | Purpose |
| --- | --- |
| `benchmark.py` | Command-line entry point |
| `modules/` | Module-specific planning, execution, and evaluation |
| `pipeline/` | Shared configuration, planning, preparation, and evaluation |
| `engines/` | LAMMPS and other execution adapters |
| `converter/`, `resolvers/`, `parsers/` | Input conversion, asset resolution, and result parsing |
| `analysis/` | Post-processing and reports |
| `input_json_files/` | Example benchmark configurations |
| `docs/` | Installation, configuration, output, and module guides |

## Reproducibility

Keep the input JSON and complete run directory together. Record the selected
materials and force-field assets, software versions, random seeds, simulation
settings, and reference-data provenance when reporting results. Prepared runs
archive their source inputs; Module D also records source checksums in its
planning bundle. Technical smoke tests should not be presented as production
results.

## Citations

There is not yet a single project-wide bibliography. Cite the source paper and
data/model releases for the specific workflow being reported; start with the
relevant [module guide](#benchmark-modules) and its protocol/provenance notes.
