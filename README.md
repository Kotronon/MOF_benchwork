# MOF SImulation Benchmark

## Overview

This project provides a modular benchmarking framework for evaluating molecular simulations of metal-organic frameworks (MOFs).

The benchmark supports adsorption simulations, gas-mixture selectivity, machine-learning interatomic potentials (MLIPs), and framework flexibility.

LAMMPS is used as the primary simulation engine.

## Benchmark Modules

| Module | Description | Main metrics |
| :---:  | :---: | :---: |
| A | Single-component CO_2 adsorption | Uptake, isotherms, reference error, runtime |
| B | Gas mixture adsorption with active learning | Selectivity, prediction error, simulation savings, uncertainty |
| C | Evaluation of MLIPs | Energy/force errors, adsorption-related properties, runtime |
| D | Framework flexibility | Structural deformation, pore-volume changes, adsorption differences |

### Module D starting point

The NbOFFIVE-1-Ni/H2O campaign supports offline planning, preparation and
analysis of imported sorption-relaxation cycles. MD/GCMC execution is not yet
implemented. See [the Module D guide](MODULE_D_FLEXIBILITY.md) for the paper
protocol, missing assets, measurement schema and validation boundaries.

```bash
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --dry-run
python benchmark.py input_json_files/benchmark_module_d_nboffive_water.json --prepare
```

The preparation creates an isolated bundle under
`outputs/module_D_nboffive_water/runs/planning_v1/` without starting simulations
or updating shared registries. Use `--prepare --resume` to verify an existing
identical bundle, or choose a new `--run-id` for changed settings.


## Installation

### Requirements

* Python
* LAMMPS
* ASE
* TODO

## Quickstart

1. Prepare the benchmark configurations.
2. Select the material and benchmark module.
3. Specify simulation parameters.
4. Run the benchmark.
5. Inspect the generated results and evaluation reports.

## Project structure

* modules/: Benchmark-specific implementations
* engines/:
* analysis/:
* converter/:
* input_json_files/:
* parsers/:
* pipeline:/
* resolvers:/

## Reproducibility
TODO: Simulation settings, force-field parameters, software versions, random seeds and reference data

## Citations
