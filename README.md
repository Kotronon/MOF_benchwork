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