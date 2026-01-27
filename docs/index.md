# JFChemistry

[![License](https://img.shields.io/github/license/cfarm6/jfchemistry)](https://github.com/cfarm6/jfchemistry/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/cfarm6/jfchemistry/test.yml?branch=master&logo=github-actions)](https://github.com/cfarm6/jfchemistry/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/cfarm6/jfchemistry)](https://codecov.io/gh/cfarm6/jfchemistry)

A comprehensive computational chemistry workflow package built on [jobflow](https://github.com/materialsproject/jobflow).

## Overview

JFChemistry provides a flexible framework for building computational chemistry workflows using various methods ranging from force fields to machine learning potentials. It seamlessly integrates with popular chemistry libraries like RDKit, ASE, and Pymatgen.

## Key Features

-   **Structure Generation**: Create 3D molecular structures from SMILES strings or PubChem database
-   **Conformer Search**: Generate and optimize multiple conformers using RDKit and CREST
-   **Geometry Optimization**: Optimize structures with multiple methods (GFN-xTB, AimNet2, ORB models)
-   **Structure Modification**: Perform protonation/deprotonation using CREST
-   **Workflow Management**: Build complex, parallelizable workflows with jobflow
-   **Multiple Calculators**: Support for AimNet2, ORB, and FairChem machine learning potentials along with traditional approaches such as PySCF, TBLite, and ORCA

## Installation

The package uses [Pixi](https://pixi.sh) for dependency management:

```bash
# Clone the repository
git clone https://github.com/cfarm6/jfchemistry.git
cd jfchemistry

# Install with pixi
pixi install

# For development
pixi install -e dev

# For documentation
pixi install -e docs
```

## Quick Start

Here's a simple workflow to create a molecule from SMILES, generate conformers, and optimize:

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import TBLiteOptimizer

# Create molecule from SMILES
smiles_maker = Smiles(add_hydrogens=True)
smiles_job = smiles_maker.make("CCO")  # Ethanol

# Generate 3D conformers
generator = RDKitGeneration(num_conformers=10, method="ETKDGv3")
gen_job = generator.make(smiles_job.output["structure"])

# Optimize with GFN2-xTB
optimizer = TBLiteOptimizer(method="GFN2-xTB", fmax=0.01)
opt_job = optimizer.make(gen_job.output["structure"])

# Access results
optimized_structures = opt_job.output["structure"]
properties = opt_job.output["properties"]
```

## Architecture

JFChemistry is built around a set of base classes that handle the core functionality of the package.

-   **SingleMoleculeMaker**: For operations on non-periodic structures such as molecules
-   **SingleStructureMaker**: For operations on periodic structures such as crystals
-   **SingleStructureMoleculeMaker**: For operations that apply to both periodic and non-periodic structures
-   **SingleRDMoleculeMaker**: For operations on molecules without 3D coordinates (RDKit molecules)

These base classes automatically handle:

-   Job distribution for lists of structures
-   Parallel processing of multiple conformers
-   Consistent output formats across different methods

## Modules

-   **[Inputs](inputs.md)**: Create molecules from SMILES, PubChem CID
-   **[Generation](generation.md)**: Generate 3D structures from molecular graphs
-   **[Conformers](conformers.md)**: Search conformational space with CREST
-   **[Optimizers](geometry_optimizers.md)**: Optimize geometries with various methods
-   **[Calculators](calculators.md)**: Set up and run energy/property calculations
-   **[Modification](modification.md)**: Modify structures (protonation, deprotonation)
-   **[Base Classes](base_nodes.md)**: Core framework classes

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
