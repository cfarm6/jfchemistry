---
title: User Guide
summary: Complete guide to using JFChemistry
---

# User Guide

This guide provides comprehensive examples of using JFChemistry to build computational chemistry workflows.

## Basic Concepts

JFChemistry is built on [jobflow](https://github.com/materialsproject/jobflow), which enables creating complex, parallelizable workflows. The package is organized around two main base classes:

-   **SingleMoleculeMaker**: For operations on molecules without 3D coordinates (RDKit molecules)
-   **SingleStructureMaker**: For operations on structures with 3D coordinates (Pymatgen structures)

All workflow components in JFChemistry inherit from these base classes and follow a consistent interface.

## Installation

Install using Pixi (recommended):

```bash
git clone https://github.com/cfarm6/jfchemistry.git
cd jfchemistry
pixi install
```

For specific features:

```bash
# For AimNet2 neural network potentials
pixi install -e aimnet2

# For ORB machine learning models
pixi install -e orb

# For development
pixi install -e dev
```

## Workflow Examples

### Example 1: Simple Geometry Optimization

The simplest workflow: create a molecule from SMILES and optimize its geometry.

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import TBLiteOptimizer

# Create molecule from SMILES
smiles_maker = Smiles(add_hydrogens=True)
smiles_job = smiles_maker.make("CCO")  # Ethanol

# Generate 3D structure
generator = RDKitGeneration(num_conformers=1, method="ETKDGv3")
gen_job = generator.make(smiles_job.output["structure"])

# Optimize with GFN2-xTB
optimizer = TBLiteOptimizer(method="GFN2-xTB", fmax=0.01, steps=1000)
opt_job = optimizer.make(gen_job.output["structure"])

# Access results
optimized_structure = opt_job.output["structure"]
energy = opt_job.output["properties"]["Global"]["Total Energy [Eh]"]
print(f"Optimized energy: {energy} Eh")
```

### Example 2: Conformer Search Workflow

Generate and optimize multiple conformers to find the global minimum.

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.conformers import CRESTConformers
from jfchemistry.optimizers import TBLiteOptimizer

# Create molecule
smiles = Smiles(add_hydrogens=True)
mol_job = smiles.make("CC(C)C(C)C")  # Branched alkane

# Generate initial 3D structure
gen = RDKitGeneration(num_conformers=1)
gen_job = gen.make(mol_job.output["structure"])

# Search conformational space with CREST
conformer_search = CRESTConformers(
    runtype="imtd-gc",
    ewin=6.0,  # 6 kcal/mol energy window
    calculation_energy_method="gfnff",
    calculation_dynamics_method="gfnff",
    threads=4
)
crest_job = conformer_search.make(gen_job.output["structure"])

# Refine conformers with higher-level method
optimizer = TBLiteOptimizer(method="GFN2-xTB", fmax=0.005)
opt_job = optimizer.make(crest_job.output["structure"])

# Results
conformers = opt_job.output["structure"]  # List of optimized conformers
energies = [p["Global"]["Total Energy [Eh]"]
            for p in opt_job.output["properties"]]

print(f"Found {len(conformers)} unique conformers")
print(f"Energy range: {max(energies) - min(energies):.4f} Eh")
```

### Example 3: High-Throughput Screening

Process multiple molecules in parallel using jobflow.

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import TBLiteOptimizer
from jobflow import Flow

# Define molecules to screen
molecules = [
    "CCO",           # Ethanol
    "CC(C)O",        # Isopropanol
    "CCCO",          # 1-Propanol
    "CC(C)(C)O",     # tert-Butanol
]

# Create workflow components
smiles = Smiles(add_hydrogens=True)
generator = RDKitGeneration(num_conformers=10)
optimizer = TBLiteOptimizer(method="GFN2-xTB")

# Build jobs for all molecules
all_jobs = []
for mol_smiles in molecules:
    smiles_job = smiles.make(mol_smiles)
    gen_job = generator.make(smiles_job.output["structure"])
    opt_job = optimizer.make(gen_job.output["structure"])
    all_jobs.append(opt_job)

# Create and run flow
flow = Flow(all_jobs)
# flow.run()  # Uncomment to run with jobflow
```

### Example 4: Using PubChem Database

Retrieve molecules from PubChem and process them.

```python
from jfchemistry.inputs import PubChemCID
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import AimNet2Optimizer

# Get molecule from PubChem
pubchem = PubChemCID()
mol_job = pubchem.make(5950)  # Aspirin CID

# Generate 3D conformers
gen = RDKitGeneration(num_conformers=20, prune_rms_thresh=0.5)
gen_job = gen.make(mol_job.output["structure"])

# Optimize with AimNet2 (fast neural network potential)
optimizer = AimNet2Optimizer(fmax=0.01)
opt_job = optimizer.make(gen_job.output["structure"])

# Results include charges from AimNet2
structures = opt_job.output["structure"]
properties = opt_job.output["properties"]
```

### Example 5: Protonation State Exploration

Explore different protonation states of a molecule.

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.modification import CRESTProtonation, CRESTDeprotonation
from jfchemistry.optimizers import TBLiteOptimizer

# Create molecule (acetic acid)
smiles = Smiles(add_hydrogens=True)
mol_job = smiles.make("CC(=O)O")

# Generate 3D structure
gen = RDKitGeneration(num_conformers=1)
gen_job = gen.make(mol_job.output["structure"])

# Deprotonate (create acetate anion)
deprot = CRESTDeprotonation(
    ewin=6.0,
    calculation_energy_method="gfn2",
    threads=4
)
deprot_job = deprot.make(gen_job.output["structure"])

# Optimize deprotonated structures
optimizer = TBLiteOptimizer(method="GFN2-xTB", charge=-1)
opt_job = optimizer.make(deprot_job.output["structure"])

# Compare energies
deprotonated_structures = opt_job.output["structure"]
```

### Example 6: Custom Calculator Usage

Use calculators directly for single-point energy calculations.

```python
from jfchemistry.calculators import TBLiteCalculator
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration

# Generate structure
smiles = Smiles()
gen = RDKitGeneration(num_conformers=1)
smiles_job = smiles.make("C")
gen_job = gen.make(smiles_job.output["structure"])

# Get structure and convert to ASE
structure = gen_job.output["structure"]
atoms = structure.to_ase_atoms()

# Setup calculator
calc = TBLiteCalculator(method="GFN2-xTB")
atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1)

# Calculate properties
properties = calc.get_properties(atoms)
energy = properties["Global"]["Total Energy [eV]"]
homo = properties["Orbital"]["HOMO [eV]"]
lumo = properties["Orbital"]["LUMO [eV]"]
gap = properties["Orbital"]["HOMO-LUMO Gap [eV]"]

print(f"Energy: {energy:.4f} eV")
print(f"HOMO: {homo:.4f} eV")
print(f"LUMO: {lumo:.4f} eV")
print(f"Gap: {gap:.4f} eV")
```

### Example 7: Multi-Level Optimization

Optimize structures with increasingly accurate methods.

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import TBLiteOptimizer, AimNet2Optimizer

# Create complex molecule
smiles = Smiles(add_hydrogens=True)
mol_job = smiles.make("CC1=CC=CC=C1C(=O)O")  # o-Toluic acid

# Generate initial conformers
gen = RDKitGeneration(num_conformers=50, prune_rms_thresh=0.3)
gen_job = gen.make(mol_job.output["structure"])

# Quick pre-optimization with GFN-FF (force field)
pre_opt = TBLiteOptimizer(method="GFNFF", fmax=0.1)
pre_job = pre_opt.make(gen_job.output["structure"])

# Refine with GFN2-xTB (semi-empirical)
opt1 = TBLiteOptimizer(method="GFN2-xTB", fmax=0.01)
opt1_job = opt1.make(pre_job.output["structure"])

# Final optimization with AimNet2 (ML potential)
final_opt = AimNet2Optimizer(fmax=0.005)
final_job = final_opt.make(opt1_job.output["structure"])

# Get lowest energy conformer
energies = [p["Global"]["Total Energy [eV]"]
            for p in final_job.output["properties"]]
min_idx = energies.index(min(energies))
best_structure = final_job.output["structure"][min_idx]
```

## Working with Results

### Accessing Output

All JFChemistry jobs return a consistent output structure:

```python
job_output = job.output

# Structure(s) as Pymatgen Molecule/Structure objects
structures = job_output["structure"]

# File representations (XYZ or MOL format)
files = job_output["files"]

# Computed properties (method-dependent)
properties = job_output["properties"]
```

### Properties Dictionary Structure

Properties are organized hierarchically:

```python
properties = {
    "Global": {
        "Total Energy [Eh]": -15.234,
        "Total Energy [eV]": -414.567,
        # ... other global properties
    },
    "Orbital": {
        "HOMO [eV]": -6.5,
        "LUMO [eV]": -1.2,
        "HOMO-LUMO Gap [eV]": 5.3,
        # ... other orbital properties
    },
    # ... other property categories
}
```

### Saving Results

```python
from pymatgen.core import Molecule

# Save structure to file
structure = job.output["structure"]
structure.to(filename="output.xyz")
structure.to(filename="output.mol")

# Save to common formats
from pymatgen.io.ase import AseAtomsAdaptor
atoms = AseAtomsAdaptor.get_atoms(structure)
atoms.write("output.pdb")
```

## Tips and Best Practices

### 1. Choosing Methods

-   **Quick screening**: Use GFN-FF or GFN2-xTB
-   **Accurate energies**: Use AimNet2 or ORB models
-   **Production calculations**: Use GFN2-xTB → AimNet2 hierarchy

### 2. Conformer Generation

-   Start with RDKit (10-50 conformers) for small molecules
-   Use CREST for flexible molecules or thorough conformational search
-   Prune similar conformers with `prune_rms_thresh` parameter

### 3. Parallelization

JFChemistry automatically handles:

-   Lists of molecules → parallel processing
-   Multiple conformers → separate jobs
-   Workflow distribution via jobflow

### 4. Performance Optimization

```python
# Good: Generate many conformers, optimize in parallel
gen = RDKitGeneration(num_conformers=100)
opt = TBLiteOptimizer(method="GFN2-xTB")
gen_job = gen.make(molecule)
opt_job = opt.make(gen_job.output["structure"])  # Parallelized

# Better: Use CREST for efficient conformer search
crest = CRESTConformers(runtype="imtd-gc", threads=8)
crest_job = crest.make(structure)
```

### 5. Error Handling

```python
# Check for successful completion
if job.output["structure"] is not None:
    # Process results
    pass
else:
    # Handle failure
    print("Job failed")
```

## Next Steps

-   Explore the [API Reference](base_nodes.md) for detailed class documentation
-   Check the [examples directory](https://github.com/cfarm6/jfchemistry/tree/master/examples) for more use cases
-   Learn about [jobflow](https://materialsproject.github.io/jobflow/) for advanced workflow features

## Common Issues

### Missing Dependencies

Some calculators require optional dependencies:

```bash
# For AimNet2
pixi install -e aimnet2

# For ORB models
pixi install -e orb
```

### CREST Not Found

Ensure CREST is installed:

```bash
pixi install  # Includes CREST by default
```

### Memory Issues

For large molecules or many conformers:

-   Reduce `num_conformers` parameter
-   Process molecules in smaller batches
-   Increase available memory or use compute cluster
