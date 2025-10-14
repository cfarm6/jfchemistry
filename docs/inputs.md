---
title: Input Types
summary: Create molecules from various chemical identifiers
---

# Input Types

Input nodes create RDKit molecules from various chemical identifiers and representations. These serve as the entry points for computational chemistry workflows.

::: jfchemistry.inputs
options:
show_root_heading: true
show_source: true
show_root_toc_entry: true
members: true
show_bases: true
show_inheritance_diagram: true

## Example Usage

### Creating Molecules from SMILES

```python
from jfchemistry.inputs import Smiles

# Basic usage
smiles = Smiles()
job = smiles.make("CCO")  # Ethanol
molecule = job.output["structure"]

# With options
smiles = Smiles(
    add_hydrogens=True,  # Add explicit hydrogens
    remove_salts=True     # Remove salt fragments
)

# Multiple molecules
ethanol = smiles.make("CCO")
benzene = smiles.make("c1ccccc1")
water = smiles.make("O")
```

### Retrieving from PubChem

```python
from jfchemistry.inputs import PubChemCID

# Get molecule by CID
pubchem = PubChemCID()
job = pubchem.make(702)  # Ethanol CID

molecule = job.output["structure"]

# Common molecule CIDs:
# 702 - Ethanol
# 241 - Benzene
# 962 - Water
# 6324 - Glucose
# 5950 - Aspirin
```

## Building Workflows

Input nodes are typically the first step in a workflow:

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration
from jfchemistry.optimizers import TBLiteOptimizer

# Create workflow
smiles = Smiles(add_hydrogens=True)
generator = RDKitGeneration(num_conformers=10)
optimizer = TBLiteOptimizer(method="GFN2-xTB")

# Chain jobs
smiles_job = smiles.make("CC(C)CC(C)C")
gen_job = generator.make(smiles_job.output["structure"])
opt_job = optimizer.make(gen_job.output["structure"])

# Access final results
optimized_structures = opt_job.output["structure"]
energies = opt_job.output["properties"]
```

## Tips

1. **Use explicit hydrogens**: Enable `add_hydrogens=True` when you need accurate 3D geometries
2. **Clean up structures**: Use `remove_salts=True` to handle molecules with counterions
3. **Validate SMILES**: Invalid SMILES strings will raise an error - use RDKit's sanitization
4. **PubChem CIDs**: Search PubChem database online to find CIDs for your molecules of interest
