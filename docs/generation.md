---
title: 3D Structure Generation
summary: Generate 3D molecular structures from 2D representations
---

# 3D Structure Generation

Structure generation methods convert molecular representations without 3D coordinates (like SMILES strings or molecular graphs) into 3D structures with embedded conformers.

::: jfchemistry.generation
options:
show_root_heading: true
show_source: true
show_root_toc_entry: true
members: true
show_bases: true
show_inheritance_diagram: true

## Example Usage

```python
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration

# Create molecule from SMILES
smiles = Smiles(add_hydrogens=True)
smiles_job = smiles.make("c1ccccc1")  # Benzene

# Generate multiple 3D conformers
generator = RDKitGeneration(
    method="ETKDGv3",  # Use ETKDG v3 algorithm
    num_conformers=50,
    prune_rms_thresh=0.5,  # Remove similar conformers
    randomSeed=42,
    useRandomCoords=True
)

job = generator.make(smiles_job.output["structure"])

# Access generated structures
structures = job.output["structure"]  # List of Pymatgen Molecule objects
files = job.output["files"]  # XYZ format files
```

## Key Parameters

-   **method**: Embedding algorithm to use
-   **num_conformers**: Number of conformers to generate
-   **prune_rms_thresh**: RMS threshold for pruning similar conformers (in Angstroms)
-   **randomSeed**: Random seed for reproducibility
-   **useRandomCoords**: Start from random coordinates
-   **maxAttempts**: Maximum attempts per conformer generation
