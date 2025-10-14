---
title: Conformer Generation
summary: Generate and explore conformational space
---

# Conformer Generation

Conformer generation methods explore the conformational space of molecules to identify energetically favorable conformations. This is crucial for understanding molecular flexibility and finding global minimum structures.

::: jfchemistry.conformers
options:
show_root_heading: true
show_source: true
show_root_toc_entry: true
members: true
show_bases: true
show_inheritance_diagram: true

## Example Usage

```python
from jfchemistry.conformers import CRESTConformers
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration

# Create molecule and generate initial structure
smiles = Smiles()
smiles_job = smiles.make("CC(C)C(C)C")  # Branched alkane

generator = RDKitGeneration(num_conformers=1)
gen_job = generator.make(smiles_job.output["structure"])

# Generate conformers with CREST
conformer_gen = CRESTConformers(
    runtype="imtd-gc",
    ewin=6.0,  # Energy window in kcal/mol
    calculation_energy_method="gfnff",
    calculation_dynamics_method="gfnff",
    threads=4
)

job = conformer_gen.make(gen_job.output["structure"])

# Access results
conformers = job.output["structure"]  # List of conformers
properties = job.output["properties"]
```

## Key Parameters

-   **runtype**: Type of CREST calculation (e.g., "imtd-gc" for iterative metadynamics)
-   **ewin**: Energy window in kcal/mol for conformer selection
-   **calculation_energy_method**: Method for energy calculations (e.g., "gfnff", "gfn2")
-   **calculation_dynamics_method**: Method for molecular dynamics
-   **threads**: Number of parallel threads to use
