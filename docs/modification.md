---
title: Structure Modification
summary: Modify molecular structures (protonation, deprotonation)
---

# Structure Modification

Structure modification methods allow chemical transformations of molecules, including protonation state changes and other structural alterations.

::: jfchemistry.modification
options:
show_root_heading: true
show_source: true
show_root_toc_entry: true
members: true
show_bases: true
show_inheritance_diagram: true

## Example Usage

### Protonation

```python
from jfchemistry.modification import CRESTProtonation
from jfchemistry.inputs import Smiles
from jfchemistry.generation import RDKitGeneration

# Create molecule
smiles = Smiles()
smiles_job = smiles.make("CCO")  # Ethanol

# Generate 3D structure
gen = RDKitGeneration(num_conformers=1)
gen_job = gen.make(smiles_job.output["structure"])

# Protonate the molecule
protonation = CRESTProtonation(
    ewin=6.0,
    calculation_energy_method="gfn2",
    threads=4
)

job = protonation.make(gen_job.output["structure"])

# Access protonated structures
protonated = job.output["structure"]  # List of protonated conformers
properties = job.output["properties"]
```

### Deprotonation

```python
from jfchemistry.modification import CRESTDeprotonation

# Deprotonate a molecule
deprotonation = CRESTDeprotonation(
    ewin=6.0,
    calculation_energy_method="gfn2",
    threads=4
)

job = deprotonation.make(gen_job.output["structure"])

# Access deprotonated structures
deprotonated = job.output["structure"]
properties = job.output["properties"]
```

## Key Parameters

-   **ewin**: Energy window in kcal/mol for selecting structures
-   **calculation_energy_method**: Method for energy evaluation (e.g., "gfn2", "gfnff")
-   **threads**: Number of parallel threads to use

## Use Cases

-   **pH-dependent chemistry**: Model molecules at different pH values
-   **Ionizable groups**: Identify favorable ionization states
-   **Reaction mechanisms**: Explore proton transfer pathways
-   **pKa predictions**: Generate structures for pKa calculations
