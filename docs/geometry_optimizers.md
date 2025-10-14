---
title: Geometry Optimizers
summary: Optimize molecular geometries using various methods
---

# Geometry Optimizers

Geometry optimizers perform structure relaxation to find local minima on the potential energy surface using various computational methods.

::: jfchemistry.optimizers
options:
show_root_heading: true
show_source: true
show_root_toc_entry: true
members: true
show_bases: true
show_inheritance_diagram: true

## Example Usage

### Optimizing a Molecule

```python
from jfchemistry.optimizers import TBLiteOptimizer
from jfchemistry.inputs import Smiles

smiles = Smiles()
smiles_job = smiles.make("CCO")
opt = TBLiteOptimizer(
    method="GFN2-xTB",
    optimizer="LBFGS",
    fmax=0.05,
    accuracy=1.0
)
job = opt.make(smiles_job.output["structure"])
optimized = job.output["structure"]
```

## Key Parameters

-   **optimizer**: Optimizer to use for optimization
-   **fmax**: Maximum force for optimization
-   **accuracy**: Accuracy for optimization
