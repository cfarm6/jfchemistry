---
title: FAQ & Troubleshooting
summary: Frequently asked questions and common issues
---

# Frequently Asked Questions

!!! note "Disclaimer"

    These are Claude 4.5 Sonnet generated questions and answers. Ask more questions and file issues to populate this page with accurate FAQs.

## Installation

### Q: Why do I get "CREST not found" errors?

**A:** CREST should be installed automatically with the default pixi environment. Make sure you've run:

```bash
pixi install
```

If you still have issues, check that CREST is in your PATH:

```bash
which crest
```

### Q: Can I use jfchemistry without Pixi?

**A:** While Pixi is the recommended way to manage dependencies, you can install jfchemistry with pip:

```bash
pip install -e .
```

However, you'll need to manually install system dependencies like CREST, OpenBabel, etc.

### Q: Which optional features should I install?

**A:** It depends on your needs:

-   **aimnet2**: Fast neural network potential for accurate energies
-   **orb**: Machine learning force fields for materials
-   **dev**: Development tools (testing, linting)
-   **docs**: Documentation building tools

## Usage

### Q: How do I process multiple molecules in parallel?

**A:** JFChemistry automatically parallelizes when you pass lists of molecules:

```python
# This will process all structures in parallel
molecules = [mol1, mol2, mol3]
optimizer = TBLiteOptimizer()
job = optimizer.make(molecules)
```

### Q: What's the difference between SingleRDMoleculeMaker and SingleStructureMaker?

**A:**

-   **SingleRDMoleculeMaker**: For RDKit molecules without 3D coordinates (e.g., from SMILES)
-   **SingleStructureMaker**: For Pymatgen structures with 3D coordinates

The workflow typically flows: SMILES → RDMolMolecule → 3D Structure → Optimization

### Q: How do I choose between different optimization methods?

**A:** Here's a quick guide:

| Method   | Speed  | Accuracy | Use Case                    |
| -------- | ------ | -------- | --------------------------- |
| GFN-FF   | Fast   | Low      | Quick pre-optimization      |
| GFN2-xTB | Medium | Medium   | General purpose, production |
| AimNet2  | Fast   | High     | Accurate energies, charges  |
| ORB      | Fast   | High     | Materials, periodic systems |

### Q: Can I use custom ASE calculators?

**A:** Yes! Use the ASEOptimizer base class:

```python
from jfchemistry.optimizers import ASEOptimizer
from your_calculator import YourCalculator

optimizer = ASEOptimizer(
    calculator=YourCalculator(),
    fmax=0.01,
    steps=1000
)
```

## Performance

### Q: How can I speed up conformer generation?

**A:** Several strategies:

1. Use RDKit for initial generation (fast but less thorough)
2. Use CREST with GFN-FF instead of GFN2 for speed
3. Reduce the energy window (`ewin` parameter)
4. Increase the number of threads

```python
# Fast conformer search
conformer_gen = CRESTConformers(
    runtype="imtd-gc",
    ewin=4.0,  # Smaller window = fewer conformers
    calculation_energy_method="gfnff",  # Faster than gfn2
    threads=8  # Use more cores
)
```

### Q: My optimization is taking too long. What can I do?

**A:** Try a multi-level approach:

```python
# Quick pre-optimization with force field
pre_opt = TBLiteOptimizer(method="GFNFF", fmax=0.1)
pre_job = pre_opt.make(structure)

# Refine with better method
final_opt = TBLiteOptimizer(method="GFN2-xTB", fmax=0.01)
final_job = final_opt.make(pre_job.output["structure"])
```

### Q: How much memory do I need?

**A:** Depends on molecule size and method:

-   Small molecules (<20 atoms): 1-2 GB
-   Medium molecules (20-50 atoms): 2-8 GB
-   Large molecules (>50 atoms): 8+ GB
-   AimNet2/ORB: Additional 2-4 GB for model loading

## Output & Results

### Q: How do I save optimized structures?

**A:** Use Pymatgen's `to()` method:

```python
structure = job.output["structure"]
structure.to(filename="optimized.xyz")
structure.to(filename="optimized.mol")

# For ASE formats
from pymatgen.io.ase import AseAtomsAdaptor
atoms = AseAtomsAdaptor.get_atoms(structure)
atoms.write("optimized.pdb")
```

### Q: What properties are calculated?

**A:** Depends on the calculator, but typically includes:

-   **Global**: Total energy, dipole moment
-   **Orbital**: HOMO, LUMO, gap (for xTB methods)
-   **Atomic**: Charges, forces
-   **Vibrational**: Frequencies (if requested)

Access via:

```python
properties = job.output["properties"]
energy = properties["Global"]["Total Energy [Eh]"]
```

### Q: Can I visualize the structures?

**A:** Yes! Use standard chemistry visualization tools:

```python
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view

atoms = AseAtomsAdaptor.get_atoms(structure)
view(atoms)  # Opens ASE GUI
```

Or export to formats for external viewers (Avogadro, PyMOL, etc.):

```python
structure.to(filename="molecule.xyz")  # Open in Avogadro
atoms.write("molecule.pdb")  # Open in PyMOL
```

## Errors & Debugging

### Q: I get "Calculator failed" errors. What's wrong?

**A:** Common causes:

1. **Invalid geometry**: Try pre-optimization with GFN-FF
2. **Wrong charge/multiplicity**: Check molecular charge and spin
3. **SCF convergence**: Some molecules need different settings
4. **Memory**: Reduce molecule size or increase available RAM

### Q: CREST jobs fail silently. How do I debug?

**A:** Check the working directory for CREST output files:

```python
# CREST writes to temporary directories
# Look for crest_*.out files for error messages
```

Enable verbose output in your workflow setup.

### Q: How do I handle charged or radical species?

**A:** Set the charge and spin multiplicity:

```python
# For H3O+ (charge=+1, singlet)
optimizer = TBLiteOptimizer(
    method="GFN2-xTB",
    charge=1,
    spin_multiplicity=1
)

# For O2 (charge=0, triplet)
optimizer = TBLiteOptimizer(
    method="GFN2-xTB",
    charge=0,
    spin_multiplicity=3
)
```

## Workflows

### Q: Can I integrate jfchemistry with jobflow/FireWorks?

**A:** Yes! JFChemistry is built on jobflow:

```python
from jobflow import Flow, run_locally

# Create workflow
jobs = [job1, job2, job3]
flow = Flow(jobs)

# Run locally
run_locally(flow)

# Or submit to FireWorks
# (requires FireWorks setup)
```

### Q: How do I chain multiple operations?

**A:** Jobs can be chained by referencing outputs:

```python
smiles_job = Smiles().make("CCO")
gen_job = RDKitGeneration().make(smiles_job.output["structure"])
opt_job = TBLiteOptimizer().make(gen_job.output["structure"])
```

The workflow manager handles dependencies automatically.

## Contributing

### Q: How can I add a new calculator?

**A:** Create a subclass of `ASECalculator`:

```python
from jfchemistry.calculators import ASECalculator

class MyCalculator(ASECalculator):
    def set_calculator(self, atoms, charge=0, spin_multiplicity=1):
        # Set up your calculator
        calc = YourASECalculator(...)
        atoms.calc = calc
        return atoms

    def get_properties(self, atoms):
        # Extract properties
        return {
            "Global": {
                "Total Energy [eV]": atoms.get_potential_energy(),
                # ...
            }
        }
```

### Q: How do I report bugs?

**A:** Open an issue on [GitHub](https://github.com/cfarm6/jfchemistry/issues) with:

1. Minimal code example to reproduce
2. Error message/traceback
3. Environment info (Python version, OS, etc.)
4. Expected vs actual behavior

### Q: Can I contribute examples?

**A:** Absolutely! Submit a PR with:

1. Working example code
2. Brief description of what it demonstrates
3. Any special requirements or dependencies

Place in the `examples/` directory.

## Getting Help

Still stuck? Try these resources:

-   **GitHub Issues**: [github.com/cfarm6/jfchemistry/issues](https://github.com/cfarm6/jfchemistry/issues)
-   **Jobflow Docs**: [materialsproject.github.io/jobflow](https://materialsproject.github.io/jobflow/)
-   **ASE Docs**: [wiki.fysik.dtu.dk/ase](https://wiki.fysik.dtu.dk/ase/)
