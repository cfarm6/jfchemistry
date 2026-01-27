"""Example of using the PubChemCID node to get a molecule from PubChem."""

import inspect
from importlib import import_module
from pkgutil import walk_packages

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.single_point import ASESinglePoint


# Dynamically discover all Calculator subclasses
def discover_calculators():
    """Discover all concrete Calculator subclasses from the calculators module."""
    calculators = []
    # Abstract base classes to exclude
    abstract_class = ASECalculator

    # Import the calculators module to ensure all subclasses are registered
    calculators_module = import_module("jfchemistry.calculators.ase")
    imported_modules = {calculators_module}

    # Walk through all submodules to ensure they're imported
    for _, modname, _ in walk_packages(
        calculators_module.__path__, calculators_module.__name__ + "."
    ):
        try:
            mod = import_module(modname)
            imported_modules.add(mod)
        except (ImportError, AttributeError):
            continue

    # Find all Calculator subclasses across all imported modules
    seen_classes = set()
    for module in imported_modules:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            print("--------------------------------")
            print(obj.__name__)
            print(issubclass(obj, ASECalculator))
            print(obj is not ASECalculator)
            print("--------------------------------")
            if (
                obj not in seen_classes
                and issubclass(obj, ASECalculator)
                and obj is not ASECalculator
            ):
                print(obj.__name__)
                # Check if it's a concrete class (not abstract base)
                if not inspect.isabstract(obj):
                    calculators.append(obj)
                    seen_classes.add(obj)

    return calculators


# Get all calculator classes
calculator_classes = discover_calculators()
print(calculator_classes)
pubchem_cid = Smiles().make("C(CO)Cl")

generate_structure = RDKitGeneration(num_conformers=1).make(pubchem_cid.output.structure)

# Instantiate calculators (with default parameters where possible)
calculators = []
for calc_class in calculator_classes:
    try:
        calculators.append(calc_class())
    except (TypeError, ValueError) as e:
        # Skip calculators that can't be instantiated with defaults
        print(f"Skipping {calc_class.__name__}: {e}")
        continue

# Create optimizer jobs for each calculator
opts = []
for calculator in calculators:
    opt = ASESinglePoint(
        calculator=calculator,
    ).make(generate_structure.output.structure)
    opts.append(opt)


flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        *opts,
    ]
)

response = run_locally(flow)
