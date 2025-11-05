"""Calculators for computing molecular properties.

This module provides calculator interfaces for various computational chemistry
methods. Calculators set up the computational method for a structure and
extract properties from the calculations.

Available Calculators:
    - ASECalculator: Base class for ASE-compatible calculators
    - AimNet2Calculator: Neural network potential for fast energy/charge calculations
    - ORBModelCalculator: Orbital Materials machine learning force field
    - TBLiteCalculator: Tight-binding extended Hückel theory (xTB methods)

Examples:
    >>> from jfchemistry.calculators import TBLiteCalculator # doctest: +SKIP
    >>> from pymatgen.core import Molecule # doctest: +SKIP
    >>>
    >>> # Setup calculator
    >>> calc = TBLiteCalculator(method="GFN2-xTB") # doctest: +SKIP
    >>>
    >>> # Set calculator on ASE atoms
    >>> atoms = molecule.to_ase_atoms() # doctest: +SKIP
    >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
    >>>
    >>> # Get properties
    >>> properties = calc.get_properties(atoms) # doctest: +SKIP
    >>> energy = properties["Global"]["Total Energy [eV]"] # doctest: +SKIP
"""

from .aimnet2_calculator import AimNet2Calculator
from .ase_calculator import ASECalculator
from .crest import CRESTCalculator
from .orb_calculator import ORBModelCalculator
from .orca_calculator import (
    BasisSetType,
    ECPType,
    ORCACalculator,
    SolvationModelType,
    SolvationType,
    SolventType,
    XCFunctionalType,
)
from .tblite_calculator import TBLiteCalculator

__all__ = [
    "ASECalculator",
    "AimNet2Calculator",
    "BasisSetType",
    "CRESTCalculator",
    "ECPType",
    "ORBModelCalculator",
    "ORCACalculator",
    "SolvationModelType",
    "SolvationType",
    "SolventType",
    "TBLiteCalculator",
    "XCFunctionalType",
]
