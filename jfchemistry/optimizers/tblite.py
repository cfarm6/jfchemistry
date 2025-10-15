"""Geometry optimization using GFN-xTB semi-empirical methods.

This module provides geometry optimization using TBLite's implementation
of GFN-xTB semi-empirical quantum chemistry methods.
"""

from dataclasses import dataclass

from jfchemistry.calculators.tblite_calculator import TBLiteCalculator
from jfchemistry.optimizers.ase import ASEOptimizer


@dataclass
class TBLiteOptimizer(TBLiteCalculator, ASEOptimizer):
    """Optimize molecular structures using GFN-xTB methods.

    Combines TBLite's GFN-xTB semi-empirical methods with ASE optimization
    algorithms for geometry optimization with comprehensive property calculation.
    Provides an excellent balance between speed and accuracy for organic molecules.

    The optimizer computes extensive molecular properties including energies,
    partial charges, bond orders, orbital information, and HOMO-LUMO gaps
    during optimization.

    Inherits all attributes from TBLiteCalculator (method, charge, multiplicity,
    accuracy, electronic_temperature, etc.) and ASEOptimizer (optimizer, fmax, steps).

    Attributes:
        name: Name of the optimizer (default: "TBLite Optimizer").
        Additional attributes inherited from TBLiteCalculator and ASEOptimizer.

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.optimizers import TBLiteOptimizer # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>> # Standard optimization with GFN2-xTB
        >>> opt = TBLiteOptimizer( # doctest: +SKIP
        ...     method="GFN2-xTB", # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.05, # doctest: +SKIP
        ...     accuracy=1.0  # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt.make(molecule) # doctest: +SKIP
        >>> optimized = job.output["structure"] # doctest: +SKIP
        >>>
        >>> # Tight optimization with detailed properties
        >>> opt_tight = TBLiteOptimizer( # doctest: +SKIP
        ...     method="GFN2-xTB", # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.01, # doctest: +SKIP
        ...     accuracy=0.1,  # Tighter SCF # doctest: +SKIP
        ...     max_iterations=500, # doctest: +SKIP
        ...     charge=-1, # doctest: +SKIP
        ...     multiplicity=1 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_tight.make(molecule) # doctest: +SKIP
        >>> props = job.output["properties"] # doctest: +SKIP
        >>> gap = props["Global"]["HOMO-LUMO Gap [eV]"] # doctest: +SKIP
        >>> charges = props["Atomic"]["Mulliken Partial Charges [e]"] # doctest: +SKIP
        >>> bond_orders = props["Bond"]["Wiberg Bond Order"] # doctest: +SKIP
    """

    name: str = "TBLite Optimizer"
