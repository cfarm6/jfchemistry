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
        >>> from ase.build import molecule
        >>> from pymatgen.core import Molecule
        >>> from jfchemistry.optimizers import TBLiteOptimizer
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH"))
        >>> # Standard optimization with GFN2-xTB
        >>> opt = TBLiteOptimizer(
        ...     method="GFN2-xTB",
        ...     optimizer="LBFGS",
        ...     fmax=0.05,
        ...     accuracy=1.0
        ... )
        >>> job = opt.make(molecule)
        >>> optimized = job.output["structure"]
        >>>
        >>> # Tight optimization with detailed properties
        >>> opt_tight = TBLiteOptimizer(
        ...     method="GFN2-xTB",
        ...     optimizer="LBFGS",
        ...     fmax=0.01,
        ...     accuracy=0.1,  # Tighter SCF
        ...     max_iterations=500,
        ...     charge=-1,
        ...     multiplicity=1
        ... )
        >>> job = opt_tight.make(molecule)
        >>> props = job.output["properties"]
        >>> gap = props["Global"]["HOMO-LUMO Gap [eV]"]
        >>> charges = props["Atomic"]["Mulliken Partial Charges [e]"]
        >>> bond_orders = props["Bond"]["Wiberg Bond Order"]
    """

    name: str = "TBLite Optimizer"
