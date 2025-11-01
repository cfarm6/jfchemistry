"""Geometry optimization methods for molecular structures.

This module provides geometry optimization workflows using various computational
methods including neural network potentials, machine learning force fields, and
semi-empirical quantum chemistry.

Available Optimizers:
    - GeometryOptimization: Base class for geometry optimization
    - ASEOptimizer: Base class for ASE-based optimizers
    - AimNet2Optimizer: Neural network potential optimizer
    - ORBModelOptimizer: Machine learning force field optimizer
    - TBLiteOptimizer: GFN-xTB semi-empirical optimizer

Examples:
    >>> from jfchemistry.optimizers import TBLiteOptimizer
    >>> from ase.build import molecule
    >>> from pymatgen.core import Molecule
    >>> molecule = Molecule.from_ase_atoms(molecule("CCH"))
    >>>
    >>> # Optimize with GFN2-xTB
    >>> optimizer = TBLiteOptimizer(
    ...     method="GFN2-xTB",
    ...     fmax=0.01,
    ...     steps=1000
    ... )
    >>> job = optimizer.make(molecule)
    >>> optimized = job.output["structure"]
    >>> properties = job.output["properties"]
"""

from .aimnet2 import AimNet2Optimizer
from .ase import ASEOptimizer
from .orb import ORBModelOptimizer
from .orca import ORCAOptimizer
from .tblite import TBLiteOptimizer

__all__ = [
    "ASEOptimizer",
    "AimNet2Optimizer",
    "ORBModelOptimizer",
    "ORCAOptimizer",
    "TBLiteOptimizer",
]
