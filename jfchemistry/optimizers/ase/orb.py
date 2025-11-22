"""Geometry optimization using ORB machine learning force field.

This module provides geometry optimization using Orbital Materials' ORB
machine learning force field combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.ase.orb_calculator import ORBModelCalculator
from jfchemistry.optimizers.ase.ase import ASEOptimizer


@dataclass
class ORBModelOptimizer(ORBModelCalculator, ASEOptimizer):
    """Optimize molecular structures using ORB machine learning force field.

    Combines ORB's graph neural network force field with ASE optimization
    algorithms for accurate and efficient geometry optimization. Supports
    GPU acceleration and multiple precision options.

    Inherits all attributes from ORBModelCalculator (model, device, precision,
    compile) and ASEOptimizer (optimizer, fmax, steps).

    Attributes:
        name: Name of the optimizer (default: "ORB Model Optimizer").
        Additional attributes inherited from ORBModelCalculator and ASEOptimizer.

    Examples:
        >>> from jfchemistry.optimizers import ORBModelOptimizer # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>> opt_cpu = ORBModelOptimizer( # doctest: +SKIP
        ...     model="orb-v3-conservative-omol", # doctest: +SKIP
        ...     device="cpu", # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.05, # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_cpu.make(molecule) # doctest: +SKIP
    """

    name: str = "ORB Model Optimizer"
