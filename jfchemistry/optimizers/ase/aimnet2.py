"""Geometry optimization using AimNet2 neural network potential.

This module provides fast geometry optimization using the AimNet2 neural
network potential combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.ase.aimnet2_calculator import AimNet2Calculator
from jfchemistry.optimizers.ase.ase import ASEOptimizer


@dataclass
class AimNet2Optimizer(AimNet2Calculator, ASEOptimizer):
    """Optimize molecular structures using AimNet2 neural network potential.

    Combines AimNet2's fast neural network energy/force predictions with
    ASE optimization algorithms for efficient geometry optimization. Ideal
    for large molecular systems where speed is critical.

    Inherits all attributes from AimNet2Calculator (model, charge, multiplicity)
    and ASEOptimizer (optimizer, fmax, steps).

    Attributes:
        name: Name of the optimizer (default: "AimNet2 Optimizer").
        Additional attributes inherited from AimNet2Calculator and ASEOptimizer.

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.optimizers import AimNet2Optimizer # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>>
        >>> # Fast optimization for screening
        >>> opt_fast = AimNet2Optimizer( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.1,  # Looser convergence # doctest: +SKIP
        ...     steps=500 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_fast.make(molecule) # doctest: +SKIP
        >>>
        >>> # Tight optimization
        >>> opt_tight = AimNet2Optimizer( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.01, # doctest: +SKIP
        ...     charge=-1, # doctest: +SKIP
        ...     multiplicity=1 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_tight.make(molecule) # doctest: +SKIP
        >>> optimized = job.output["structure"] # doctest: +SKIP
        >>> energy = job.output["properties"]["Global"]["Total Energy [eV]"] # doctest: +SKIP
    """

    name: str = "AimNet2 Optimizer"
