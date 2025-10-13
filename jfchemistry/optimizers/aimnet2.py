"""Geometry optimization using AimNet2 neural network potential.

This module provides fast geometry optimization using the AimNet2 neural
network potential combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.aimnet2_calculator import AimNet2Calculator
from jfchemistry.optimizers.ase import ASEOptimizer


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
        >>> from ase.build import molecule
        >>> from pymatgen.core import Molecule
        >>> from jfchemistry.optimizers import AimNet2Optimizer
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH"))
        >>>
        >>> # Fast optimization for screening
        >>> opt_fast = AimNet2Optimizer(
        ...     optimizer="LBFGS",
        ...     fmax=0.1,  # Looser convergence
        ...     steps=500
        ... )
        >>> job = opt_fast.make(molecule)
        >>>
        >>> # Tight optimization
        >>> opt_tight = AimNet2Optimizer(
        ...     optimizer="LBFGS",
        ...     fmax=0.01,
        ...     charge=-1,
        ...     multiplicity=1
        ... )
        >>> job = opt_tight.make(molecule)
        >>> optimized = job.output["structure"]
        >>> energy = job.output["properties"]["Global"]["Total Energy [eV]"]
    """

    name: str = "AimNet2 Optimizer"
