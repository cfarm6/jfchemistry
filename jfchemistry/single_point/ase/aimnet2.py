"""Geometry optimization using AimNet2 neural network potential.

This module provides fast geometry optimization using the AimNet2 neural
network potential combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.ase.aimnet2_calculator import AimNet2Calculator
from jfchemistry.single_point.ase.base import ASESinglePointCalculator


@dataclass
class AimNet2SinglePointCalculator(AimNet2Calculator, ASESinglePointCalculator):
    """Calculate the single point energy of a structure using AimNet2 neural network potential.

    Inherits all attributes from AimNet2Calculator and ASESinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "AimNet2 Single Point Calculator").
        Additional attributes inherited from AimNet2Calculator and ASESinglePointCalculator.

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

    name: str = "AimNet2 Single Point Calculator"
