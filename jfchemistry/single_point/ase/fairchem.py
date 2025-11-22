"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.ase.fairchem_calculator import FairChemCalculator
from jfchemistry.single_point.ase.base import ASESinglePointCalculator


@dataclass
class FairChemSinglePointCalculator(FairChemCalculator, ASESinglePointCalculator):
    """Calculate the single point energy of a structure using FairChem neural network potential.

    Inherits all attributes from FairChemCalculator and ASESinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "FairChem Single Point Calculator").
        Additional attributes inherited from FairChemCalculator and ASESinglePointCalculator.

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.single_point import FairChemSinglePointCalculator # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>>
        >>> # Fast optimization for screening
        >>> opt_fast = FairChemSinglePointCalculator( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.1,  # Looser convergence # doctest: +SKIP
        ...     steps=500 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_fast.make(molecule) # doctest: +SKIP
        >>>
        >>> # Tight optimization
        >>> opt_tight = FairChemSinglePointCalculator( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.01, # doctest: +SKIP
        ...     charge=-1, # doctest: +SKIP
        ...     multiplicity=1 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_tight.make(molecule) # doctest: +SKIP
        >>> optimized = job.output["structure"] # doctest: +SKIP
        >>> energy = job.output["properties"]["Global"]["Total Energy [eV]"] # doctest: +SKIP
    """

    name: str = "FairChem Single Point Calculator"
