"""Geometry optimization using ORB machine learning force field.

This module provides geometry optimization using Orbital Materials' ORB
machine learning force field combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.ase.orb_calculator import ORBModelCalculator
from jfchemistry.single_point.ase.base import ASESinglePointCalculator


@dataclass
class ORBModelSinglePointCalculator(ORBModelCalculator, ASESinglePointCalculator):
    """Calculate the single point energy of a structure using ORB machine learning force field.

    Inherits all attributes from ORBModelCalculator and ASESinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "ORB Model Single Point Calculator").
        Additional attributes inherited from ORBModelCalculator and ASESinglePointCalculator.

    Examples:
        >>> from jfchemistry.optimizers import ORBModelOptimizer # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>> opt_cpu = ORBModelSinglePointCalculator( # doctest: +SKIP
        ...     model="orb-v3-conservative-omol", # doctest: +SKIP
        ...     device="cpu", # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_cpu.make(molecule) # doctest: +SKIP
        >>> energy = job.output["properties"]["Global"]["Total Energy [eV]"] # doctest: +SKIP
    """

    name: str = "ORB Model Single Point Calculator"
