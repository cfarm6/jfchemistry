"""Base class for calculators."""

from dataclasses import dataclass
from typing import Optional

from pymatgen.core import SiteCollection


@dataclass
class Calculator:
    """Base class for calculators."""

    charge: Optional[int | float] = None
    spin_multiplicity: Optional[int] = None

    def set_properties(self, structure: SiteCollection) -> None:
        """Set the properties for the structure."""


@dataclass
class WavefunctionCalculator(Calculator):
    """Base Class for Wavefunction Based Calculators."""


@dataclass
class SemiempiricalCalculator(Calculator):
    """Base Class for Semiempirical Calculators."""


@dataclass
class MachineLearnedInteratomicPotentialCalculator(Calculator):
    """Base Class for Machine Learned Interatomic Potential Calculators."""
