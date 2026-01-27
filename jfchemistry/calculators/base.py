"""Base class for calculators."""

from dataclasses import dataclass
from typing import Optional

from monty.json import MSONable

from jfchemistry.core.properties import Properties


@dataclass
class Calculator(MSONable):
    """Base class for calculators."""

    charge: Optional[int | float] = None
    spin_multiplicity: Optional[int] = None
    _properties_model: type[Properties] = Properties

    def _get_properties(*args, **kwargs) -> Properties:
        """Set the properties for the structure."""
        raise NotImplementedError


@dataclass
class WavefunctionCalculator(Calculator):
    """Base Class for Wavefunction Based Calculators."""


@dataclass
class SemiempiricalCalculator(Calculator):
    """Base Class for Semiempirical Calculators."""


@dataclass
class MachineLearnedInteratomicPotentialCalculator(Calculator):
    """Base Class for Machine Learned Interatomic Potential Calculators."""
