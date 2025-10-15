"""Base class for calculators."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Calculator:
    """Base class for calculators."""

    charge: Optional[int] = None
    spin_multiplicity: Optional[int] = None
