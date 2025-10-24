"""Base class for calculators."""

from dataclasses import dataclass
from typing import Optional

from pymatgen.core import SiteCollection


@dataclass
class Calculator:
    """Base class for calculators."""

    charge: Optional[int] = None
    spin_multiplicity: Optional[int] = None

    def set_properties(self, structure: SiteCollection) -> None:
        """Set the properties for the structure."""
