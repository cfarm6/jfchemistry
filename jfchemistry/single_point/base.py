"""Base class for single point energy calculations.

This module provides the base Maker class for single point energy calculations
in jfchemistry.
"""

from dataclasses import dataclass
from typing import Any, Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry import SingleStructureMaker


@dataclass
class SinglePointEnergyCalculator(SingleStructureMaker):
    """Base Maker for calculating the single point energy of a structure.

    This class serves as the base interface for all single point energy calculation
    implementations in jfchemistry. Subclasses should implement the
    operation and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Single Point Energy Calculator"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Calculate the single point energy of a structure.

        Args:
            structure: The molecular structure to calculate the single point energy of a structure.

        Returns:
            A tuple containing the molecular structure with the single point energy calculated
            and a dictionary of properties from the single point energy calculation and the energy.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
