"""Base class for structure generation.

This module provides the base Maker class for geometry optimization workflows
in jfchemistry.
"""

from dataclasses import dataclass
from typing import Any, Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry import SingleStructureMaker


@dataclass
class GeometryOptimization(SingleStructureMaker):
    """Base Maker for optimizing a structure.

    This class serves as the base interface for all geometry optimization
    implementations in jfchemistry. Subclasses should implement the
    optimize_structure and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Geometry Optimization"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Optimize a structure.

        Args:
            structure: The molecular structure to optimize.

        Returns:
            A tuple containing the optimized molecular structure and a dictionary
            of properties from the optimization.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_properties(self, structure: SiteCollection) -> dict[str, Any]:
        """Get the properties of the structure.

        Args:
            structure: The molecular structure to extract properties from.

        Returns:
            A dictionary containing the properties of the structure.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
