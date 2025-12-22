"""Base class for structure generation.

This module provides the base Maker class for geometry optimization workflows
in jfchemistry.
"""

from dataclasses import dataclass

from jobflow.core.maker import Maker


@dataclass
class GeometryOptimization(Maker):
    """Base Maker for optimizing a structure.

    This class serves as the base interface for all geometry optimization
    implementations in jfchemistry. Subclasses should implement the
    optimize_structure and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Geometry Optimization"
