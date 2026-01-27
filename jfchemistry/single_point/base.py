"""Base class for single point energy calculations.

This module provides the base Maker class for single point energy calculations
in jfchemistry.
"""

from dataclasses import dataclass


@dataclass
class SinglePointCalculation:
    """Base Maker for calculating the single point energy of a structure.

    This class serves as the base interface for all single point energy calculation
    implementations in jfchemistry. Subclasses should implement the
    operation and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Single Point Energy Calculator"
