"""Base class for structure modifications.

This module provides the abstract base class for implementing structure
modification methods in jfchemistry workflows.
"""

from dataclasses import dataclass

from jfchemistry.modification.base import StructureModification


@dataclass
class DeprotonationMaker(StructureModification):
    """Base class for structure modification methods.

    This abstract class defines the interface for structure modification
    implementations. Subclasses should implement the operation() method
    to perform specific modifications such as protonation, deprotonation,
    atom substitutions, or other structural transformations.

    Structure modifications typically generate multiple output structures
    representing different possible modification sites or states.

    Attributes:
        name: Descriptive name for the modification method.
    """

    name: str = "Deprotonation Maker"
