"""Base class for structure packing.

This module provides the base Maker class for structure packing workflows
in jfchemistry.
"""

from dataclasses import dataclass


@dataclass
class StructurePacking:
    """Base Maker for optimizing a structure.

    This class serves as the base interface for all structure packing
    implementations in jfchemistry. Subclasses should implement the
    pack_structure and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Structure Packing"
