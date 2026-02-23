"""This module provides the base Maker class for geometry optimization workflows in jfchemistry."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GeometryOptimization:
    """Base Maker for optimizing a structure.

    This class serves as the base interface for all geometry optimization
    implementations in jfchemistry. Subclasses should implement the
    optimize_structure and get_properties methods.

    Standard optimizer interface for jfchemistry workflows:
        - ``name``: Human-readable optimizer label.
        - ``charge``: Optional charge override used by charge-aware optimizers/calculators.
        - ``spin_multiplicity``: Optional spin multiplicity override (2S+1).
        - ``steps``: Maximum optimization steps. Set to ``0`` for fixed-geometry
          single-point style evaluation when supported.

    New optimizer implementations should expose these attributes to ensure
    compatibility with workflows that dispatch multiple electronic states (e.g.,
    Nelson's four-point method).
    """

    name: str = "Geometry Optimization"
    charge: Optional[int | float] = None
    spin_multiplicity: Optional[int] = None
    steps: int = 250000
