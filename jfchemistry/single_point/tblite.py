"""Geometry optimization using GFN-xTB semi-empirical methods.

This module provides geometry optimization using TBLite's implementation
of GFN-xTB semi-empirical quantum chemistry methods.
"""

from dataclasses import dataclass

from jfchemistry.calculators.tblite_calculator import TBLiteCalculator
from jfchemistry.single_point.ase import ASESinglePointCalculator


@dataclass
class TBLiteSinglePointCalculator(TBLiteCalculator, ASESinglePointCalculator):
    """Calculate the single point energy of a structure using GFN-xTB methods.

    Inherits all attributes from TBLiteCalculator and ASESinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "TBLite Single Point Calculator").
        Additional attributes inherited from TBLiteCalculator and ASESinglePointCalculator.
    """

    name: str = "TBLite Single Point Calculator"
