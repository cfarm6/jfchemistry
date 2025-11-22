"""Geometry optimization using Orb neural network potential.

This module provides fast geometry optimization using the Orb neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.torchsim.orb_ts_calculator import OrbTSCalculator
from jfchemistry.single_point.torchsim.base import TorchSimSinglePointCalculator


@dataclass
class OrbTorchSimSinglePointCalculator(OrbTSCalculator, TorchSimSinglePointCalculator):
    """Calculate the single point energy of a structure using Orb neural network potential.

    Inherits all attributes from OrbTSCalculator and TorchSimSinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "Orb TorchSim Single Point Calculator").
        Additional attributes inherited from OrbTSCalculator and TorchSimSinglePointCalculator.

    """

    name: str = "Orb TorchSim Single Point Calculator"
