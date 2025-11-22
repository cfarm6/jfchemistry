"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.torchsim.fairchem_ts_calculator import FairChemTSCalculator
from jfchemistry.single_point.torchsim.base import TorchSimSinglePointCalculator


@dataclass
class FairChemTorchSimSinglePointCalculator(FairChemTSCalculator, TorchSimSinglePointCalculator):
    """Calculate the single point energy of a structure using FairChem neural network potential.

    Inherits all attributes from FairChemTSCalculator and TorchSimSinglePointCalculator.

    Attributes:
        name: Name of the calculator (default: "FairChem TorchSim Single Point Calculator").
        Additional attributes inherited from FairChemTSCalculator and TorchSimSinglePointCalculator.

    """

    name: str = "FairChem TorchSim Single Point Calculator"
