"""TorchSim-based single point energy calculations.

This module provides single point energy calculations using TorchSim calculators.
"""

from dataclasses import dataclass, field
from typing import cast

from pymatgen.core import Molecule, Structure

from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointCalculation


@dataclass
class TorchSimSinglePoint[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    SinglePointCalculation, PymatGenMaker[InputType, OutputType], TorchSimCalculator
):
    """Base class for single point energy calculations using TorchSim calculators.

    Combines single point energy calculations with TorchSim calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various TorchSim calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").
    """

    name: str = "TorchSim Single Point Calculator"
    calculator: TorchSimCalculator = field(
        default_factory=lambda: TorchSimCalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Calculate the single point energy of a structure using TorchSim.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            input: Input molecular structure with 3D coordinates.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        """
        properties = self.calculator._get_properties(input)
        return cast("OutputType", input), properties
