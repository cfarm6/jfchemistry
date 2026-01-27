"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field
from typing import cast

from pymatgen.core import Molecule, Structure

from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.makers.base_maker import JFChemistryBaseMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointCalculation


@dataclass
class TorchSimSinglePoint[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    SinglePointCalculation, JFChemistryBaseMaker[InputType, OutputType], TorchSimCalculator
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
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Calculate the single point energy of a structure using TorchSim.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        """
        properties = self.calculator._get_properties(structure)
        return cast("OutputType", structure), properties
