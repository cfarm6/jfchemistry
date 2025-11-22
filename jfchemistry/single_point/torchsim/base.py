"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass

from pymatgen.core.structure import Structure

from jfchemistry.base_jobs import Properties
from jfchemistry.calculators.torchsim.base import TorchSimCalculator
from jfchemistry.single_point.base import SinglePointEnergyCalculator


@dataclass
class TorchSimSinglePointCalculator(SinglePointEnergyCalculator, TorchSimCalculator):
    """Base class for single point energy calculations using TorchSim calculators.

    Combines single point energy calculations with TorchSim calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various TorchSim calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").
    """

    name: str = "TorchSim Single Point Calculator"

    def operation(self, structure: Structure) -> tuple[Structure, Properties]:
        """Optimize molecular structure using ASE.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        """
        properties = self.get_properties(structure)
        return structure, properties
