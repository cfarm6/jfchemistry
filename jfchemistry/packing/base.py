"""Base class for structure packing.

This module provides the base Maker class for structure packing workflows
in jfchemistry.
"""

from dataclasses import dataclass

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.makers.single_molecule import SingleMoleculeMaker
from jfchemistry.core.properties import Properties


@dataclass
class StructurePacking(SingleMoleculeMaker):
    """Base Maker for optimizing a structure.

    This class serves as the base interface for all structure packing
    implementations in jfchemistry. Subclasses should implement the
    pack_structure and get_properties methods.

    Attributes:
        name: The name of the geometry optimization job.
    """

    name: str = "Structure Packing"

    def operation(
        self, molecule: Molecule
    ) -> tuple[Molecule | list[Molecule], Properties | list[Properties]]:
        """Pack a structure.

        Args:
            molecule: The molecular structure to optimize.

        Returns:
            A tuple containing the optimized molecular structure and a dictionary
            of properties from the optimization.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_properties(self, structure: Structure):
        """Get the properties of the structure.

        Args:
            structure: The molecular structure to extract properties from.

        Returns:
            A dictionary containing the properties of the structure.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
