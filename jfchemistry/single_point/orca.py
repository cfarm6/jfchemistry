"""Geometry optimization using ORCA DFT calculator.

This module provides fast geometry optimization using the ORCA DFT calculator
"""

from dataclasses import dataclass
from pathlib import Path

from opi.core import Calculator
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator, ORCAProperties
from jfchemistry.core.makers.single_molecule import SingleMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointEnergyCalculator


@dataclass
class ORCASinglePointCalculator(SinglePointEnergyCalculator, SingleMoleculeMaker, ORCACalculator):
    """Calculate the single point energy of a structure using ORCA DFT calculator.

    Inherits all attributes from ORCACalculator.

    Attributes:
        name: Name of the calculator (default: "ORCA Single Point Calculator").
        Additional attributes inherited from ORCACalculator.

    """

    name: str = "ORCA Single Point Calculator"
    _basename: str = "orca_single_point"
    _properties_model: type[ORCAProperties] = ORCAProperties

    def operation(
        self, molecule: Molecule
    ) -> tuple[Molecule | list[Molecule], Properties | list[Properties]]:
        """Calculate the single point energy of a molecule using ORCA."""
        # Write to XYZ file
        molecule.to("input.xyz", fmt="xyz")
        # Get the default calculator SK_list
        sk_list = super().set_keywords()
        # Make the calculator
        calc = Calculator(basename=self._basename, working_dir=Path("."))
        calc.structure = Structure.from_xyz("input.xyz")
        calc.input.add_simple_keywords(*sk_list)
        calc.input.ncores = self.cores
        calc.write_input()
        # Run the calculator
        calc.run()
        # Parse the output
        output = calc.get_output()
        properties = super().parse_output(output)
        return molecule, properties
