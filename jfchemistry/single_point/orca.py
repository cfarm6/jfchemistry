"""Geometry optimization using ORCA DFT calculator.

This module provides fast geometry optimization using the ORCA DFT calculator
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from opi.core import Calculator
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator, ORCAProperties
from jfchemistry.core.makers.base_maker import JFChemistryBaseMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointCalculation


@dataclass
class ORCASinglePointCalculator[InputType: Molecule, OutputType: Molecule](
    SinglePointCalculation, JFChemistryBaseMaker[InputType, OutputType], ORCACalculator
):
    """Calculate the single point energy of a structure using ORCA DFT calculator.

    Inherits all attributes from ORCACalculator.

    Attributes:
        name: Name of the calculator (default: "ORCA Single Point Calculator").
        basename: Basename of the calculator (default: "orca_single_point").
    """

    name: str = "ORCA Single Point Calculator"
    basename: str = "orca_single_point"
    _properties_model: type[ORCAProperties] = ORCAProperties

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Calculate the single point energy of a molecule using ORCA."""
        # Write to XYZ file
        structure.to("input.xyz", fmt="xyz")
        # Get the default calculator SK_list
        sk_list = super()._set_keywords()
        # Make the calculator
        calc = Calculator(basename=self.basename, working_dir=Path("."))
        calc.structure = Structure.from_xyz("input.xyz")
        calc.input.add_simple_keywords(*sk_list)
        calc.input.ncores = self.cores
        calc.write_input()
        # Run the calculator
        calc.run()
        # Parse the output
        output = calc.get_output()
        properties = super()._parse_output(output)
        return cast("OutputType", structure), properties
