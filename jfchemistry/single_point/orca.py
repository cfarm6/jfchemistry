"""Single point energy calculations using ORCA DFT calculator.

This module provides single point energy calculations using the ORCA DFT calculator.
"""

from dataclasses import dataclass
from typing import cast

from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator, ORCAProperties
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointCalculation


@dataclass
class ORCASinglePointCalculator[InputType: Molecule, OutputType: Molecule](
    SinglePointCalculation, PymatGenMaker[InputType, OutputType], ORCACalculator
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
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Calculate the single point energy of a molecule using ORCA."""
        input.to("input.xyz", fmt="xyz")
        sk_list = super()._set_keywords()
        calc = super()._build_calculator(self.basename)
        calc.structure = Structure.from_xyz("input.xyz")
        super()._set_structure_charge_and_spin(calc, input.charge, input.spin_multiplicity)
        super()._configure_calculator_input(calc, sk_list)
        calc.write_input()
        calc.run()
        output = calc.get_output()
        properties = super()._parse_output(output)
        return cast("OutputType", input), properties
