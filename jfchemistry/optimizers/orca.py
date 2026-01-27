"""Geometry optimization using ORCA DFT calculator.

This module provides fast geometry optimization using the ORCA DFT calculator
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast

from opi.core import Calculator
from opi.input.simple_keywords.opt import Opt
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.calculators.orca.orca_keywords import OptModelType
from jfchemistry.core.makers.base_maker import JFChemistryBaseMaker
from jfchemistry.core.properties import Properties
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ORCAOptimizer[InputType: Molecule, OutputType: Molecule](
    ORCACalculator, GeometryOptimization, JFChemistryBaseMaker[InputType, OutputType]
):
    """Optimize molecular structures using ORCA DFT calculator.

    Inherits all attributes from ORCACalculator.

    Attributes:
        name: Name of the optimizer (default: "Orca Optimizer").
        opt: The ORCA optimizer to use for the calculation (default: ["OPT"]).
    """

    name: str = "Orca Optimizer"
    opt: Optional[list[OptModelType]] = field(
        default_factory=lambda: ["OPT"],
        metadata={"description": "the ORCA optimizer to use for the calculation"},
    )
    _basename: str = "orca_optimizer"

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Optimize a molecule using ORCA DFT calculator."""
        # Write to XYZ file
        structure.to("input.xyz", fmt="xyz")
        # Get the default calculator SK_list
        sk_list = super()._set_keywords()
        # Add the optimizer keywords
        for opt_kw in self.opt or []:
            sk_list.append(getattr(Opt, opt_kw))  # type: ignore
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
        properties = super()._parse_output(output)
        final_molecule = Molecule.from_file(output.get_file(".xyz"))
        final_molecule = cast("OutputType", final_molecule)
        return final_molecule, properties
