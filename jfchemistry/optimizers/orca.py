"""Geometry optimization using ORCA DFT calculator.

This module provides fast geometry optimization using the ORCA DFT calculator
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from opi.core import Calculator
from opi.input.simple_keywords.opt import Opt
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca_calculator import ORCACalculator, ORCAProperties
from jfchemistry.calculators.orca_keywords import OptModelType
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ORCAOptimizer(ORCACalculator, GeometryOptimization):
    """Optimize molecular structures using ORCA DFT calculator.

    Inherits all attributes from ORCACalculator.

    Attributes:
        name: Name of the optimizer (default: "Orca Optimizer").
        Additional attributes inherited from ORCACalculator.

    """

    name: str = "Orca Optimizer"
    opt: Optional[list[OptModelType]] = field(default_factory=lambda: ["OPT"])
    _basename: str = "orca_optimizer"

    def operation(self, molecule: Molecule) -> tuple[Molecule, ORCAProperties]:
        """Optimize a molecule using ORCA DFT calculator."""
        # Write to XYZ file
        molecule.to("input.xyz", fmt="xyz")
        # Get the default calculator SK_list
        sk_list = super().set_keywords()
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
        properties = super().parse_output(output)
        final_molecule = Molecule.from_file(output.get_file(".xyz"))
        return final_molecule, properties
