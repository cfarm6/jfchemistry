"""Geometry optimization using ORCA DFT calculator.

This module provides fast geometry optimization using the ORCA DFT calculator
"""

from dataclasses import dataclass, field
from typing import Optional, cast

from opi.input.simple_keywords.opt import Opt
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.calculators.orca.orca_keywords import OptModelType
from jfchemistry.core.makers.pymatgen_maker import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ORCAOptimizer[InputType: Molecule, OutputType: Molecule](
    ORCACalculator, GeometryOptimization, PymatGenMaker[InputType, OutputType]
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
    steps: int = field(
        default=250000,
        metadata={
            "description": "Maximum optimization steps. Set to 0 for fixed-geometry evaluation."
        },
    )
    _basename: str = "orca_optimizer"

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Optimize a molecule using ORCA DFT calculator."""
        input.to("input.xyz", fmt="xyz")
        sk_list = super()._set_keywords()
        if self.steps != 0:
            for opt_kw in self.opt or []:
                sk_list.append(getattr(Opt, opt_kw))
        calc = super()._build_calculator(self._basename)
        calc.structure = Structure.from_xyz("input.xyz")
        super()._set_structure_charge_and_spin(calc, input.charge, input.spin_multiplicity)
        super()._configure_calculator_input(calc, sk_list)
        calc.write_input()
        calc.run()
        output = calc.get_output()
        properties = super()._parse_output(output)
        final_molecule = Molecule.from_file(output.get_file(".xyz"))
        final_molecule = cast("OutputType", final_molecule)
        return final_molecule, properties
