"""GOAT Conformer Generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from opi.core import Calculator
from opi.input.simple_keywords.goat import Goat
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule
from pymatgen.io.xyz import XYZ

from jfchemistry.calculators.orca_calculator import ORCACalculator, ORCAProperties

from .base import ConformerGeneration

type GoatKeywordsType = Literal["GOAT", "GOAT-EXPLORE", "GOAT-DIVERSITY"]


@dataclass
class GOATConformerGeneration(ORCACalculator, ConformerGeneration):
    """Generate conformers using GOAT."""

    name: str = "GOAT Conformer Generation"
    goat: GoatKeywordsType = field(
        default="GOAT", metadata={"description": "The GOAT keyword to use for the calculation"}
    )
    _basename: str = "goat_conformer_generation"

    def operation(self, molecule: Molecule) -> tuple[list[Molecule], ORCAProperties]:
        """Generate conformers using GOAT."""
        # Write to XYZ file
        molecule.to("input.xyz", fmt="xyz")
        # Get the default calculator SK_list
        sk_list = super().set_keywords()
        # Add the GOAT keywords
        sk_list.append(getattr(Goat, self.goat.upper()))  # type: ignore
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
        conformers = XYZ.from_file(output.get_file(".xyz")).all_molecules
        return conformers, properties
