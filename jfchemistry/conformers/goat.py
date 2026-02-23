"""GOAT Conformer Generation."""

from dataclasses import dataclass, field
from typing import Literal

from opi.input.simple_keywords.goat import Goat
from opi.input.structures.structure import Structure
from pymatgen.core.structure import Molecule
from pymatgen.io.xyz import XYZ

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator, ORCAProperties

from .base import ConformerGeneration

type GoatKeywordsType = Literal["GOAT", "GOAT-EXPLORE", "GOAT-DIVERSITY"]


@dataclass
class GOATConformers(ORCACalculator, ConformerGeneration):
    """Generate conformers using GOAT."""

    name: str = "GOAT Conformer Generation"
    goat: GoatKeywordsType = field(
        default="GOAT", metadata={"description": "The GOAT keyword to use for the calculation"}
    )
    _basename: str = "goat_conformer_generation"

    def _operation(self, molecule: Molecule) -> tuple[list[Molecule], ORCAProperties]:
        """Generate conformers using GOAT."""
        molecule.to("input.xyz", fmt="xyz")
        sk_list = super()._set_keywords()
        sk_list.append(getattr(Goat, self.goat.upper()))
        calc = super()._build_calculator(self._basename)
        calc.structure = Structure.from_xyz("input.xyz")
        super()._set_structure_charge_and_spin(
            calc, int(molecule.charge), int(molecule.spin_multiplicity)
        )
        super()._configure_calculator_input(calc, sk_list)
        calc.write_input()
        calc.run()
        output = calc.get_output()
        properties = super()._parse_output(output)
        conformers = XYZ.from_file(output.get_file(".xyz")).all_molecules
        return conformers, properties
