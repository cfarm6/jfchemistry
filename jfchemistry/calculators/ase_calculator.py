"""Apply an ASE calculator to a structure."""

from ase import Atoms
from pydantic.dataclasses import dataclass


@dataclass
class ASECalculator:
    """Apply an ASE calculator to a structure."""

    name: str = "ASE Calculator"

    def set_calculator(self, atoms: Atoms, charge: int, spin_multiplicity: int) -> Atoms:
        """Set the calculator for the atoms."""
        raise NotImplementedError
