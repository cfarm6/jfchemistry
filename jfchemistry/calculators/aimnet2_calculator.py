"""Apply the AimNet2 calculator to a structure."""

from dataclasses import dataclass
from typing import Any, Optional

from ase import Atoms

from .ase_calculator import ASECalculator


@dataclass
class AimNet2Calculator(ASECalculator):
    """Apply the AimNet2 calculator to a structure."""

    name: str = "AimNet2 Calculator"
    model: str = "aimnet2"
    charge: Optional[int] = None
    multiplicity: Optional[int] = None

    def set_calculator(self, atoms: Atoms, charge: int = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the calculator for the atoms."""
        try:
            from aimnet.calculators import AIMNet2ASE  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'aimnet' package is required to use AimNet2Calculator but is not available. "
                "Please install it from: https://github.com/cfarm6/aimnetcentral.git"
            ) from e
        if self.charge is not None:
            charge = self.charge
        if self.multiplicity is not None:
            spin_multiplicity = self.multiplicity
        atoms.calc = AIMNet2ASE(self.model, charge, spin_multiplicity)

        aimnet2_atomtypes = [1, 6, 7, 8, 9, 17, 16, 5, 14, 15, 33, 34, 35, 53]
        atomic_nums = atoms.get_atomic_numbers()  # type: ignore
        if not all(atom in aimnet2_atomtypes for atom in atomic_nums):
            raise ValueError(
                f"Unsupport atomtype by AimNet2. Supported atom types are {aimnet2_atomtypes}"
            )

        return atoms

    def get_properties(self, atoms: Atoms) -> dict[str, Any]:
        """Return the properties of the structure."""
        energy = atoms.get_total_energy()  # type: ignore
        charge = atoms.get_charges()  # type: ignore
        properties = {
            "Global": {"Total Energy [eV]": energy},
            "Atomic": {"AimNet2 Partial Charges [e]": charge},
        }

        return properties
