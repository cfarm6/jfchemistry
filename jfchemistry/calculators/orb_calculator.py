"""Apply the AimNet2 calculator to a structure."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ase import Atoms

from .ase_calculator import ASECalculator


@dataclass
class ORBModelCalculator(ASECalculator):
    """Apply the AimNet2 calculator to a structure."""

    name: str = "ORB Model Calculator"
    model: Literal["orb-v3-conservative-omol", "orb-v3-direct-omol"] = "orb-v3-conservative-omol"
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32-high", "float32-highest", "float64"] = "float32-high"
    compile: bool = False

    def set_calculator(self, atoms: Atoms, charge: int = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the calculator for the atoms."""
        try:
            from orb_models.forcefield import pretrained  # type: ignore
            from orb_models.forcefield.calculator import ORBCalculator  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'orb-models' package is required to use ORBCalculator but is not available."
                "Please install it from: https://github.com/orbital-materials/orb-models"
            ) from e
        if self.charge is not None:
            charge = self.charge
        if self.multiplicity is not None:
            spin_multiplicity = self.multiplicity

        orbff = getattr(pretrained, self.model.replace("-", "_"))(
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )

        atoms.calc = ORBCalculator(orbff, device=self.device)
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin_multiplicity
        return atoms

    def get_properties(self, atoms: Atoms) -> dict[str, Any]:
        """Return the properties of the structure."""
        energy = atoms.get_total_energy()  # type: ignore
        properties = {
            "Global": {"Total Energy [eV]": energy},
        }

        return properties
