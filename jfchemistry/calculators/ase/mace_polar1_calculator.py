"""MACE-Polar-1 calculator integration for ASE workflows.

This module integrates the MACE-Polar-1 foundation model via the ``mace-models``
package and exposes it through jfchemistry's ASE calculator interface.
"""

from dataclasses import dataclass, field
from typing import Literal

from ase import Atoms
from monty.json import MSONable

from jfchemistry import ureg
from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.calculators.base import MachineLearnedInteratomicPotentialCalculator
from jfchemistry.core.properties import AtomicProperty, Properties, PropertyClass, SystemProperty


class MACEPolar1AtomicProperties(PropertyClass):
    """Atomic properties from a MACE-Polar-1 calculation."""

    forces: AtomicProperty


class MACEPolar1SystemProperties(PropertyClass):
    """System-level properties from a MACE-Polar-1 calculation."""

    total_energy: SystemProperty


class MACEPolar1Properties(Properties):
    """Properties container for MACE-Polar-1 results."""

    atomic: MACEPolar1AtomicProperties
    system: MACEPolar1SystemProperties


@dataclass
class MACEPolar1Calculator(ASECalculator, MachineLearnedInteratomicPotentialCalculator, MSONable):
    """ASE calculator wrapper for the MACE-Polar-1 model.

    Uses ``mace_models.load`` to fetch the requested checkpoint and attaches the
    returned ASE calculator to the provided atoms object.

    Attributes:
        name: Human-readable calculator name.
        model: MACE model checkpoint name. Defaults to ``"MACE-Polar-1"``.
        dtype: Floating precision passed to the underlying model calculator.
    """

    name: str = "MACE-Polar-1 Calculator"
    model: str = field(default="MACE-Polar-1", metadata={"description": "MACE checkpoint name"})
    dtype: Literal["float32", "float64"] = field(
        default="float64",
        metadata={"description": "Floating precision for model inference"},
    )

    _properties_model: type[MACEPolar1Properties] = MACEPolar1Properties

    def _set_calculator(self, atoms: Atoms, charge: float = 0, spin_multiplicity: int = 1) -> Atoms:
        """Attach the MACE-Polar-1 ASE calculator to atoms.

        Args:
            atoms: ASE atoms object.
            charge: Included for interface compatibility; currently unused by
                MACE-Polar-1 model loading.
            spin_multiplicity: Included for interface compatibility; currently
                unused by MACE-Polar-1 model loading.

        Returns:
            ASE atoms object with ``atoms.calc`` set.

        Raises:
            ImportError: If ``mace-models`` is not installed.
        """
        del charge, spin_multiplicity
        try:
            import mace_models
        except ImportError as e:
            raise ImportError(
                "The 'mace-models' package is required for MACEPolar1Calculator. "
                "Install with: pip install mace-models"
            ) from e

        model = mace_models.load(self.model)
        atoms.calc = model.get_calculator(dtype=self.dtype)
        return atoms

    def _get_properties(self, atoms: Atoms) -> MACEPolar1Properties:
        """Extract total energy and forces from an atoms object.

        Args:
            atoms: ASE atoms object with MACE calculator attached.

        Returns:
            Structured MACE-Polar-1 properties (energy + forces).
        """
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        return MACEPolar1Properties(
            atomic=MACEPolar1AtomicProperties(
                forces=AtomicProperty(
                    name="MACE-Polar-1 Forces",
                    value=forces * ureg.eV / ureg.angstrom,
                    description=f"Forces predicted by {self.model}",
                )
            ),
            system=MACEPolar1SystemProperties(
                total_energy=SystemProperty(
                    name="Total Energy",
                    value=energy * ureg.eV,
                    description=f"Total energy predicted by {self.model}",
                )
            ),
        )
