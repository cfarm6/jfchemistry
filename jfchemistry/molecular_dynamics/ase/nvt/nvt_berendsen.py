"""NVT molecular dynamics using Berendsen dynamics.

This module provides NVT molecular dynamics simulation using ASE's
Berendsen dynamics integrator (simple but less rigorous thermostat).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.nvtberendsen import NVTBerendsen
from ase.units import fs

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNVTBerendsen(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NVT ensemble with Berendsen dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NVT Berendsen").
        integrator: The integrator type (fixed to "nvt_berendsen").
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
    """

    name: str = "ASE Molecular Dynamics NVT Berendsen"
    integrator: Literal["nvt_berendsen"] = "nvt_berendsen"
    ttime: Optional[float] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant [fs]. Defaults to 100*timestep if None.",
            "unit": "fs",
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> NVTBerendsen:
        """Create the Berendsen NVT dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE NPTBerendsen dynamics object configured for NVT.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        return NVTBerendsen(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            ttime=ttime * fs,
        )
