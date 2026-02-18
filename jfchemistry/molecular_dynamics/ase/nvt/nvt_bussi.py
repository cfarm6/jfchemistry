"""NVT molecular dynamics using Bussi dynamics.

This module provides NVT molecular dynamics simulation using ASE's
Bussi dynamics integrator (modern Nose-Hoover variant).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.bussi import Bussi
from ase.units import fs

from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNVTBussi(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NVT ensemble with Bussi dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NVT Bussi").
        integrator: The integrator type (fixed to "nvt_bussi").
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
    """

    name: str = "ASE Molecular Dynamics NVT Bussi"
    integrator: Literal["nvt_bussi"] = "nvt_bussi"
    ttime: Optional[float] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant [fs]. Defaults to 100*timestep if None.",
            "unit": "fs",
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> Bussi:
        """Create the Bussi dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE NVT dynamics object configured for NVT Bussi dynamics.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        return Bussi(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=self.temperature,
            taut=ttime * fs,
        )
