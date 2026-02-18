"""NVT molecular dynamics using Nose-Hoover dynamics.

This module provides NVT molecular dynamics simulation using ASE's
Nose-Hoover dynamics integrator.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.units import fs

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNVTNoseHoover(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NVT ensemble with Nose-Hoover dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NVT Nose-Hoover").
        integrator: The integrator type (fixed to "nvt_nose_hoover").
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
    """

    name: str = "ASE Molecular Dynamics NVT Nose-Hoover"
    integrator: Literal["nvt_nose_hoover"] = "nvt_nose_hoover"
    ttime: Optional[float] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant [fs]. Defaults to 100*timestep if None.",
            "unit": "fs",
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> NoseHooverChainNVT:
        """Create the Nose-Hoover dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE NoseHoover dynamics object.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        return NoseHooverChainNVT(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            tdamp=ttime * fs,
        )
