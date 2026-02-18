"""NVT molecular dynamics using Langevin dynamics.

This module provides NVT molecular dynamics simulation using ASE's
Langevin dynamics integrator.
"""

from dataclasses import dataclass, field
from typing import Literal

from ase import Atoms
from ase.md import Langevin
from ase.units import fs

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNVTLangevin(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NVT ensemble with Langevin dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NVT Langevin").
        integrator: The integrator type (fixed to "nvt_langevin").
        friction: Friction coefficient [fs^-1] (default: 0.01).
        fixcm: Whether to fix the center of mass (default: True).
    """

    name: str = "ASE Molecular Dynamics NVT Langevin"
    integrator: Literal["nvt_langevin"] = "nvt_langevin"
    friction: float = field(
        default=0.01,
        metadata={"description": "Friction coefficient [fs^-1]", "unit": "fs^-1"},
    )
    fixcm: bool = field(
        default=True,
        metadata={"description": "Whether to fix the center of mass"},
    )

    def _create_dynamics(self, atoms: Atoms) -> Langevin:
        """Create the Langevin dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE Langevin dynamics object.
        """
        return Langevin(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            friction=self.friction / fs,  # Convert from fs^-1 to s^-1
            fixcm=self.fixcm,
        )
