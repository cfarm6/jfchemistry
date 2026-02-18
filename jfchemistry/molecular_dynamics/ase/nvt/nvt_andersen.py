"""NVT molecular dynamics using Andersen dynamics.

This module provides NVT molecular dynamics simulation using ASE's
Andersen dynamics integrator (stochastic collision-based thermostat).
"""

from dataclasses import dataclass, field
from typing import Literal

from ase import Atoms
from ase.md import Andersen
from ase.units import fs

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNVTAndersen(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NVT ensemble with Andersen dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NVT Andersen").
        integrator: The integrator type (fixed to "nvt_andersen").
        andersen_prob: Probability of collision per atom per timestep (default: 0.1).
    """

    name: str = "ASE Molecular Dynamics NVT Andersen"
    integrator: Literal["nvt_andersen"] = "nvt_andersen"
    andersen_prob: float = field(
        default=0.1,
        metadata={
            "description": "Probability of collision per atom per timestep",
            "unit": "dimensionless",
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> Andersen:
        """Create the Andersen dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE Andersen dynamics object.
        """
        return Andersen(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            andersen_prob=self.andersen_prob,
        )
