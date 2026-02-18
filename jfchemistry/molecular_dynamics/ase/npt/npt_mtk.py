"""NPT molecular dynamics using Berendsen dynamics.

This module provides NPT molecular dynamics simulation using ASE's
Berendsen thermostat and barostat.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.units import fs

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNPTBerendsen(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NPT ensemble with Berendsen dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NPT Berendsen").
        integrator: The integrator type (fixed to "npt_berendsen").
        external_pressure: External pressure [atm] (default: 1.0).
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
        ptime: Barostat time constant [fs] (default: None, uses 1000*timestep).
    """

    name: str = "ASE Molecular Dynamics NPT Berendsen"
    integrator: Literal["npt_berendsen"] = "npt_berendsen"
    external_pressure: float = field(
        default=1.0,
        metadata={"description": "External pressure [atm]", "unit": "atm"},
    )
    ttime: Optional[float] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant [fs]. Defaults to 100*timestep if None.",
            "unit": "fs",
        },
    )
    ptime: Optional[float] = field(
        default=None,
        metadata={
            "description": "Barostat time constant [fs]. Defaults to 1000*timestep if None.",
            "unit": "fs",
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> IsotropicMTKNPT:
        """Create the IsotropicMTKNPT NPT dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE IsotropicMTKNPT dynamics object.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        if self.ptime is None:
            ptime = 1000.0 * self.timestep
        else:
            ptime = self.ptime

        # Convert pressure from atm to atomic units (ASE NPTBerendsen uses a.u.)
        # 1 atm = 1.01325 bar = 1.01325e5 Pa
        # 1 a.u. pressure = 2.942e13 Pa
        external_pressure_au = self.external_pressure * 1.01325e5 / 2.942e13

        return IsotropicMTKNPT(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            tdamp=ttime * fs,
            pdamp=ptime * fs,
            pressure_au=external_pressure_au,
        )
