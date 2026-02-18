"""NPT molecular dynamics using Inhomogeneous Berendsen dynamics.

This module provides NPT molecular dynamics simulation using ASE's
InhomogeneousNPTBerendsen which allows independent cell scaling along
three directions while maintaining orthogonal angles.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.units import fs

from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNPTBerendsenInhomogeneous(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NPT ensemble with Inhomogeneous Berendsen.

    This allows independent scaling of basis vectors while maintaining orthogonal angles.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NPT Berendsen Inhomogeneous")
        integrator: The integrator type (fixed to "npt_berendsen_inhomogeneous").
        external_pressure: External pressure [atm] (default: 1.0).
            Can be a scalar or a 3-tuple for different pressures along each axis.
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
        ptime: Barostat time constant [fs] (default: None, uses 1000*timestep).
    """

    name: str = "ASE Molecular Dynamics NPT Berendsen Inhomogeneous"
    integrator: Literal["npt_berendsen_inhomogeneous"] = "npt_berendsen_inhomogeneous"
    external_pressure: float | tuple[float, float, float] = field(
        default=1.0,
        metadata={
            "description": "External pressure [atm]. Can be scalar or 3-tuple for each axis.",
            "unit": "atm",
        },
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

    def _create_dynamics(self, atoms: Atoms):
        """Create the Inhomogeneous Berendsen NPT dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE Inhomogeneous_NPTBerendsen dynamics object.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        if self.ptime is None:
            ptime = 1000.0 * self.timestep
        else:
            ptime = self.ptime

        if Inhomogeneous_NPTBerendsen is None:
            raise ImportError(
                "Inhomogeneous_NPTBerendsen is not available in this version of ASE. "
                "Please use ASEMolecularDynamicsNPTBerendsen instead."
            )

        # Convert pressure from atm to atomic units
        # 1 atm = 1.01325 bar = 1.01325e5 Pa
        # 1 a.u. pressure = 2.942e13 Pa
        if isinstance(self.external_pressure, tuple):
            pressure_au = tuple(p * 1.01325e5 / 2.942e13 for p in self.external_pressure)
        else:
            pressure_au = self.external_pressure * 1.01325e5 / 2.942e13

        return Inhomogeneous_NPTBerendsen(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=self.temperature,
            ttime=ttime * fs,
            pressure_au=pressure_au,
            ptime=ptime * fs,
        )
