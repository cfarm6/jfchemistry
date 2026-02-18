"""NPT molecular dynamics using Berendsen dynamics.

This module provides NPT molecular dynamics simulation using ASE's
Berendsen thermostat and barostat.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from ase.md.nptberendsen import NPTBerendsen
from ase.units import fs
from pint import Quantity

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNPTBerendsen(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NPT ensemble with Berendsen dynamics.

    Inherits all attributes from ASEMolecularDynamics.

    Units:
        Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
        or ``jfchemistry.Q_``):

        - external_pressure: [atm]
        - ttime: [fs]
        - ptime: [fs]

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NPT Berendsen").
        integrator: The integrator type (fixed to "npt_berendsen").
        external_pressure: External pressure [atm] (default: 1.0). Accepts float or pint Quantity.
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep). \
            Accepts float or pint Quantity.
        ptime: Barostat time constant [fs] (default: None, uses 1000*timestep). \
            Accepts float or pint Quantity.
    """

    name: str = "ASE Molecular Dynamics NPT Berendsen"
    integrator: Literal["npt_berendsen"] = "npt_berendsen"
    external_pressure: float | Quantity = field(
        default=1.0,
        metadata={
            "description": "External pressure [atm]. Accepts float or pint Quantity.",
            "unit": "atm",
        },
    )
    compressibility_au: float = field(
        default=1.0,
        metadata={
            "description": "Compressibility [atomic units]",
            "unit": "atomic_units",
        },
    )
    ttime: Optional[float | Quantity] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant [fs]. Defaults to 100*timestep if None. \
                Accepts float or pint Quantity.",
            "unit": "fs",
        },
    )
    ptime: Optional[float | Quantity] = field(
        default=None,
        metadata={
            "description": "Barostat time constant [fs]. Defaults to 1000*timestep if None. \
                Accepts float or pint Quantity.",
            "unit": "fs",
        },
    )

    def __post_init__(self):
        """Normalize unit-bearing attributes."""
        if isinstance(self.external_pressure, Quantity):
            object.__setattr__(
                self, "external_pressure", to_magnitude(self.external_pressure, "atm")
            )
        if self.ttime is not None and isinstance(self.ttime, Quantity):
            object.__setattr__(self, "ttime", to_magnitude(self.ttime, "fs"))
        if self.ptime is not None and isinstance(self.ptime, Quantity):
            object.__setattr__(self, "ptime", to_magnitude(self.ptime, "fs"))
        super().__post_init__()

    def _create_dynamics(self, atoms: Atoms) -> NPTBerendsen:
        """Create the Berendsen NPT dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE NPTBerendsen dynamics object.
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

        return NPTBerendsen(
            atoms,
            timestep=self.timestep * fs,
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            taut=ttime * fs,
            compressibility_au=self.compressibility_au,
            pressure_au=external_pressure_au,
            taup=ptime * fs,
        )
