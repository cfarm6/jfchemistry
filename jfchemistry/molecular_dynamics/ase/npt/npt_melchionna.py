"""NPT molecular dynamics using Melchionna dynamics.

This module provides NPT molecular dynamics simulation using ASE's
NPT class which combines Nose-Hoover and Parrinello-Rahman dynamics.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
from ase import Atoms
from ase.md.melchionna import MelchionnaNPT

from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.molecular_dynamics.ase.base import ASEMolecularDynamics


@dataclass
class ASEMolecularDynamicsNPTMelchionna(ASEMolecularDynamics):
    """Run a molecular dynamics simulation using ASE in NPT ensemble with Melchionna dynamics.

    This uses the NPT class which combines Nose-Hoover thermostat and Parrinello-Rahman barostat.

    Inherits all attributes from ASEMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics NPT Melchionna").
        integrator: The integrator type (fixed to "npt_melchionna").
        external_pressure: External pressure [atm] (default: 1.0).
        ttime: Thermostat time constant [fs] (default: None, uses 100*timestep).
        pfactor: Barostat coupling constant (default: None, calculated from bulk modulus).
        mask: Control which cell dimensions can change. Can be a 3-tuple (int, int, int)
            for isotropic control or a 3x3 numpy array for anisotropic strain control
            (default: None, all dimensions).
    """

    name: str = "ASE Molecular Dynamics NPT Melchionna"
    integrator: Literal["npt_melchionna"] = "npt_melchionna"
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
    pfactor: Optional[float] = field(
        default=None,
        metadata={
            "description": "Barostat coupling constant. Should be ptime² x Bulk Modulus. "
            "Defaults to None (calculated automatically)."
        },
    )
    mask: Optional[Union[tuple[int, int, int], np.ndarray]] = field(
        default=None,
        metadata={
            "description": "Control which cell dimensions can change. "
            "Can be a 3-tuple (int, int, int) for isotropic control or a 3x3 numpy array "
            "for anisotropic strain control. None means all dimensions can change."
        },
    )

    def _create_dynamics(self, atoms: Atoms) -> MelchionnaNPT:
        """Create the NPT dynamics object for the simulation.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE NPT dynamics object.
        """
        if self.ttime is None:
            ttime = 100.0 * self.timestep
        else:
            ttime = self.ttime

        # Convert pressure from atm to bar (ASE uses bar)
        external_pressure_bar = self.external_pressure * 1.01325

        # Convert to stress tensor (positive in tension for ASE)
        external_stress = -external_pressure_bar

        # Handle mask: if tuple, convert to array; if already array, use as-is
        if self.mask is not None:
            if isinstance(self.mask, np.ndarray):
                mask_array = self.mask
            else:
                mask_array = np.array(self.mask)
        else:
            mask_array = None

        return MelchionnaNPT(
            atoms,
            timestep=to_magnitude(self.timestep, "fs"),
            temperature_K=to_magnitude(self.temperature, "kelvin"),
            ttime=to_magnitude(ttime, "fs"),
            externalstress=external_stress,
            pfactor=self.pfactor,
            mask=mask_array,
        )
