"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch_sim.models.interface import ModelInterface
from torch_sim.units import UnitConversion as Uc

from jfchemistry.molecular_dynamics.torchsim.base import TorchSimMolecularDynamics


@dataclass
class TorchSimMolecularDynamicsNPTLangevin(TorchSimMolecularDynamics):
    """Run a molecular dynamics simulation using TorchSim in NVE ensemble.

    Inherits all attributes from TorchSimMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "TorchSim Molecular Dynamics NPT Langevin").
        alpha: Atom friction coefficient controlling noise strength.
        cell_alpha: Cell friction coefficient controlling noise strength.
        b_tau: Barostat time constant controlling how quickly the system responds to pressure differences.
        external_pressure: External pressure applied to the system [atm] (default: 1 atm).

    """

    name: str = "TorchSim Molecular Dynamics NPT Langevin"
    integrator: Literal["npt_langevin"] = "npt_langevin"
    alpha: Optional[float] = field(
        default=None,
        metadata={"description": "Atom friction coefficient controlling noise strength"},
    )
    cell_alpha: Optional[float] = field(
        default=None,
        metadata={"description": "Cell friction coefficient controlling noise strength"},
    )
    b_tau: Optional[float] = field(
        default=None,
        metadata={
            "description": "Barostat time constant controlling how\
                 quickly the system responds to pressure differences"
        },
    )
    external_pressure: float = field(
        default=1.0,
        metadata={"description": "External pressure applied to the system [atm] (default: 1 atm)"},
    )

    def _setup_dicts(self, model: ModelInterface):
        """Post initialization hook."""
        # Init KWargs
        if self.alpha is None:
            self.alpha = 1.0 / (100 * self.timestep / 1000)  # ps^-1
        if self.cell_alpha is None:
            self.cell_alpha = self.alpha
        if self.b_tau is None:
            self.b_tau = 1 / (1000 * self.timestep / 1000)  # ps^-1

        alpha = torch.tensor(
            self.alpha,
            device=model.device,
            dtype=model.dtype,
        )

        cell_alpha = torch.tensor(
            self.cell_alpha,
            device=model.device,
            dtype=model.dtype,
        )
        b_tau = torch.tensor(self.b_tau, device=model.device, dtype=model.dtype)

        self.init_kwargs["alpha"] = alpha
        self.step_kwargs["alpha"] = alpha
        self.init_kwargs["cell_alpha"] = cell_alpha
        self.step_kwargs["cell_alpha"] = cell_alpha
        self.init_kwargs["b_tau"] = b_tau
        self.step_kwargs["b_tau"] = b_tau
        # Step KWargs
        self.step_kwargs["external_pressure"] = torch.tensor(
            self.external_pressure * Uc.atm_to_pa / Uc.bar_to_pa,
            device=model.device,
            dtype=model.dtype,
        )
