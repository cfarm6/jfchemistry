"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch_sim.models.interface import ModelInterface

from jfchemistry.molecular_dynamics.torchsim.base import TorchSimMolecularDynamics


@dataclass
class TorchSimMolecularDynamicsNVTNoseHoover(TorchSimMolecularDynamics):
    """Run a molecular dynamics simulation using TorchSim in NVE ensemble.

    Inherits all attributes from TorchSimMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "TorchSim Molecular Dynamics NVT Nose-Hoover").
        tau: Thermostat relaxation time, defaults to 100*timestep
        chain_length: Number of Nose-Hoover chains
        chain_steps: Number of steps per chain
        sy_steps: Number of Suzuki-Yoshida steps
    """

    name: str = "TorchSim Molecular Dynamics NVT Nose-Hoover"
    integrator: Literal["nvt_nose_hoover"] = "nvt_nose_hoover"
    tau: Optional[float] = field(
        default=None,
        metadata={"description": "Thermostat relaxation time, defaults to 100*timestep"},
    )
    chain_length: int = field(default=3, metadata={"description": "Number of Nose-Hoover chains"})
    chain_steps: int = field(default=3, metadata={"description": "Number of steps per chain"})
    sy_steps: Literal[1, 3, 5, 7] = field(
        default=3, metadata={"description": "Number of Suzuki-Yoshida steps"}
    )

    def _setup_dicts(self, model: ModelInterface):
        """Post initialization hook."""
        device = model.device
        dtype = model.dtype
        self.init_kwargs["tau"] = (
            torch.tensor(self.tau / 1000, device=device, dtype=dtype)
            if self.tau is not None
            else None
        )
        self.init_kwargs["chain_length"] = self.chain_length
        self.init_kwargs["chain_steps"] = self.chain_steps
        self.init_kwargs["sy_steps"] = self.sy_steps
