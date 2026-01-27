"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass, field
from typing import Literal

from torch_sim.models.interface import ModelInterface

from jfchemistry.molecular_dynamics.torchsim.base import TorchSimMolecularDynamics


@dataclass
class TorchSimMolecularDynamicsNVTLangevin(TorchSimMolecularDynamics):
    """Run a molecular dynamics simulation using TorchSim in NVE ensemble.

    Inherits all attributes from TorchSimMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "TorchSim Molecular Dynamics NVT Langevin").
        gamma: Friction coefficient controlling noise strength.

    """

    name: str = "TorchSim Molecular Dynamics NVT Langevin"
    integrator: Literal["nvt_langevin"] = "nvt_langevin"
    gamma: float = field(
        default=1.0, metadata={"description": "Friction coefficient controlling noise strength"}
    )

    def _setup_dicts(self, model: ModelInterface):
        """Post initialization hook."""
        self.step_kwargs["gamma"] = self.gamma
