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
class TorchSimMolecularDynamicsNPTNoseHoover(TorchSimMolecularDynamics):
    """Run a molecular dynamics simulation using TorchSim in NVE ensemble.

    Inherits all attributes from TorchSimMolecularDynamics.

    Attributes:
        name: Name of the calculator (default: "FairChem TorchSim Single Point Calculator").
        Additional attributes inherited from FairChemTSCalculator and TorchSimSinglePointCalculator.

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.optimizers import AimNet2Optimizer # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>>
        >>> # Fast optimization for screening
        >>> opt_fast = AimNet2Optimizer( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.1,  # Looser convergence # doctest: +SKIP
        ...     steps=500 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_fast.make(molecule) # doctest: +SKIP
        >>>
        >>> # Tight optimization
        >>> opt_tight = AimNet2Optimizer( # doctest: +SKIP
        ...     optimizer="LBFGS", # doctest: +SKIP
        ...     fmax=0.01, # doctest: +SKIP
        ...     charge=-1, # doctest: +SKIP
        ...     multiplicity=1 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = opt_tight.make(molecule) # doctest: +SKIP
        >>> optimized = job.output["structure"] # doctest: +SKIP
        >>> energy = job.output["properties"]["Global"]["Total Energy [eV]"] # doctest: +SKIP
    """

    name: str = "TorchSim Molecular Dynamics NPT"
    integrator: Literal["npt_nose_hoover"] = "npt_nose_hoover"
    b_tau: Optional[float] = field(
        default=None,
        metadata={
            "description": "Barostat time constant controlling how quickly the system\
                 responds to pressure differences. Defaults to 1000*timestep"
        },
    )
    t_tau: Optional[float] = field(
        default=None,
        metadata={
            "description": "Thermostat time constant controlling how quickly the system\
                 responds to temperature differences. Defaults to 100*timestep"
        },
    )
    external_pressure: float = field(
        default=1.0,
        metadata={"description": "External pressure applied to the system [atm] (default: 1 atm)"},
    )
    chain_length: int = field(default=3, metadata={"description": "Number of Nose-Hoover chains"})
    chain_steps: int = field(default=3, metadata={"description": "Number of steps per chain"})
    sy_steps: Literal[1, 3, 5, 7] = field(
        default=3, metadata={"description": "Number of Suzuki-Yoshida steps"}
    )

    def setup_dicts(self, model: ModelInterface):
        """Post initialization hook."""
        device = model.device
        dtype = model.dtype
        self.init_kwargs["chain_length"] = self.chain_length
        self.init_kwargs["chain_steps"] = self.chain_steps
        self.init_kwargs["sy_steps"] = self.sy_steps
        if self.b_tau is None:
            self.b_tau = 1 / (1000 * self.timestep / 1000)
        if self.t_tau is None:
            self.t_tau = 1 / (100 * self.timestep / 1000)
        self.init_kwargs["b_tau"] = torch.tensor(self.b_tau, device=device, dtype=dtype)
        self.init_kwargs["t_tau"] = torch.tensor(self.t_tau, device=device, dtype=dtype)
        self.step_kwargs["external_pressure"] = torch.tensor(
            self.external_pressure * Uc.atm_to_pa / Uc.bar_to_pa,
            device=model.device,
            dtype=model.dtype,
        )
