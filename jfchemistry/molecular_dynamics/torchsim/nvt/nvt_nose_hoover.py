"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
from pymatgen.core import SiteCollection
from torch_sim.models.interface import ModelInterface

from jfchemistry.molecular_dynamics.torchsim.base import TorchSimMolecularDynamics, TSMDProperties


@dataclass
class TorchSimMolecularDynamicsNVTNoseHoover(TorchSimMolecularDynamics):
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

    def setup_dicts(self, model: ModelInterface):
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

    def operation(
        self,
        structure: SiteCollection | list[SiteCollection],
        **kwargs: Any,
    ) -> tuple[
        SiteCollection | list[SiteCollection],
        TSMDProperties | list[TSMDProperties],
    ]:
        """Run an MD simulation with TorchSim in NVT ensemble with Nose-Hoover thermostat.

        Args:
            structure: Input molecular structure with 3D coordinates.
            calculator: TorchSimCalculator to use for the calculation.
            **kwargs: Additional kwargs to pass to the operation.
        """
        if isinstance(structure, list) and len(structure) > 1:
            raise TypeError("NVT Nose-Hoover does not support batching")
        return super().operation(structure)
