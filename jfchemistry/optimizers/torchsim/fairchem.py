"""Geometry optimization using FairChem neural network potential.

This module provides fast geometry optimization using the FairChem neural
network potential combined with TorchSim optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.torchsim.fairchem_ts_calculator import FairChemTSCalculator
from jfchemistry.optimizers.torchsim.base import TorchSimOptimizer


@dataclass
class FairChemTorchSimOptimizer(FairChemTSCalculator, TorchSimOptimizer):
    """Calculate the single point energy of a structure using FairChem neural network potential.

    Inherits all attributes from FairChemTSCalculator and TorchSimSinglePointCalculator.

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

    name: str = "FairChem TorchSim Optimizer"
