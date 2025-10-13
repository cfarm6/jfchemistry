"""Geometry optimization using ORB machine learning force field.

This module provides geometry optimization using Orbital Materials' ORB
machine learning force field combined with ASE optimizers.
"""

from dataclasses import dataclass

from jfchemistry.calculators.orb_calculator import ORBModelCalculator
from jfchemistry.optimizers.ase import ASEOptimizer


@dataclass
class ORBModelOptimizer(ORBModelCalculator, ASEOptimizer):
    """Optimize molecular structures using ORB machine learning force field.

    Combines ORB's graph neural network force field with ASE optimization
    algorithms for accurate and efficient geometry optimization. Supports
    GPU acceleration and multiple precision options.

    Inherits all attributes from ORBModelCalculator (model, device, precision,
    compile) and ASEOptimizer (optimizer, fmax, steps).

    Attributes:
        name: Name of the optimizer (default: "ORB Model Optimizer").
        Additional attributes inherited from ORBModelCalculator and ASEOptimizer.

    Examples:
        >>> from jfchemistry.optimizers import ORBModelOptimizer
        >>> from ase.build import molecule
        >>> from pymatgen.core import Molecule
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH"))
        >>> # CPU optimization
        >>> opt_cpu = ORBModelOptimizer(
        ...     model="orb-v3-conservative-omol",
        ...     device="cpu",
        ...     optimizer="LBFGS",
        ...     fmax=0.05
        ... )
        >>> job = opt_cpu.make(molecule)
        >>>
        >>> # GPU optimization with compilation
        >>> opt_gpu = ORBModelOptimizer(
        ...     model="orb-v3-conservative-omol",
        ...     device="cuda",
        ...     compile=True,
        ...     precision="float32-highest",
        ...     optimizer="LBFGS",
        ...     fmax=0.01
        ... )
        >>> job = opt_gpu.make(molecule)
    """

    name: str = "ORB Model Optimizer"
