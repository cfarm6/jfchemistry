"""Geometry Optimization using AimNet2."""

from dataclasses import dataclass

from jfchemistry.calculators.aimnet2_calculator import AimNet2Calculator
from jfchemistry.optimizers.ase_optimizer import ASEOptimizer


@dataclass
class AimNet2Optimizer(AimNet2Calculator, ASEOptimizer):
    """Geometry Optimization using AimNet2."""

    name: str = "AimNet2 Optimizer"
