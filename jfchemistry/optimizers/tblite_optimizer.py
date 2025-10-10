"""Geometry Optimization using AimNet2."""

from dataclasses import dataclass

from jfchemistry.calculators.tblite_calculator import TBLiteCalculator
from jfchemistry.optimizers.ase_optimizer import ASEOptimizer


@dataclass
class TBLiteOptimizer(TBLiteCalculator, ASEOptimizer):
    """Geometry Optimization using AimNet2."""

    name: str = "TBLite Optimizer"
