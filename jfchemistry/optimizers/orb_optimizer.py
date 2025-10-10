"""Geometry Optimization using AimNet2."""

from dataclasses import dataclass

from jfchemistry.calculators.orb_calculator import ORBModelCalculator
from jfchemistry.optimizers.ase_optimizer import ASEOptimizer


@dataclass
class ORBModelOptimizer(ORBModelCalculator, ASEOptimizer):
    """Geometry Optimization using ORB Model."""

    name: str = "ORB Model Optimizer"
