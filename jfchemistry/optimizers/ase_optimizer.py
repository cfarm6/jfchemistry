"""Geometry Optimization using ASE."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ase.optimize
from pymatgen.core.structure import Molecule, SiteCollection

from jfchemistry.calculators.ase_calculator import ASECalculator
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ASEOptimizer(GeometryOptimization, ASECalculator):
    """Geometry Optimization using ASE."""

    name: str = "ASE Optimizer"
    optimizer: Literal["LBFGS", "BFGS", "GPMin", "MDMin", "FIRE", "FIRE2", "QuasiNewton"] = field(
        default="LBFGS"
    )
    fmax: float = 0.05
    steps: int = 250000

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Optimize the structure using ASE."""
        atoms = structure.to_ase_atoms()
        charge = int(structure.charge)
        spin_multiplicity = int(structure.spin_multiplicity)
        atoms = self.set_calculator(atoms, charge=charge, spin_multiplicity=spin_multiplicity)
        opt = getattr(ase.optimize, self.optimizer)(atoms, logfile=None)
        opt.run(self.fmax, self.steps)
        return Molecule.from_ase_atoms(atoms), self.get_properties(atoms)
