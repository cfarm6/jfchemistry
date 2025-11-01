"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ase.optimize
from ase import Atoms
from pymatgen.core.structure import SiteCollection

from jfchemistry.calculators.ase_calculator import ASECalculator
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ASEOptimizer(GeometryOptimization, ASECalculator):
    """Base class for geometry optimization using ASE optimizers.

    Combines geometry optimization workflows with ASE calculator interfaces.
    This class provides the framework for optimizing molecular structures
    using various ASE optimization algorithms (LBFGS, BFGS, FIRE, etc.) and
    different calculators (neural networks, machine learning, semi-empirical).

    Subclasses should inherit from both a specific ASECalculator implementation
    and ASEOptimizer to create complete optimization workflows.

    Attributes:
        name: Name of the optimizer (default: "ASE Optimizer").
        optimizer: ASE optimization algorithm to use:
            - "LBFGS": Limited-memory BFGS (default, recommended)
            - "BFGS": Broyden-Fletcher-Goldfarb-Shanno
            - "GPMin": Conjugate gradient
            - "MDMin": Molecular dynamics minimization
            - "FIRE": Fast Inertial Relaxation Engine
            - "FIRE2": FIRE version 2
            - "QuasiNewton": Quasi-Newton method
        fmax: Maximum force convergence criterion in eV/Angstrom (default: 0.05).
        steps: Maximum number of optimization steps (default: 250000).

    Examples:
        >>> from ase.build import molecule
        >>> from pymatgen.core import Molecule
        >>> from jfchemistry.optimizers import ASEOptimizer
        >>> from jfchemistry.calculators import TBLiteCalculator
        >>> molecule = Molecule.from_ase_atoms(molecule("C2H6"))
        >>> # Create custom optimizer by inheriting
        >>> class MyOptimizer(ASEOptimizer, TBLiteCalculator):
        ...     pass
        >>>
        >>> opt = MyOptimizer(optimizer="LBFGS", fmax=0.01)
        >>> job = opt.make(molecule)
    """

    name: str = "ASE Optimizer"
    optimizer: Literal["LBFGS", "BFGS", "GPMin", "MDMin", "FIRE", "FIRE2", "QuasiNewton"] = field(
        default="LBFGS"
    )
    fmax: float = 0.05
    steps: int = 250000

    def get_properties(self, structure: Atoms):
        """Get the properties for an ASE Atoms object."""
        raise NotImplementedError

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Optimize molecular structure using ASE.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the specified ASE optimizer
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        Examples:
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from jfchemistry.optimizers import TBLiteOptimizer # doctest: +SKIP
            >>> ethane = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> opt = TBLiteOptimizer(optimizer="LBFGS", fmax=0.01) # doctest: +SKIP
            >>> structures, properties = opt.operation(ethane) # doctest: +SKIP
        """
        atoms = structure.to_ase_atoms()
        charge = int(structure.charge)
        spin_multiplicity = int(structure.spin_multiplicity)
        atoms = self.set_calculator(atoms, charge=charge, spin_multiplicity=spin_multiplicity)
        opt = getattr(ase.optimize, self.optimizer)(atoms, logfile=None)
        opt.run(self.fmax, self.steps)
        opt_structure = type(structure).from_ase_atoms(atoms)
        properties = self.get_properties(atoms)
        return opt_structure, properties
