"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, cast

import ase.optimize
from ase import filters
from ase.filters import Filter
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.core.makers import SingleJFChemistryMaker
from jfchemistry.core.properties import Properties
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class ASEOptimizer[InputType: Structure | Molecule, OutputType: Structure | Molecule](
    SingleJFChemistryMaker[InputType, OutputType], GeometryOptimization
):
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
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.optimizers import ASEOptimizer # doctest: +SKIP
        >>> from jfchemistry.calculators import TBLiteCalculator # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> # Create custom optimizer by inheriting
        >>> class MyOptimizer(ASEOptimizer, TBLiteCalculator): # doctest: +SKIP
        ...     pass # doctest: +SKIP
        >>> opt = MyOptimizer(optimizer="LBFGS", fmax=0.01) # doctest: +SKIP
        >>> job = opt.make(molecule) # doctest: +SKIP
    """

    name: str = "ASE Optimizer"
    calculator: ASECalculator = field(
        default_factory=lambda: ASECalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )
    optimizer: Literal["LBFGS", "BFGS", "GPMin", "MDMin", "FIRE", "FIRE2", "QuasiNewton"] = field(
        default="LBFGS",
        metadata={"description": "the ASE optimizer to use for the calculation"},
    )
    unit_cell_optimizer: Optional[
        Literal["UnitCellFilter", "ExpCellFilter", "FrechetCellFilter"]
    ] = field(
        default=None,
        metadata={"description": "the ASE unit cell optimizer to use for the calculation"},
    )
    fmax: float = field(
        default=0.05,
        metadata={"description": "the maximum force convergence criterion in eV/Angstrom"},
    )
    steps: int = field(
        default=250000,
        metadata={"description": "the maximum number of optimization steps"},
    )
    trajectory: Optional[str] = field(
        default=None,
        metadata={"description": "the trajectory file to save the optimization"},
    )
    logfile: Optional[str] = field(
        default=None,
        metadata={"description": "the log file to save the optimization"},
    )

    def __post_init__(self):
        """Post-initialization hook."""
        self.name = f"{self.name} with {self.calculator.name}"
        super().__post_init__()

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Optimize molecular structure using ASE.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the specified ASE optimizer
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.
            **kwargs: Additional kwargs to pass to the operation.

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
        if type(structure) is Molecule:
            spin_multiplicity = int(structure.spin_multiplicity)
        else:
            spin_multiplicity = 1
        atoms = self.calculator._set_calculator(
            atoms, charge=charge, spin_multiplicity=spin_multiplicity
        )

        if type(structure) is Structure and self.unit_cell_optimizer is not None:
            opt_atoms = getattr(filters, self.unit_cell_optimizer)(atoms)
        else:
            opt_atoms = atoms

        opt_func = getattr(ase.optimize, self.optimizer)
        opt = opt_func(opt_atoms, logfile=self.logfile, trajectory=self.trajectory)
        opt.run(self.fmax, self.steps)
        if type(structure) is Structure:
            if self.unit_cell_optimizer is not None and isinstance(opt_atoms, Filter):
                opt_atoms = opt_atoms.atoms

        properties = self.calculator._get_properties(opt_atoms)
        from ase import io

        io.write("opt.cif", opt_atoms)
        if isinstance(structure, Structure):
            opt_structure = Structure.from_ase_atoms(opt_atoms)
        elif isinstance(structure, Molecule):
            opt_structure = Molecule.from_ase_atoms(opt_atoms)
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

        return cast("OutputType", opt_structure), properties
