"""TorchSim-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch_sim as ts
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.makers.single_maker import SingleJFChemistryMaker

# from jfchemistry.core.makers.single_structure_molecule import SingleStructureMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class TorchSimOptimizer[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    SingleJFChemistryMaker[InputType, OutputType], GeometryOptimization
):
    """Base class for single point energy calculations using TorchSim calculators.

    Combines single point energy calculations with TorchSim calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various TorchSim calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").

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

    name: str = "TorchSim Optimizer"
    calculator: TorchSimCalculator = field(
        default_factory=lambda: TorchSimCalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )
    optimizer: Literal["FIRE", "Gradient Descent"] = field(
        default="FIRE", metadata={"description": "The optimizer to use"}
    )
    autobatcher: bool = field(
        default=True, metadata={"description": "Whether to use the autobatcher"}
    )
    max_steps: int = field(
        default=10_000, metadata={"description": "The maximum number of steps to take"}
    )
    steps_between_swaps: int = field(
        default=5,
        metadata={
            "description": "Number of steps to take before checking convergence\
                 and swapping out states."
        },
    )

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Optimize molecular structure using ASE.

        Performs geometry optimization by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
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
        optimizer = getattr(ts.Optimizer, self.optimizer.lower().replace(" ", "_"))
        model = self.calculator._get_model()
        structure.to_ase_atoms().write("initial_structure.xyz")
        final_state = ts.optimize(
            system=structure.to_ase_atoms(),
            model=model,
            optimizer=optimizer,
        )
        final_atoms = final_state.to_atoms()[0]
        final_structure = type(structure).from_ase_atoms(final_atoms)
        properties = self.calculator._get_properties(final_structure)
        final_structure.to_ase_atoms().write("final_structure.xyz")
        return final_structure, properties
