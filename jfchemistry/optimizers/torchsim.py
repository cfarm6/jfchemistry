"""TorchSim-based geometry optimization framework.

This module provides the base framework for geometry optimization using
TorchSim calculators.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, cast

import torch_sim as ts
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.makers.pymatgen_maker import PymatGenMaker

# from jfchemistry.core.makers.single_structure_molecule import SingleStructureMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.optimizers.base import GeometryOptimization


@dataclass
class TorchSimOptimizer[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    PymatGenMaker[InputType, OutputType], GeometryOptimization
):
    """Base class for geometry optimization using TorchSim calculators.

    Combines geometry optimization with TorchSim calculator interfaces.
    This class provides the framework for optimizing structures
    using various TorchSim calculators (neural networks, machine learning,
    semi-empirical, etc.).

    Attributes:
        name: Name of the optimizer (default: "TorchSim Optimizer").

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.optimizers import TorchSimOptimizer # doctest: +SKIP
        >>> from jfchemistry.calculators.torchsim import OrbCalculator # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> # Create custom optimizer by inheriting
        >>> class MyOptimizer(TorchSimOptimizer, OrbCalculator): # doctest: +SKIP
        ...     pass # doctest: +SKIP
        >>> opt = MyOptimizer(optimizer="FIRE") # doctest: +SKIP
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
    steps: int = field(
        default=10_000,
        metadata={
            "description": "The maximum number of steps to take. Set to 0 for fixed geometry."
        },
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"description": "Deprecated alias for steps; if set, overrides steps."},
    )
    steps_between_swaps: int = field(
        default=5,
        metadata={
            "description": "Number of steps to take before checking convergence\
                 and swapping out states."
        },
    )

    def __post_init__(self):
        """Post-initialization hook."""
        if self.max_steps is not None:
            self.steps = self.max_steps
        self.max_steps = self.steps
        self.name = f"{self.name} with {self.calculator.name}"
        super().__post_init__()

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Optimize molecular structure using TorchSim.

        Performs geometry optimization by:
        1. Converting structure to TorchSim state
        2. Setting up the calculator with charge and spin
        3. Running the optimizer
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            input: Input molecular structure with 3D coordinates.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        Examples:
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from jfchemistry.optimizers import TorchSimOptimizer # doctest: +SKIP
            >>> ethane = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> opt = TorchSimOptimizer(optimizer="FIRE") # doctest: +SKIP
            >>> structures, properties = opt.operation(ethane) # doctest: +SKIP
        """
        optimizer = getattr(ts.Optimizer, self.optimizer.lower().replace(" ", "_"))
        model = self.calculator._get_model()
        input.to_ase_atoms().write("initial_structure.xyz")
        if self.steps == 0:
            final_structure = input
            properties = self.calculator._get_properties(final_structure)
        else:
            final_state = ts.optimize(
                system=input.to_ase_atoms(),
                model=model,
                optimizer=optimizer,
                max_steps=self.steps,
                steps_between_swaps=self.steps_between_swaps,
                autobatcher=self.autobatcher,
                pbar=True,
            )
            final_atoms = final_state.to_atoms()[0]
            final_structure = type(input).from_ase_atoms(final_atoms)
            properties = self.calculator._get_properties(final_structure)
        final_structure.to_ase_atoms().write("final_structure.xyz")
        return cast("OutputType", final_structure), properties
