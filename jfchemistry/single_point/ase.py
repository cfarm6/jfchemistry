"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field
from typing import cast

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.core.makers.single_maker import SingleJFChemistryMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointCalculation


@dataclass
class ASESinglePoint[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    SingleJFChemistryMaker[InputType, OutputType],
    SinglePointCalculation,
):
    """Base class for single point energy calculations using ASE calculators.

    Combines single point energy calculations with ASE calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various ASE calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").
        calculator: The calculator to use for the calculation.
    """

    name: str = "ASE Single Point Calculator"
    calculator: ASECalculator = field(
        default_factory=lambda: ASECalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )

    def __post_init__(self):
        """Post initialization setup."""
        self.name = f"{self.name} with {self.calculator.name}"
        super().__post_init__()

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
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator
        """
        atoms = structure.to_ase_atoms()
        charge = int(structure.charge)
        if isinstance(structure, Molecule):
            spin_multiplicity = int(structure.spin_multiplicity)
        else:
            spin_multiplicity = 1
        self.calculator._set_calculator(atoms, charge=charge, spin_multiplicity=spin_multiplicity)
        print(atoms.get_potential_energy())
        properties = self.calculator._get_properties(atoms)
        return cast("OutputType", structure), properties
