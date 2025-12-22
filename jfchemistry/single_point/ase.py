"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass, field

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.core.makers.single_structure_molecule import SingleStructureMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointEnergyCalculator


@dataclass
class ASESinglePoint(SinglePointEnergyCalculator, SingleStructureMoleculeMaker):
    """Base class for single point energy calculations using ASE calculators.

    Combines single point energy calculations with ASE calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various ASE calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").
    """

    name: str = "ASE Single Point Calculator"
    calculator: ASECalculator = field(
        default_factory=lambda: ASECalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )

    def operation(
        self, structure: Molecule | Structure
    ) -> tuple[Molecule | Structure | list[Molecule] | list[Structure], Properties]:
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
        """
        atoms = structure.to_ase_atoms()
        charge = int(structure.charge)
        if isinstance(structure, Molecule):
            spin_multiplicity = int(structure.spin_multiplicity)
        else:
            spin_multiplicity = 1
        atoms = self.calculator.set_calculator(
            atoms, charge=charge, spin_multiplicity=spin_multiplicity
        )
        properties = self.calculator.get_properties(atoms)
        return structure, properties
