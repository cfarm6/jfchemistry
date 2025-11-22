"""ASE-based geometry optimization framework.

This module provides the base framework for geometry optimization using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

from dataclasses import dataclass
from typing import Any, Optional

from ase import Atoms
from pymatgen.core.structure import Molecule, SiteCollection

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.single_point.base import SinglePointEnergyCalculator


@dataclass
class ASESinglePointCalculator(SinglePointEnergyCalculator, ASECalculator):
    """Base class for single point energy calculations using ASE calculators.

    Combines single point energy calculations with ASE calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various ASE calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").
    """

    name: str = "ASE Single Point Calculator"

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
            spin_multiplicity = None
        atoms = self.set_calculator(atoms, charge=charge, spin_multiplicity=spin_multiplicity)
        properties = self.get_properties(atoms)
        return structure, properties
