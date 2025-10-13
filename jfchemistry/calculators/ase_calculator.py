"""Base class for ASE calculator integration.

This module provides the base interface for integrating ASE (Atomic Simulation
Environment) calculators into jfchemistry workflows.
"""

from ase import Atoms
from pydantic.dataclasses import dataclass


@dataclass
class ASECalculator:
    """Base class for ASE calculator integration.

    This class provides the interface for setting up ASE calculators on
    molecular structures. Subclasses implement specific calculators like
    AimNet2, ORB models, or TBLite.

    Attributes:
        name: Descriptive name for the calculator.

    Examples:
        >>> # Subclass implementation
        >>> from ase import Atoms # doctest: +SKIP
        >>>
        >>> class MyCalculator(ASECalculator): # doctest: +SKIP
        ...     def set_calculator(self, atoms, charge, spin_multiplicity): # doctest: +SKIP
        ...         from some_package import Calculator # doctest: +SKIP
        ...         atoms.calc = Calculator(charge=charge) # doctest: +SKIP
        ...         return atoms # doctest: +SKIP
        >>>
        >>> calc = MyCalculator() # doctest: +SKIP
        >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]) # doctest: +SKIP
        >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
    """

    name: str = "ASE Calculator"

    def set_calculator(self, atoms: Atoms, charge: int, spin_multiplicity: int) -> Atoms:
        """Set the calculator for the atoms.

        This method must be implemented by subclasses to attach a specific
        ASE calculator to the atoms object.

        Args:
            atoms: ASE Atoms object representing the molecular structure.
            charge: Total molecular charge.
            spin_multiplicity: Spin multiplicity (2S+1 where S is total spin).

        Returns:
            ASE Atoms object with the calculator attached.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> # In a subclass
            >>> def set_calculator(self, atoms, charge, spin_multiplicity):
            ...     from ase.calculators.emt import EMT
            ...     atoms.calc = EMT()
            ...     return atoms
        """
        raise NotImplementedError
