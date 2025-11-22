"""Base class for TorchSim calculator integration.

This module provides the base interface for integrating TorchSim (TorchSim) calculators into
 jfchemistry workflows.
Environment) calculators into jfchemistry workflows.
"""

from dataclasses import dataclass, field
from typing import Literal

from monty.json import MSONable
from pymatgen.core import Structure
from torch_sim.models.interface import ModelInterface

from jfchemistry.base_jobs import Properties
from jfchemistry.calculators.base import Calculator


@dataclass
class TorchSimCalculator(Calculator, MSONable):
    """Base class for ASE calculator integration.

    This class provides the interface for setting up TorchSim calculators on
    molecular structures. Subclasses implement specific calculators like
    Fairchem and Orb.

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

    name: str = "TorchSim Calculator"
    device: Literal["cpu", "cuda"] = field(
        default="cpu", metadata={"description": "The device to use for the calculator"}
    )

    def get_model(self) -> ModelInterface:
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

    def get_properties(self, system: Structure) -> Properties:
        """Get the properties for the atoms."""
        raise NotImplementedError
