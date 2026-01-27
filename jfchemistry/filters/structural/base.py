"""Base class for structural filters."""

from dataclasses import dataclass

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.makers import EnsembleMaker


@dataclass
class StructuralFilter[T: Structure | Molecule](EnsembleMaker[T]):
    """Base class for structural filters."""

    name: str = "Structural Filter"
