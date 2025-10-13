"""Structure modification methods.

This module provides tools for modifying molecular structures including
protonation state changes, atom additions/removals, and other structural
transformations.

Available Methods:
    - StructureModification: Base class for structure modifications
    - CRESTDeprotonation: CREST-based deprotonation
    - CREST Protonation: CREST-based protonation

Examples:
    >>> from jfchemistry.modification import CRESTProtonation # doctest: +SKIP
    >>> from pymatgen.core import Molecule
    >>> from ase.build import molecule
    >>> ethane = Molecule.from_ase_atoms(molecule("C2H6"))
    >>> # Protonate a molecule
    >>> protonation = CRESTProtonation(ewin=6.0, threads=4)
    >>> job = protonation.make(ethane)
    >>> protonated_structures = job.output["structure"]
"""

from .base import StructureModification
from .crest_deprotonation import CRESTDeprotonation
from .crest_protonation import CRESTProtonation

__all__ = ["CRESTDeprotonation", "CRESTProtonation", "StructureModification"]
