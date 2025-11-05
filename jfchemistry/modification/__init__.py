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
    >>> from pymatgen.core import Molecule # doctest: +SKIP
    >>> from ase.build import molecule # doctest: +SKIP
    >>> ethane = Molecule.from_ase_atoms(molecule("C2H6"))# doctest: +SKIP
    >>> # Protonate a molecule
    >>> protonation = CRESTProtonation(ewin=6.0, threads=4) # doctest: +SKIP
    >>> job = protonation.make(ethane) # doctest: +SKIP
    >>> protonated_structures = job.output["structure"] # doctest: +SKIP
"""

from .crest_deprotonation import CRESTDeprotonation
from .crest_protonation import CRESTProtonation
from .crest_tautomers import CRESTTautomers

__all__ = [
    "CRESTDeprotonation",
    "CRESTProtonation",
    "CRESTTautomers",
]
