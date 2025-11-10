"""Base class for structure modifications.

This module provides the abstract base class for implementing structure
modification methods in jfchemistry workflows.
"""

from dataclasses import dataclass
from typing import Any, Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry.modification.base import StructureModification


@dataclass
class TautomerMaker(StructureModification):
    """Base class for structure modification methods.

    This abstract class defines the interface for structure modification
    implementations. Subclasses should implement the operation() method
    to perform specific modifications such as protonation, deprotonation,
    atom substitutions, or other structural transformations.

    Structure modifications typically generate multiple output structures
    representing different possible modification sites or states.

    Attributes:
        name: Descriptive name for the modification method.

    Examples:
        >>> # Subclass implementation
        >>> from ase.build import molecule
        >>> from pymatgen.core import Molecule
        >>> ethane = Molecule.from_ase_atoms(molecule("C2H6"))
        >>> class MyModification(StructureModification):
        ...     def operation(self, structure):
        ...         # Perform modification
        ...         modified_structures = modify(structure)
        ...         properties = {"num_variants": len(modified_structures)}
        ...         return modified_structures, properties
        >>>
        >>> mod = MyModification()
        >>> job = mod.make(ethane)
        >>> modified = job.output["structure"]
    """

    name: str = "Structure Modification"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Modify the input structure.

        This method must be implemented by subclasses to perform the specific
        structural modification.

        Args:
            structure: Input molecular structure with 3D coordinates.

        Returns:
            Tuple containing:
                - Modified structure(s) (single or list of SiteCollections)
                - Dictionary of properties from modification (or None)

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> # In a subclass
            >>> def operation(self, structure):
            ...     modified = []
            ...     for site in get_modification_sites(structure):
            ...         mod_struct = apply_modification(structure, site)
            ...         modified.append(mod_struct)
            ...     return modified, {"num_sites": len(modified)}
        """
        raise NotImplementedError
