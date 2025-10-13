"""Base class for 3D structure generation.

This module provides the abstract base class for implementing 3D structure
generation methods from molecular representations.
"""

from dataclasses import dataclass, field
from typing import Any

from pymatgen.core.structure import SiteCollection

from jfchemistry import RDMolMolecule, SingleMoleculeMaker


@dataclass
class StructureGeneration(SingleMoleculeMaker):
    """Base class for generating 3D structures from RDKit molecules.

    This abstract class defines the interface for structure generation
    implementations that convert RDKit molecular representations (without
    3D coordinates) into Pymatgen structures with explicit 3D coordinates.

    Subclasses should implement the operation() method to perform the actual
    embedding/generation using specific algorithms (e.g., distance geometry,
    force fields, or machine learning models).

    Attributes:
        name: Descriptive name for the structure generation method.
        check_structure: Whether to validate the generated structure using
            PoseBusters or similar validation tools (default: False).

    Examples:
        >>> # Subclass implementation
        >>> class MyGenerator(StructureGeneration): # doctest: +SKIP
        ...     def operation(self, structure): # doctest: +SKIP
        ...         # Embed 3D coordinates
        ...         mol_3d = embed_molecule(structure) # doctest: +SKIP
        ...         properties = {"method": "my_embedding"} # doctest: +SKIP
        ...         return mol_3d, properties # doctest: +SKIP
        >>>
        >>> gen = MyGenerator() # doctest: +SKIP
        >>> job = gen.make(rdmol) # doctest: +SKIP
        >>> structure = job.output["structure"] # doctest: +SKIP
    """

    # Input parameters
    name: str = "Structure Generation"
    # Check the structure with PoseBusters
    check_structure: bool = field(default=False)

    def operation(
        self, structure: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], dict[str, Any]]:
        """Generate 3D structure(s) from an RDKit molecule.

        This method must be implemented by subclasses to perform the actual
        3D coordinate generation using a specific algorithm.

        Args:
            structure: RDKit molecule without 3D coordinates.

        Returns:
            Tuple containing:
                - Generated 3D structure(s) as Pymatgen SiteCollection(s)
                - Dictionary of properties from the generation process

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> # In a subclass
            >>> def operation(self, structure):
            ...     # Generate 3D coordinates
            ...     structure_3d = generate_3d(structure)
            ...     props = {"energy": -123.45, "rmsd": 0.12}
            ...     return structure_3d, props
        """
        raise NotImplementedError
