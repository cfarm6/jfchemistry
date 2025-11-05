"""Base class for conformer generation.

This module provides the abstract base class for implementing conformer
generation methods in jfchemistry workflows.
"""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel
from pymatgen.core.structure import SiteCollection

from jfchemistry import SingleStructureMaker


@dataclass
class ConformerGeneration(SingleStructureMaker):
    """Base class for conformer generation methods.

    This abstract class defines the interface for conformer generation
    implementations. Subclasses should implement the operation() method
    to generate multiple conformations of the input structure.

    Conformer generation explores the potential energy surface to find
    multiple low-energy conformations of a molecule, which is essential
    for accurately predicting molecular properties that depend on
    conformational flexibility.

    Attributes:
        name: Descriptive name for the conformer generation method.

    Examples:
        >>> # Subclass implementation
        >>> class MyConformerGenerator(ConformerGeneration): # doctest: +SKIP
        ...     def operation(self, structure): # doctest: +SKIP
        ...         # Generate conformers using some method
        ...         conformers = generate_conformers(structure) # doctest: +SKIP
        ...         properties = {"num_conformers": len(conformers)} # doctest: +SKIP
        ...         return conformers, properties # doctest: +SKIP
        >>>
        >>> gen = MyConformerGenerator() # doctest: +SKIP
        >>> job = gen.make(molecule) # doctest: +SKIP
        >>> conformers = job.output["structure"] # doctest: +SKIP
    """

    name: str = "Conformer Generation"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[BaseModel]]:
        """Generate conformers from the input structure.

        This method must be implemented by subclasses to perform the actual
        conformer generation using a specific algorithm or external tool.

        Args:
            structure: Input molecular structure with 3D coordinates.

        Returns:
            Tuple containing:
                - List of generated conformer structures (or single structure)
                - Dictionary of properties from conformer generation (or None)

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> # In a subclass
            >>> def operation(self, structure):
            ...     conformers = []
            ...     for i in range(10):
            ...         conf = perform_conformer_search(structure)
            ...         conformers.append(conf)
            ...     return conformers, {"method": "my_method"}
        """
        raise NotImplementedError
