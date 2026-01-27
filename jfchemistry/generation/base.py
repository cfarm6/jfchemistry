"""Base class for 3D structure generation.

This module provides the abstract base class for implementing 3D structure
generation methods from molecular representations.
"""

from dataclasses import dataclass

from jfchemistry.core.makers.single_rdmolecule import SingleRDMoleculeMaker


@dataclass
class StructureGeneration(SingleRDMoleculeMaker):
    """Base class for generating 3D structures from RDKit molecules.

    This abstract class defines the interface for structure generation
    implementations that convert RDKit molecular representations (without
    3D coordinates) into Pymatgen structures with explicit 3D coordinates.

    Subclasses should implement the operation() method to perform the actual
    embedding/generation using specific algorithms (e.g., distance geometry,
    force fields, or machine learning models).

    Attributes:
        name: Descriptive name for the structure generation method.

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
