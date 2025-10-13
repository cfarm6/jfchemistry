"""3D structure generation from molecular representations.

This module provides tools for generating 3D molecular structures from
representations without explicit coordinates (e.g., SMILES, molecular graphs).

Available Methods:
    - StructureGeneration: Base class for structure generation
    - RDKitGeneration: Generate 3D structures using RDKit's embedding methods

Examples:
    >>> from jfchemistry.inputs import Smiles
    >>> from jfchemistry.generation import RDKitGeneration
    >>>
    >>> # Get molecule from SMILES
    >>> smiles_job = Smiles().make("CCO")
    >>>
    >>> # Generate 3D structures
    >>> gen = RDKitGeneration(
    ...     method="ETKDGv3",
    ...     num_conformers=10,
    ...     prune_rms_thresh=0.5
    ... )
    >>> job = gen.make(smiles_job.output["structure"])
    >>> structures = job.output["structure"]
"""

from .base import StructureGeneration
from .rdkit_generation import RDKitGeneration

__all__ = ["RDKitGeneration", "StructureGeneration"]
