"""3D structure generation from molecular representations.

This module provides tools for generating 3D molecular structures from
representations without explicit coordinates (e.g., SMILES, molecular graphs).

Available Methods:
    - StructureGeneration: Base class for structure generation
    - RDKitGeneration: Generate 3D structures using RDKit's embedding methods
"""

from .rdkit_generation import RDKitGeneration

__all__ = ["RDKitGeneration"]
