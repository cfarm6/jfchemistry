"""3D structure generation from molecular representations.

This module provides tools for generating 3D molecular structures from
representations without explicit coordinates (e.g., SMILES, molecular graphs).
"""

from .rdkit_generation import RDKitGeneration

__all__ = ["RDKitGeneration"]
