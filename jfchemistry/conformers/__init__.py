"""Conformer generation methods.

This module provides tools for generating multiple conformations of molecular
structures to explore conformational space and identify low-energy structures.

Available Methods:
    - ConformerGeneration: Base class for conformer generation
    - CRESTConformers: CREST-based conformer search using metadynamics

Examples:
    >>> from jfchemistry.conformers import CRESTConformers # doctest: +SKIP
    >>> from pymatgen.core import Molecule # doctest: +SKIP
    >>>
    >>> # Generate conformers using CREST
    >>> conformer_gen = CRESTConformers( # doctest: +SKIP
    ...     runtype="imtd-gc", # doctest: +SKIP
    ...     ewin=6.0,  # Energy window in kcal/mol # doctest: +SKIP
    ...     calculation_energy_method="gfnff", # doctest: +SKIP
    ...     calculation_dynamics_method="gfnff" # doctest: +SKIP
    ... ) # doctest: +SKIP
    >>> # Generate conformers
    >>> job = conformer_gen.make(molecule) # doctest: +SKIP
    >>> conformers = job.output["structure"]  # doctest: +SKIP

"""

from .crest import CRESTConformers

__all__ = ["CRESTConformers"]
