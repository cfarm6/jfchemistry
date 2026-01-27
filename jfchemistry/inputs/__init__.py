"""Input types for chemical structure representations.

This module provides Maker classes for creating RDKit molecules from various
chemical identifiers and representations.

Examples:
    >>> from jfchemistry.inputs import Smiles, PubChemCID
    >>>
    >>> smiles_maker = Smiles(add_hydrogens=True, remove_salts=True)
    >>> smiles_job = smiles_maker.make("CCO")
    >>> mol = smiles_job.output["structure"]
    >>>
    >>> # Retrieve molecule from PubChem
    >>> pubchem_maker = PubChemCID()
    >>> pubchem_job = pubchem_maker.make(702)  # Ethanol
    >>> mol = pubchem_job.output["structure"]
"""

from .polymer import PolymerInput
from .pubchem import PubChemCID
from .smiles import Smiles

__all__ = ["PolymerInput", "PubChemCID", "Smiles"]
