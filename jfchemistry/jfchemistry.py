"""The jfchemistry package."""

import pickle
from typing import Any

from rdkit.Chem.rdchem import Mol


class RDMolMolecule(Mol):
    """
    Represents a molecule in the RDKit format.

    Inherits from rdkit.Chem.rdchem.Mol.
    """

    def as_dict(self) -> dict[str, Any]:
        """Convert the molecule to a dictionary."""
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": pickle.dumps(super()),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Convert a dictionary to a molecule."""
        if type(d["data"]) is str:
            return pickle.loads(eval(d["data"]))
        else:
            return pickle.loads(d["data"])
