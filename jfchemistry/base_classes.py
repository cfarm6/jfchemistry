"""Base Classes for the jfchemistry package."""

import pickle
from typing import Any, Optional

from pydantic import BaseModel
from rdkit.Chem import rdchem

type NestedFloatList = list[float] | list["NestedFloatList"] | float


class Property(BaseModel):
    """A calculated property."""

    name: str
    value: float | list[float]
    units: str
    uncertainty: Optional[float | list[float]] = None
    description: Optional[str] = None


class AtomicProperty(Property):
    """An atomic property."""

    value: NestedFloatList


class BondProperty(Property):
    """A bond property."""

    value: NestedFloatList
    atoms1: list[int]
    atoms2: list[int]


class OrbitalProperty(Property):
    """An orbital property."""

    value: NestedFloatList


class SystemProperty(Property):
    """A system property."""

    value: NestedFloatList


class RDMolMolecule(rdchem.Mol):
    """RDKit molecule wrapper with serialization support.

    This class extends RDKit's Mol class to provide serialization capabilities
    for use in jobflow workflows. It enables molecules to be stored in databases
    and passed between workflow jobs.

    The class uses pickle serialization to convert RDKit molecules to/from
    dictionary representations compatible with MongoDB and other document stores.

    Attributes:
        None

    Raises:
        None

    Examples:
        >>> from rdkit import Chem
        >>> from jfchemistry import RDMolMolecule
        >>>
        >>> # Create from SMILES
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> rdmol = RDMolMolecule(mol)
        >>>
        >>> # Serialize to dictionary
        >>> mol_dict = rdmol.as_dict()
        >>>
        >>> # Deserialize from dictionary
        >>> restored_mol = RDMolMolecule.from_dict(mol_dict)
    """

    def as_dict(self) -> dict[str, Any]:
        """Convert the molecule to a dictionary representation.

        Serializes the RDKit molecule using pickle and stores it in a dictionary
        format suitable for storage in MongoDB or other document databases.

        Returns:
            Dictionary containing the serialized molecule with module and class
            metadata for reconstruction.

        Examples:
            >>> from rdkit import Chem
            >>> from jfchemistry import RDMolMolecule
            >>> mol = RDMolMolecule(Chem.MolFromSmiles("CCO"))
            >>> mol_dict = mol.as_dict()
            >>> print(mol_dict.keys())
            dict_keys(['@module', '@class', 'data'])
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": pickle.dumps(super()),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Reconstruct a molecule from a dictionary representation.

        Deserializes an RDMolMolecule from a dictionary created by as_dict().
        Handles both string and bytes representations of pickled data.

        Args:
            d: Dictionary containing serialized molecule data with '@module',
                '@class', and 'data' keys.

        Returns:
            RDMolMolecule instance reconstructed from the dictionary.

        Examples:
            >>> from rdkit import Chem
            >>> from jfchemistry import RDMolMolecule
            >>> mol = RDMolMolecule(Chem.MolFromSmiles("CCO"))
            >>> mol_dict = mol.as_dict()
            >>> restored_mol = RDMolMolecule.from_dict(mol_dict)
            >>> Chem.MolToSmiles(restored_mol)
            'CCO'
        """
        if type(d["data"]) is str:
            return pickle.loads(eval(d["data"]))
        else:
            return pickle.loads(d["data"])
