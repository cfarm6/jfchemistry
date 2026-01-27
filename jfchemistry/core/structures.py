"""Base Classes for the jfchemistry package."""

import base64
import binascii
import pickle
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, SerializeAsAny, field_validator
from rdkit.Chem import rdchem, rdmolfiles, rdmolops


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
    """

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        """Provide a JSON schema representation compatible with Pydantic."""
        return {
            "type": "string",
            "format": "rdkit-mol",
            "title": "RDMolMolecule",
            "description": (
                "Pickle serialized RDKit molecule encoded as base64 via `RDMolMolecule.as_dict`."
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        """Convert the molecule to a dictionary representation.

        Serializes the RDKit molecule using pickle and stores it in a dictionary
        format suitable for storage in MongoDB or other document databases.

        Returns:
            Dictionary containing the serialized molecule with module and class
            metadata for reconstruction.
        """
        data = pickle.dumps(self)
        encoded = base64.b64encode(data).decode("utf-8")

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": encoded,
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
        """
        data = d["data"]

        if isinstance(data, str):
            try:
                decoded = base64.b64decode(data.encode("utf-8"))
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Invalid base64 data for RDMolMolecule") from exc
        else:
            decoded = data

        molecule = pickle.loads(decoded)
        return molecule


class Polymer(BaseModel):
    """A Class for representing a polymer."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={RDMolMolecule: lambda mol: mol.as_dict()},
    )
    head: SerializeAsAny[Optional[RDMolMolecule]] = None
    monomer: SerializeAsAny[RDMolMolecule]
    tail: SerializeAsAny[Optional[RDMolMolecule]] = None

    @field_validator("monomer", "head", "tail", mode="after")
    @classmethod
    def standardize_molecule(cls, mol: Optional[RDMolMolecule]) -> Optional[RDMolMolecule]:
        if mol is None:
            return None

        # Standardize: remove Hs and round-trip through SMILES
        standardized_mol = rdmolfiles.MolFromSmiles(rdmolfiles.MolToSmiles(rdmolops.RemoveHs(mol)))

        # Wrap back in RDMolMolecule if needed
        return RDMolMolecule(standardized_mol)  # Adjust based on your RDMolMolecule constructor

    def as_dict(self) -> dict[str, Any]:
        """Convert the polymer to a dictionary representation."""
        return {
            "head": self.head.as_dict() if self.head is not None else None,
            "monomer": self.monomer.as_dict(),
            "tail": self.tail.as_dict() if self.tail is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Reconstruct a polymer from a dictionary representation."""
        return cls(
            head=RDMolMolecule.from_dict(d["head"]) if d["head"] is not None else None,
            monomer=RDMolMolecule.from_dict(d["monomer"]),
            tail=RDMolMolecule.from_dict(d["tail"]) if d["tail"] is not None else None,
        )
