"""Molecule input from various chemical identifiers.

This module provides Maker classes for creating RDKit molecules from different
chemical identifier formats including SMILES and PubChem compound IDs.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from pydantic import BaseModel, Field, create_model
from rdkit.Chem import SaltRemover, rdchem, rdmolfiles, rdmolops

if TYPE_CHECKING:
    from pydantic.fields import _FieldInfoAsDict

from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import RDMolMolecule


@dataclass
class MoleculeInput(Maker):
    """Base class for molecule input from chemical identifiers.

    This abstract class provides common functionality for creating RDKit molecules
    from various chemical identifier formats. It handles salt removal and hydrogen
    addition automatically.

    Subclasses should implement the _get_structure() method to convert specific
    identifier types (SMILES, PubChem CID, etc.) into RDKit molecules.

    Attributes:
        name: Descriptive name for the input method.
        remove_salts: Whether to remove salt fragments from the molecule (default: True).
            Uses RDKit's SaltRemover to strip common counter-ions.
        add_hydrogens: Whether to add explicit hydrogen atoms to the molecule
            (default: True). Required for most 3D structure generation methods.
    """

    # Input parameters
    name: str = "Molecule Input"
    # Remove salts from the structure
    remove_salts: bool = field(
        default=True,
        metadata={"description": "Remove salts from the structure"},
    )
    # Add hydrogen atoms to the structure
    add_hydrogens: bool = field(
        default=True,
        metadata={"description": "Add hydrogen atoms to the structure"},
    )
    _output_model: type[Output] = Output
    _properties_model: type[Properties] = Properties

    def _make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict: "_FieldInfoAsDict" = f_info.asdict()
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = properties_model | list[properties_model]  # type: ignore

            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
                    *f_dict["metadata"],  # type: ignore
                    Field(**f_dict["attributes"]),
                ],  # type: ignore
                None,
            )

        self._output_model = create_model(
            f"{self._output_model.__name__}",
            __base__=self._output_model,
            **fields,
        )

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        self._make_output_model(self._properties_model)

    def _get_structure(self, input: int | str) -> RDMolMolecule:
        """Retrieve or parse structure from the input identifier.

        This method must be implemented by subclasses to convert specific
        identifier formats into RDKit molecules.

        Args:
            input: Chemical identifier (SMILES string, PubChem CID, etc.).

        Returns:
            RDKit Mol object.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _clean_structure(self, mol: rdchem.Mol) -> rdchem.Mol:
        """Clean and prepare the molecule structure.

        Performs post-processing on the molecule including salt removal and
        hydrogen addition based on the configured options.

        Args:
            mol: RDKit molecule to clean.

        Returns:
            Cleaned RDKit molecule.

        Raises:
            ValueError: If salt removal results in an empty molecule.

        Examples:
            >>> from rdkit import Chem
            >>> # Molecule with salt
            >>> mol = Chem.MolFromSmiles("CCO.Cl")
            >>> maker = MoleculeInput(remove_salts=True)
            >>> clean_mol = maker._clean_structure(mol)
            >>> Chem.MolToSmiles(clean_mol)  # Only ethanol remains
            '[H]OC([H])([H])C([H])([H])[H]'
        """
        if self.remove_salts:
            mol = SaltRemover.SaltRemover().StripMol(mol)  # type: ignore[no-untyped-call]
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("No molecule returned after removing salts.")
        if self.add_hydrogens:
            mol = rdmolops.AddHs(mol)
        return mol

    def _make(self, input: int | str) -> Response[_output_model]:
        """Create the internal workflow job.

        Internal method that handles the complete workflow: retrieve structure,
        clean it, and package it for output.

        Args:
            input: Chemical identifier.

        Returns:
            Response containing:
                - structure: RDMolMolecule object
                - files: MOL file representation
        """
        mol = self._get_structure(input)
        mol = self._clean_structure(mol)
        mol = RDMolMolecule(mol)
        resp = Response(
            output=self._output_model(
                structure=mol,
                files=rdmolfiles.MolToV3KMolBlock(mol),
            ),
        )
        return resp
