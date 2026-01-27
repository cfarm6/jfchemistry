"""SMILES Input."""

from dataclasses import dataclass

from jobflow.core.job import Response, job
from rdkit.Chem import rdchem, rdmolfiles

from jfchemistry.core.outputs import Output
from jfchemistry.inputs.core import MoleculeInput


@dataclass
class Smiles(MoleculeInput):
    """Create molecules from SMILES strings.

    Parses SMILES (Simplified Molecular Input Line Entry System) strings
    and converts them to RDKit Mol objects. Supports standard SMILES
    notation including stereochemistry and aromaticity.

    Attributes:
        name: Name of the input method (default: "SMILES Input").
        remove_salts: Inherited from MoleculeInput.
        add_hydrogens: Inherited from MoleculeInput.
    """

    name: str = "SMILES Input"

    def _get_structure(self, input: int | str) -> rdchem.Mol:
        """Parse SMILES string and convert to RDKit molecule.

        Args:
            input: SMILES string representation of the molecule.

        Returns:
            RDKit Mol object parsed from SMILES.

        Examples:
            >>> smiles = Smiles()
            >>> mol = smiles._get_structure("CCO")
            >>> mol.GetNumAtoms()  # Without hydrogens
            3
        """
        mol = rdmolfiles.MolFromSmiles(input)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {input}")
        return mol

    @job(files="files", properties="properties")
    def make(self, input: str) -> Response[Output]:
        """Create a workflow job to generate a molecule from SMILES.

        Args:
            input: SMILES string.

        Returns:
            Response containing:
                - structure: RDMolMolecule from SMILES
                - files: MOL file representation
        """
        resp = super()._make(input)
        return resp
