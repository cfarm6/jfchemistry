"""Nodes for molecule input."""

from dataclasses import dataclass, field
from typing import Any

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from rdkit.Chem import SaltRemover, rdchem, rdmolfiles, rdmolops

from jfchemistry.jfchemistry import RDMolMolecule


@dataclass
class MoleculeInput(Maker):
    """Maker for molecule input. Molecule can be input as SMILES, SMARTS, or PubChem CID."""

    # Input parameters
    name: str = "Molecule Input"
    # Remove salts from the structure
    remove_salts: bool = field(default=True)
    # Add hydrogen atoms to the structure
    add_hydrogens: bool = field(default=True)

    def get_structure(self, input: int | str) -> rdchem.Mol:
        """Get the structure from the input."""
        raise NotImplementedError

    def clean_structure(self, input: rdchem.Mol) -> rdchem.Mol:
        """Clean the structure. Remove salts, Add hydrogen atoms, etc."""
        if self.remove_salts:
            input = SaltRemover.SaltRemover().StripMol(input)  # type: ignore[no-untyped-call]
            if input is None:
                raise ValueError("No molecule returned after removing salts.")
        if self.add_hydrogens:
            input = rdmolops.AddHs(input)

        return input

    def _make(self, input: int | str) -> Response[dict[str, Any]]:
        """Create the job."""
        mol = self.get_structure(input)
        mol = self.clean_structure(mol)
        return Response(
            output={
                "structure": RDMolMolecule(mol),
                "files": rdmolfiles.MolToV3KMolBlock(mol),
            },
        )


@dataclass
class PubChemCID(MoleculeInput):
    """Maker for retrieving a molecule from PubChem."""

    name: str = "PubChem CID Input"

    def get_structure(self, input: int | str) -> rdchem.Mol:
        """Fetch the structure from PubChem and convert to rdMol."""
        import requests
        from rdkit.Chem import rdmolfiles

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{input}/sdf"
        resp = requests.get(url)
        mol = rdmolfiles.MolFromMolBlock(resp.content.decode("utf-8"), removeHs=False)
        return mol

    @job(files="files", properties="properties")
    def make(self, input: int) -> Response[dict[str, Any]]:
        """Create the job."""
        return super()._make(input)


@dataclass
class Smiles(MoleculeInput):
    """Maker for retrieving a molecule from SMILES."""

    name: str = "SMILES Input"

    def get_structure(self, input: int | str) -> rdchem.Mol:
        """Fetch the structure from SMILES and convert to rdMol."""
        return rdmolfiles.MolFromSmiles(input)

    @job(files="files", properties="properties")
    def make(self, input: str) -> Response[dict[str, Any]]:
        """Create the job."""
        return super()._make(input)
