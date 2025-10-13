"""Molecule input from various chemical identifiers.

This module provides Maker classes for creating RDKit molecules from different
chemical identifier formats including SMILES and PubChem compound IDs.
"""

from dataclasses import dataclass, field
from typing import Any

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from rdkit.Chem import SaltRemover, rdchem, rdmolfiles, rdmolops

from jfchemistry import RDMolMolecule


@dataclass
class MoleculeInput(Maker):
    """Base class for molecule input from chemical identifiers.

    This abstract class provides common functionality for creating RDKit molecules
    from various chemical identifier formats. It handles salt removal and hydrogen
    addition automatically.

    Subclasses should implement the get_structure() method to convert specific
    identifier types (SMILES, PubChem CID, etc.) into RDKit molecules.

    Attributes:
        name: Descriptive name for the input method.
        remove_salts: Whether to remove salt fragments from the molecule (default: True).
            Uses RDKit's SaltRemover to strip common counter-ions.
        add_hydrogens: Whether to add explicit hydrogen atoms to the molecule
            (default: True). Required for most 3D structure generation methods.

    Examples:
        >>> # Subclass implementation
        >>> from rdkit import Chem
        >>> from jfchemistry import RDMolMolecule
        >>> class MyInput(MoleculeInput):
        ...     def get_structure(self, input):
        ...         # Convert identifier to RDKit molecule
        ...         return Chem.MolFromSmiles(input)
        >>>
        >>> maker = MyInput(remove_salts=True, add_hydrogens=True)
        >>> structure = maker.get_structure("my_identifier")
    """

    # Input parameters
    name: str = "Molecule Input"
    # Remove salts from the structure
    remove_salts: bool = field(default=True)
    # Add hydrogen atoms to the structure
    add_hydrogens: bool = field(default=True)

    def get_structure(self, input: int | str) -> rdchem.Mol:
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

    def clean_structure(self, input: rdchem.Mol) -> rdchem.Mol:
        """Clean and prepare the molecule structure.

        Performs post-processing on the molecule including salt removal and
        hydrogen addition based on the configured options.

        Args:
            input: RDKit molecule to clean.

        Returns:
            Cleaned RDKit molecule.

        Raises:
            ValueError: If salt removal results in an empty molecule.

        Examples:
            >>> from rdkit import Chem
            >>> # Molecule with salt
            >>> mol = Chem.MolFromSmiles("CCO.Cl")
            >>> maker = MoleculeInput(remove_salts=True)
            >>> clean_mol = maker.clean_structure(mol)
            >>> Chem.MolToSmiles(clean_mol)  # Only ethanol remains
            '[H]OC([H])([H])C([H])([H])[H]'
        """
        if self.remove_salts:
            input = SaltRemover.SaltRemover().StripMol(input)  # type: ignore[no-untyped-call]
            if input is None:
                raise ValueError("No molecule returned after removing salts.")
        if self.add_hydrogens:
            input = rdmolops.AddHs(input)

        return input

    def _make(self, input: int | str) -> Response[dict[str, Any]]:
        """Create the internal workflow job.

        Internal method that handles the complete workflow: retrieve structure,
        clean it, and package it for output.

        Args:
            input: Chemical identifier.

        Returns:
            Response containing:
                - structure: RDMolMolecule object
                - files: MOL file representation

        Examples:
            >>> # Called internally by subclass make() methods
            >>> maker = Smiles()
            >>> response = maker._make("CCO")
        """
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
    """Retrieve molecules from PubChem database by compound ID.

    Downloads molecular structures from the PubChem database using the
    compound identifier (CID). The structure is retrieved in SDF format
    from PubChem's REST API.

    Attributes:
        name: Name of the input method (default: "PubChem CID Input").
        remove_salts: Inherited from MoleculeInput.
        add_hydrogens: Inherited from MoleculeInput.

    Examples:
        >>> from jfchemistry.inputs import PubChemCID
        >>>
        >>> # Retrieve ethanol (CID: 702)
        >>> pubchem = PubChemCID()
        >>> job = pubchem.make(702)
        >>> mol = job.output["structure"]
        >>>
        >>> # Retrieve without adding hydrogens
        >>> pubchem_no_h = PubChemCID(add_hydrogens=False)
        >>> job = pubchem_no_h.make(2244)  # Aspirin
    """

    name: str = "PubChem CID Input"

    def get_structure(self, input: int | str) -> rdchem.Mol:
        """Fetch the structure from PubChem database.

        Downloads the molecular structure from PubChem using the compound ID
        and converts it to an RDKit Mol object.

        Args:
            input: PubChem compound ID (CID) as integer or string.

        Returns:
            RDKit Mol object from PubChem SDF data.

        Examples:
            >>> from rdkit import Chem
            >>> from jfchemistry import RDMolMolecule
            >>> pubchem = PubChemCID()
            >>> mol = pubchem.get_structure(702)  # Ethanol
            >>> mol.GetNumAtoms()  # Without hydrogens initially
            9
        """
        import requests
        from rdkit.Chem import rdmolfiles

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{input}/sdf"
        resp = requests.get(url)
        mol = rdmolfiles.MolFromMolBlock(resp.content.decode("utf-8"), removeHs=False)
        return mol

    @job(files="files", properties="properties")
    def make(self, input: int) -> Response[dict[str, Any]]:
        """Create a workflow job to retrieve a molecule from PubChem.

        Args:
            input: PubChem compound ID (CID).

        Returns:
            Response containing:
                - structure: RDMolMolecule from PubChem
                - files: MOL file representation

        Examples:
            >>> from jfchemistry.inputs import PubChemCID
            >>> pubchem = PubChemCID()
            >>> job = pubchem.make(702)
            >>> mol = job.output["structure"]
            >>> mol_file = job.output["files"]
        """
        return super()._make(input)


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

    Examples:
        >>> from jfchemistry.inputs import Smiles
        >>>
        >>> # Simple molecule
        >>> smiles = Smiles()
        >>> job = smiles.make("CCO")  # Ethanol
        >>> mol = job.output["structure"]
        >>>
        >>> # Aromatic molecule with stereochemistry
        >>> job = smiles.make("c1ccc(cc1)C(=O)O")  # Benzoic acid
        >>>
        >>> # Salt that will be removed
        >>> smiles_clean = Smiles(remove_salts=True)
        >>> job = smiles_clean.make("CCO.Cl")  # Only ethanol kept
    """

    name: str = "SMILES Input"

    def get_structure(self, input: int | str) -> rdchem.Mol:
        """Parse SMILES string and convert to RDKit molecule.

        Args:
            input: SMILES string representation of the molecule.

        Returns:
            RDKit Mol object parsed from SMILES.

        Examples:
            >>> smiles = Smiles()
            >>> mol = smiles.get_structure("CCO")
            >>> mol.GetNumAtoms()  # Without hydrogens
            3
        """
        return rdmolfiles.MolFromSmiles(input)

    @job(files="files", properties="properties")
    def make(self, input: str) -> Response[dict[str, Any]]:
        """Create a workflow job to generate a molecule from SMILES.

        Args:
            input: SMILES string.

        Returns:
            Response containing:
                - structure: RDMolMolecule from SMILES
                - files: MOL file representation

        Examples:
            >>> from jfchemistry.inputs import Smiles
            >>> smiles = Smiles()
            >>> job = smiles.make("c1ccccc1")  # Benzene
            >>> mol = job.output["structure"]
            >>> mol_file = job.output["files"]
        """
        return super()._make(input)
