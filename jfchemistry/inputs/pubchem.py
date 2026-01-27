"""PubChem CID Input."""

from dataclasses import dataclass

from jobflow.core.job import Response, job
from rdkit.Chem import rdchem

from jfchemistry.core.outputs import Output
from jfchemistry.inputs.core import MoleculeInput


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

    def _get_structure(self, input: int | str) -> rdchem.Mol:
        """Fetch the structure from PubChem database.

        Downloads the molecular structure from PubChem using the compound ID
        and converts it to an RDKit Mol object.

        Args:
            input: PubChem compound ID (CID) as integer or string.

        Returns:
            RDKit Mol object from PubChem SDF data.
        """
        import requests
        from rdkit.Chem import rdmolfiles

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{input}/sdf"
        resp = requests.get(url)
        mol = rdmolfiles.MolFromMolBlock(resp.content.decode("utf-8"), removeHs=False)
        return mol

    @job(files="files", properties="properties")
    def make(self, input: int) -> Response[Output]:
        """Create a workflow job to retrieve a molecule from PubChem.

        Args:
            input: PubChem compound ID (CID).

        Returns:
            Response containing:
                - structure: RDMolMolecule from PubChem
                - files: MOL file representation

        """
        return super()._make(input)
