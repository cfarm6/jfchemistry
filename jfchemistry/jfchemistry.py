"""The jfchemistry package core module.

This module provides the core data structures and base classes for the jfchemistry
package. It includes the RDMolMolecule wrapper class for RDKit molecules and base
Maker classes for building computational chemistry workflows.
"""

import pickle
from dataclasses import dataclass
from typing import Any, Optional, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from pymatgen.core.structure import SiteCollection
from rdkit.Chem import rdchem


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


def handle_conformers(maker: Maker, structure: RDMolMolecule) -> Response[dict[str, Any]]:
    """Distribute workflow jobs for each conformer in a molecule.

    Creates separate jobs for each conformer in an RDKit molecule, allowing
    parallel processing of multiple conformations. This is useful when a
    molecule has multiple embedded conformers that need to be processed
    independently (e.g., optimized separately).

    Args:
        maker: A Maker instance that will process each conformer.
        structure: RDMolMolecule containing one or more conformers.

    Returns:
        Response containing:
            - structures: List of processed structures from each job
            - files: List of output files from each job
            - properties: List of computed properties from each job
            - detour: List of jobs to be executed

    Examples:
        >>> from jfchemistry.generation import RDKitGeneration
        >>> from jfchemistry.optimizers import AimNet2Optimizer
        >>> from rdkit import Chem
        >>> rdmol = RDMolMolecule(Chem.MolFromSmiles("CCO"))
        >>> # Generate multiple conformers
        >>> gen = RDKitGeneration(num_conformers=10)
        >>> mol_job = gen.make(rdmol)
        >>>
        >>> # Optimize each conformer separately
        >>> opt = AimNet2Optimizer()
        >>> # handle_conformers is called internally when processing multi-conformer molecules
    """
    jobs: list[Response[dict[str, Any]]] = []
    for confId in range(structure.GetNumConformers()):
        s = RDMolMolecule(rdchem.Mol(structure, confId=confId))
        jobs.append(maker.make(s))  # type: ignore

    return Response(
        output={
            "structures": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


def handle_list_of_structures(
    maker: Maker, structures: list[RDMolMolecule]
) -> Response[dict[str, Any]]:
    """Distribute workflow jobs for a list of RDKit molecules.

    Processes a list of RDMolMolecule structures by creating individual jobs for each molecule.
    If any molecule contains multiple conformers, those conformers are handled
    separately using handle_conformers.

    Args:
        maker: A Maker instance that will process each structure.
        structures: List of RDMolMolecule structures to process.

    Returns:
        Response containing:
            - structures: List of processed structures from all jobs
            - files: List of output files from all jobs
            - properties: List of computed properties from all jobs
            - detour: List of jobs to be executed

    Examples:
        >>> from jfchemistry.optimizers import TBLiteOptimizer  # doctest: +SKIP
        >>> from jfchemistry.inputs import PubChemCID  # doctest: +SKIP
        >>> pubchem_cid = PubChemCID().make(21688863)  # doctest: +SKIP
        >>> opt = TBLiteOptimizer()  # doctest: +SKIP
        >>> structure = pubchem_cid.output["structure"] # doctest: +SKIP
        >>> structures = [structure, structure] # doctest: +SKIP
        >>> results = handle_list_of_structures(opt, structures)  # doctest: +SKIP
    """
    jobs: list[Response[dict[str, Any]]] = []
    for structure in structures:
        if structure.GetNumConformers() > 1:
            jobs.append(handle_conformers(maker, structure))
        else:
            jobs.append(maker.make(structure))  # type: ignore

    return Response(
        output={
            "structures": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


def handle_molecules(
    maker: Maker, structure: RDMolMolecule | list[RDMolMolecule]
) -> Response[dict[str, Any]] | None:
    """Route RDKit molecules to appropriate job distribution handler.

    Determines the appropriate distribution strategy based on whether the input
    is a single molecule, a list of molecules, or a molecule with multiple conformers.

    Args:
        maker: A Maker instance that will process the molecule(s).
        structure: Either a single RDMolMolecule or a list of RDMolMolecule structures.

    Returns:
        Response from the appropriate handler if distribution is needed, None if
        the structure should be processed directly by the maker.

    Examples:
        >>> from jfchemistry.generation import RDKitGeneration
        >>> from rdkit import Chem
        >>> rdmol = RDMolMolecule(Chem.MolFromSmiles("CCO"))
        >>> gen = RDKitGeneration(num_conformers=5)
        >>> # Returns Response with multiple conformers
        >>> result = handle_molecules(gen, rdmol)
    """
    if type(structure) is list:
        return handle_list_of_structures(maker, structure)
    elif cast("RDMolMolecule", structure).GetNumConformers() > 1:
        return handle_conformers(maker, cast("RDMolMolecule", structure))
    else:
        return None


def handle_structures(
    maker: Maker, structures: list[SiteCollection] | SiteCollection
) -> Response[dict[str, Any]] | None:
    """Distribute workflow jobs for Pymatgen structures.

    Creates individual jobs for each structure in a list. If a single structure
    is provided, returns None to indicate it should be processed directly.

    Args:
        maker: A Maker instance that will process each structure.
        structures: Either a list of SiteCollection (Molecule/Structure) objects
            or a single SiteCollection.

    Returns:
        Response containing distributed jobs if structures is a list, None if
        structures is a single SiteCollection to be processed directly.

    Examples:
        >>> from jfchemistry.optimizers import ORBModelOptimizer # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> mol1 = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> mol2 = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> mol3 = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> structures = [mol1, mol2, mol3]  # doctest: +SKIP
        >>> opt = ORBModelOptimizer() # doctest: +SKIP
        >>> # Processes each structure in parallel
        >>> result = handle_structures(opt, structures) # doctest: +SKIP
    """
    jobs: list[Response[dict[str, Any]]] = []
    if isinstance(structures, list):
        for structure in structures:
            jobs.append(maker.make(structure))  # type: ignore
    else:
        return None

    return Response(
        output={
            "structure": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


@dataclass
class SingleStructureMaker(Maker):
    """Base class for operations on structures with 3D geometry.

    This Maker processes Pymatgen SiteCollection objects (Molecule or Structure)
    that have assigned 3D coordinates. It handles automatic job distribution for
    lists of structures and provides a common interface for geometry optimization,
    conformer generation, and structure modifications.

    Subclasses should implement the operation() method to define specific
    computational tasks such as geometry optimization, property calculation,
    or structure modification.

    Attributes:
        name: Descriptive name for the job/operation being performed.

    Examples:
        >>> from jfchemistry.optimizers import TBLiteOptimizer # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>>
        >>> # Create an optimizer
        >>> optimizer = TBLiteOptimizer(method="GFN2-xTB", fmax=0.01) # doctest: +SKIP
        >>>
        >>> # Optimize a single structure
        >>> mol = Molecule.from_ase_atoms(molecule("H2O")) # doctest: +SKIP
        >>> job = optimizer.make(mol) # doctest: +SKIP
        >>> optimized_mol = job.output["structure"] # doctest: +SKIP
        >>> energy = job.output["properties"]["Global"]["Total Energy [Eh]"] # doctest: +SKIP
        >>>
        >>> # Optimize multiple structures in parallel
        >>> ethanol = Molecule.from_ase_atoms(molecule("C2H5OH")) # doctest: +SKIP
        >>> methane = Molecule.from_ase_atoms(molecule("CH4")) # doctest: +SKIP
        >>> water = Molecule.from_ase_atoms(molecule("H2O")) # doctest: +SKIP
        >>> job = optimizer.make([ethanol, methane, water]) # doctest: +SKIP
        >>> # Returns list of optimized structures
        >>> optimized_structures = job.output["structure"] # doctest: +SKIP
    """

    name: str = "Single Structure Maker"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Perform the computational operation on a structure.

        This method must be implemented by subclasses to define the specific
        operation to perform (e.g., optimization, property calculation).

        Args:
            structure: Pymatgen SiteCollection (Molecule or Structure) with 3D coordinates.

        Returns:
            Tuple containing:
                - Processed structure(s) (single SiteCollection or list)
                - Dictionary of computed properties (or None)

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(
        self,
        structure: SiteCollection | list[SiteCollection],
    ) -> Response[dict[str, Any]]:
        """Create a workflow job for processing structure(s).

        Automatically handles job distribution for lists of structures. Each
        structure in a list is processed as a separate job for parallel execution.

        Args:
            structure: Single Pymatgen SiteCollection or list of SiteCollections.

        Returns:
            Response containing:
                - structure: Processed structure(s)
                - files: XYZ format file(s) of the structure(s)
                - properties: Computed properties from the operation

        Examples:
            >>> from jfchemistry.conformers import CRESTConformers # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> # Generate conformers
            >>> conformer_gen = CRESTConformers(ewin=6.0) # doctest: +SKIP
            >>> job = conformer_gen.make(molecule) # doctest: +SKIP
        """
        resp = handle_structures(self, structure)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            structures, properties = self.operation(cast("SiteCollection", structure))
            if type(structures) is list:
                files = [s.to(fmt="xyz") for s in structures]
            else:
                files = [structures.to(fmt="xyz")]
            return Response(
                output={
                    "structure": structures,
                    "files": files,
                    "properties": properties,
                }
            )


@dataclass
class SingleMoleculeMaker(Maker):
    """Base class for operations on molecules without 3D geometry.

    This Maker processes RDMolMolecule objects that do not yet have assigned 3D
    coordinates. It is primarily used for structure generation tasks that convert
    molecular representations (SMILES, SMARTS) into 3D structures.

    The class handles automatic job distribution for lists of molecules and
    molecules with multiple conformers, enabling parallel processing of multiple
    structures.

    Attributes:
        name: Descriptive name for the job/operation being performed.

    Examples:
        >>> from jfchemistry.inputs import Smiles
        >>> from jfchemistry.generation import RDKitGeneration
        >>>
        >>> # Get molecule from SMILES
        >>> smiles_maker = Smiles()
        >>> smiles_job = smiles_maker.make("CCO")
        >>>
        >>> # Generate 3D structures
        >>> generator = RDKitGeneration(num_conformers=10, method="ETKDGv3")
        >>> gen_job = generator.make(smiles_job.output["structure"])
        >>>
        >>> # Access generated structures
        >>> structures = gen_job.output["structure"]  # List of Molecule objects
    """

    name: str = "Single RDMolMolecule Maker"

    def operation(
        self, mol: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], dict[str, Any]]:
        """Perform the computational operation on a molecule.

        This method must be implemented by subclasses to define the specific
        operation to perform (e.g., 3D structure generation, conformer embedding).

        Args:
            mol: RDMolMolecule to perform the operation on.

        Returns:
            Tuple containing:
                - Generated structure(s) as Pymatgen SiteCollection(s)
                - Dictionary of properties from the operation

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> # In a subclass implementation
            >>> def operation(self, mol: RDMolMolecule):
            ...     # Generate 3D structure
            ...     structure = generate_3d_structure(mol)
            ...     properties = {"energy": -123.45}
            ...     return structure, properties
        """
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(
        self,
        molecule: RDMolMolecule | list[RDMolMolecule],
    ) -> Response[dict[str, Any]]:
        """Create a workflow job for processing molecule(s).

        Automatically handles job distribution for lists of molecules and for
        molecules with multiple conformers. Each molecule or conformer is
        processed as a separate job for parallel execution.

        Args:
            molecule: Single RDMolMolecule or list of RDMolMolecule objects.

        Returns:
            Response containing:
                - structure: Generated structure(s) as Pymatgen objects
                - files: MOL format file(s) of the structure(s)
                - properties: Computed properties from the operation

        Examples:
            >>> from jfchemistry.generation import RDKitGeneration
            >>> from rdkit import Chem
            >>>
            >>> # Create RDKit molecule
            >>> mol = Chem.MolFromSmiles("c1ccccc1")
            >>> rdmol = RDMolMolecule(mol)
            >>>
            >>> # Generate multiple conformers
            >>> gen = RDKitGeneration(num_conformers=50)
            >>> job = gen.make(rdmol)
            >>>
            >>> # Each conformer becomes a separate structure
            >>> conformers = job.output["structure"]
        """
        resp = handle_molecules(self, molecule)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            molecule, properties = self.operation(cast("RDMolMolecule", molecule))
            if type(molecule) is list:
                files = [m.to(fmt="mol") for m in molecule]
            else:
                files = [molecule.to(fmt="mol")]
            return Response(
                output={
                    "structure": molecule,
                    "files": files,
                    "properties": properties,
                }
            )
