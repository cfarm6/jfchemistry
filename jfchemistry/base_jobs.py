"""Base Job Classes for single molecules."""

import importlib
from dataclasses import dataclass
from typing import Annotated, Any, Optional, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from monty.json import MontyDecoder
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pymatgen.core.structure import SiteCollection, Structure
from rdkit.Chem import rdchem

from jfchemistry.base_classes import Property, RDMolMolecule


class PropertyClass(BaseModel):
    """Class for property classes."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def convert_extra_fields(cls, data: Any) -> Property:
        """Convert extra fields to Property objects."""
        if not isinstance(data, dict):
            return data

        # Get known field names from the model
        known_fields = cls.model_fields.keys()

        # Convert only the extra fields
        for key, value in data.items():
            if key not in known_fields and isinstance(value, dict):
                data[key] = Property(**value)

        return data


class Properties(BaseModel):
    """Properties of the structure."""

    atomic: Optional[PropertyClass] = None
    bond: Optional[PropertyClass] = None
    system: Optional[PropertyClass] = None
    orbital: Optional[PropertyClass] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create a Property from a dictionary."""
        return cls.model_validate(d, extra="ignore", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Property to a dictionary."""
        return self.model_dump(mode="json")


class Output(BaseModel):
    """Output of the job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Optional[Any] = None
    properties: Optional[Any] = None
    files: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create an Output from a dictionary."""
        if isinstance(d["structure"], dict):
            d["structure"] = MontyDecoder().process_decoded(d["structure"])
        return cls.model_validate(d, extra="allow", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Output to a dictionary."""
        return self.model_dump(mode="json")


def jfchem_job():
    """Decorator that wraps @job and automatically pulls a field from the parent class.

    Args:
        field_name: Name of the class attribute to access
        **extra_job_kwargs: Additional kwargs to pass to @job
    """

    class DeferredJobDecorator:
        def __init__(self, func):
            self.func = func
            self.field_name = "_output_model"

        def __set_name__(self, owner, name):
            # Get the field value from the class
            field_value = getattr(owner, self.field_name)

            # Apply the @job decorator with the kwargs
            decorated_func = job(
                self.func,
                output_schema=field_value,
                files="files",
                properties="properties",
            )

            # Replace this descriptor with the decorated function
            setattr(owner, name, decorated_func)

    return DeferredJobDecorator


def write_file(structure: SiteCollection) -> str | None:
    """Write the structure to a file."""
    if isinstance(structure, Structure):
        return structure.to(fmt="cif")
    else:
        return structure.to(fmt="xyz")


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
    _output_model: type[Output] = Output
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[Properties]):
        """Make a properties model for the job."""
        fields = {}
        if isinstance(self._output_model, dict):
            module = self._output_model["@module"]
            class_name = self._output_model["@callable"]
            self._output_model = getattr(importlib.import_module(module), class_name)
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    properties_model
                    | list[type[properties_model]]
                    | OutputReference
                    | list[OutputReference]
                )  # type: ignore

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
        self.make_output_model(self._properties_model)

    def handle_structures(
        self, structures: list[SiteCollection] | SiteCollection
    ) -> Response[_output_model] | None:
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
        jobs: list[Response[type[self._output_model]]] = []
        if isinstance(structures, list):
            for structure in structures:
                jobs.append(self.make(structure))  # type: ignore
        else:
            return None

        output = self._output_model(
            structure=[job.output.structure for job in jobs],
            files=[job.output.files for job in jobs],
            properties=[job.output.properties for job in jobs],
        )
        return Response(
            output=output,
            detour=jobs,  # type: ignore
        )

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], _properties_model]:
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

    def write_file(self, structure: SiteCollection) -> str | None:
        """Write the structure to a file."""
        if isinstance(structure, Structure):
            return structure.to(fmt="cif")
        else:
            return structure.to(fmt="xyz")

    @jfchem_job()
    def make(
        self,
        structure: SiteCollection | list[SiteCollection],
    ) -> Response[_output_model]:
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
            >>> from pymatgen.core import Molecule # dokctest: +SKIP
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> # Generate conformers
            >>> conformer_gen = CRESTConformers(ewin=6.0) # doctest: +SKIP
            >>> job = conformer_gen.make(molecule) # doctest: +SKIP
        """
        if isinstance(structure, list):
            if len(structure) == 1:
                structure = structure[0]
        resp = self.handle_structures(structure)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            structures, properties = self.operation(cast("SiteCollection", structure))
            if isinstance(structures, list):
                files = [self.write_file(s) for s in structures]
            else:
                files = [self.write_file(structures)]
            return Response(
                output=self._output_model(
                    structure=structures,
                    files=files,
                    properties=properties,
                ),
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

    """

    name: str = "Single RDMolMolecule Maker"
    _output_model: type[Output] = Output
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[Properties]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    properties_model
                    | list[type[properties_model]]
                    | OutputReference
                    | list[OutputReference]
                )  # type: ignore

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
        self.make_output_model(self._properties_model)

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

    def handle_conformers(self, structure: RDMolMolecule) -> Response[_output_model]:
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

        """
        jobs: list[Response[type[self._output_model]]] = []
        for confId in range(structure.GetNumConformers()):
            s = RDMolMolecule(rdchem.Mol(structure, confId=confId))
            jobs.append(self.make(s))  # type: ignore

        return Response(
            output=self._output_model(
                structure=[job.output.structure for job in jobs],
                files=[job.output.files for job in jobs],
                properties=[job.output.properties for job in jobs],
            ),
            detour=jobs,  # type: ignore
        )

    def handle_list_of_structures(self, structures: list[RDMolMolecule]) -> Response[_output_model]:
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
        jobs: list[Response[type[self._output_model]]] = []
        for structure in structures:
            if structure.GetNumConformers() > 1:
                jobs.append(self.handle_conformers(structure))
            else:
                jobs.append(self.make(structure))  # type: ignore

        return Response(
            output=self._output_model(
                structure=[job.output.structure for job in jobs],
                files=[job.output.files for job in jobs],
                properties=[job.output.properties for job in jobs],
            ),
            detour=jobs,  # type: ignore
        )

    def handle_molecules(
        self, structure: RDMolMolecule | list[RDMolMolecule]
    ) -> Response[_output_model] | None:
        """Route RDKit molecules to appropriate job distribution handler.

        Determines the appropriate distribution strategy based on whether the input
        is a single molecule, a list of molecules, or a molecule with multiple conformers.

        Args:
            maker: A Maker instance that will process the molecule(s).
            structure: Either a single RDMolMolecule or a list of RDMolMolecule structures.

        Returns:
            Response from the appropriate handler if distribution is needed, None if
            the structure should be processed directly by the maker.

        """
        if type(structure) is list:
            return self.handle_list_of_structures(structure)
        elif cast("RDMolMolecule", structure).GetNumConformers() > 1:
            return self.handle_conformers(cast("RDMolMolecule", structure))
        else:
            return None

    @job(files="files", properties="properties")
    def make(
        self,
        molecule: RDMolMolecule | list[RDMolMolecule],
    ) -> Response[_output_model]:
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
        """
        resp = self.handle_molecules(molecule)
        if resp is not None:
            return resp
        molecule, properties = self.operation(cast("RDMolMolecule", molecule))
        if isinstance(molecule, list):
            files = [m.to(fmt="mol") for m in molecule]
        else:
            files = [molecule.to(fmt="mol")]
        return Response(
            output=self._output_model(
                structure=molecule,
                files=files,
                properties=properties,
            ),
        )
