"""Base class for makers that process single structures."""

import importlib
from dataclasses import dataclass
from typing import Annotated, Any, cast

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model
from pymatgen.core.structure import SiteCollection, Structure

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


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
        self,
        structures: list[SiteCollection] | SiteCollection,
        **kwargs: Any,
    ) -> Response[_output_model] | None:
        """Distribute workflow jobs for Pymatgen structures.

        Creates individual jobs for each structure in a list. If a single structure
        is provided, returns None to indicate it should be processed directly.

        Args:
            maker: A Maker instance that will process each structure.
            structures: Either a list of SiteCollection (Molecule/Structure) objects
                or a single SiteCollection.
            **kwargs: Additional kwargs to pass to the operation.

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
                jobs.append(self.make(structure, **kwargs))  # type: ignore
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
        **kwargs: Any,
    ) -> Response[_output_model]:
        """Create a workflow job for processing structure(s).

        Automatically handles job distribution for lists of structures. Each
        structure in a list is processed as a separate job for parallel execution.

        Args:
            structure: Single Pymatgen SiteCollection or list of SiteCollections.
            **kwargs: Additional kwargs to pass to the operation.

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
        resp = self.handle_structures(structure, **kwargs)
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
