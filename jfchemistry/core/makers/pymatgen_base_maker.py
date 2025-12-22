"""Base class for makers that process single structures."""

import importlib
from dataclasses import dataclass, field
from typing import Annotated, Type

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model
from pymatgen.core.structure import Molecule, SiteCollection, Structure

from jfchemistry.calculators.base import Calculator
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


@dataclass
class PymatgenBaseMaker(Maker):
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

    name: str = "Single Structure Calculator Maker"
    calculator: Calculator = field(default_factory=lambda: Calculator())
    _output_model: Type[Output] = Output
    _properties_model: Type[Properties] = Properties

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
                    | list[type[properties_model]]  # type: ignore
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
        self._properties_model = properties_model

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        self.make_output_model(self.calculator._properties_model)

    def handle_structures(
        self,
        structures: Structure | Molecule | list[Structure] | list[Molecule],
    ) -> Response[_output_model]:
        """Distribute workflow jobs for Pymatgen structures.

        Creates individual jobs for each structure in a list. If a single structure
        is provided, returns None to indicate it should be processed directly.

        Args:
            maker: A Maker instance that will process each structure.
            structures: Either a list of SiteCollection (Molecule/Structure) objects
                or a single SiteCollection.
            calculator: Calculator to use for the calculation.
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
        jobs: list[Response] = []
        for structure in structures:
            jobs.append(self.make(structure))  # type: ignore

        output = self._output_model(
            structure=[job.output.structure for job in jobs],
            files=[job.output.files for job in jobs],
            properties=[job.output.properties for job in jobs],
        )
        return Response(
            output=output,
            detour=jobs,  # type: ignore
        )

    def write_file(self, structure: SiteCollection) -> str | None:
        """Write the structure to a file."""
        if isinstance(structure, Structure):
            return structure.to(fmt="cif")
        elif isinstance(structure, Molecule):
            return structure.to(fmt="xyz")

    def operation(*args, **kwargs):
        """Perform the computational operation on a structure.

        This method must be implemented by subclasses to define the specific
        operation to perform (e.g., optimization, property calculation).

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple containing:
                - Processed structure(s) (single SiteCollection or list)
                - Dictionary of computed properties (or None)

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def run_job(
        self, sitecollection: Structure | Molecule | list[Structure] | list[Molecule]
    ) -> Response[_output_model]:
        """Run the job for a single structure or a list of structures."""
        if isinstance(sitecollection, list) and len(sitecollection) == 1:
            sitecollection = sitecollection[0]
        elif isinstance(sitecollection, list):
            return self.handle_structures(sitecollection)

        structures, properties = self.operation(sitecollection)
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
