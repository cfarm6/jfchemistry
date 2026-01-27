"""Base class for makers in JFChemistry."""

import importlib
from dataclasses import dataclass, field
from typing import Annotated, Type

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties

type RecursiveStructureList = Structure | list[RecursiveStructureList]
type RecursiveMoleculeList = Molecule | list[RecursiveMoleculeList]


@dataclass
class JFChemistryBaseMaker[
    InputType: RecursiveMoleculeList | RecursiveStructureList,
    OutputType: RecursiveMoleculeList | RecursiveStructureList,
](Maker):
    """Base class for operations on structures with 3D geometry.

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
    _output_model: Type[Output] = Output
    _properties_model: Type[Properties] = Properties
    _ensemble: bool = field(default=False)

    def _make_output_model(self, properties_model: type[Properties]):
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
                    | list[properties_model]  # type: ignore
                    | OutputReference
                    | list[OutputReference]
                )  # type: ignore
            elif f_name == "structure":
                annotation = OutputType | list[OutputType] | OutputReference | list[OutputReference]
            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
                    *f_dict["metadata"],
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
        self._make_output_model(self._properties_model)

    def _write_file(self, structure: Structure | Molecule) -> str | None:
        """Write the structure to a file."""
        if isinstance(structure, Structure):
            return structure.to(fmt="cif")
        elif isinstance(structure, Molecule):
            return structure.to(fmt="xyz")

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Perform the computational operation on a structure."""
        raise NotImplementedError

    def _handle_structures(
        self, structures: InputType | list[InputType], **kwargs
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
        """
        jobs: list[Response] = []
        for structure in structures:
            jobs.append(self.make(structure, **kwargs))  # type: ignore

        output = self._output_model(
            structure=[job.output.structure for job in jobs],
            files=[job.output.files for job in jobs],
            properties=[job.output.properties for job in jobs],
        )
        return Response(
            output=output,
            detour=jobs,  # type: ignore
        )

    def _run_job(self, structure: InputType | list[InputType], **kwargs) -> Response[_output_model]:
        """Run the job for a single structure or a list of structures."""
        if (not self._ensemble) and isinstance(structure, list):
            return self._handle_structures(structure, **kwargs)
        structures, properties = self._operation(structure, **kwargs)
        if isinstance(structures, list):
            files = [self._write_file(s) for s in structures]
        else:
            files = [self._write_file(structures)]
        return Response(
            output=self._output_model(
                structure=structures,
                files=files,
                properties=properties,
            ),
        )
