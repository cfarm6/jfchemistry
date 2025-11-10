"""Base class for ensemble site collection filters."""

from dataclasses import dataclass
from typing import Annotated, Any

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import BaseModel, Field, create_model
from pymatgen.core.structure import SiteCollection, Structure

from jfchemistry.base_jobs import Output, Properties, jfchem_job

type Ensemble = list[SiteCollection]
type EnsembleSiteCollection = Ensemble | list[EnsembleSiteCollection]

type PropertyEnsemble = list[Properties]
type PropertyEnsembleCollection = PropertyEnsemble | list[PropertyEnsembleCollection]


class EnsembleOutput(Output):
    """Output for an ensemble filter."""

    structure: Any
    files: Any
    properties: Any


@dataclass
class EnsembleFilter(Maker):
    """Base class for operations on structures with 3D geometry.

    This Maker processes a list of Pymatgen SiteCollection objects (Molecule or Structure)
    that have assigned 3D coordinates. It handles automatic job distribution for
    lists of structures and provides a common interface for ensemble generation,
    energy pre-screening, energy screening, conformer generation, and structure modifications.

    Subclasses should implement the operation() method to define specific
    computational tasks such as ensemble generation, energy pre-screening,
    energy screening, conformer generation, and geometry optimization.

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

    name: str = "Ensemble Site Collection Maker"
    _output_model: type[EnsembleOutput] = EnsembleOutput
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}

        def _nested_property_union(model: type[BaseModel], max_depth: int = 3):
            """Create a nested union of property models up to the specified depth."""
            unions: Any = model
            current_level: Any = model
            for _ in range(max_depth):
                current_level = list[current_level]  # type: ignore[assignment]
                unions |= current_level  # type: ignore[operator]
            return unions

        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    _nested_property_union(properties_model)
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
        ensemble: SiteCollection | EnsembleSiteCollection,
        properties: PropertyEnsembleCollection | None,
    ) -> Response[type[_output_model]] | None:
        """Distribute workflow jobs for Pymatgen structures.

        Creates individual jobs for each structure in a list. If a single structure
        is provided, returns None to indicate it should be processed directly.

        Args:
            maker: A Maker instance that will process each structure.
            ensemble: List of SiteCollections
            properties: List of properties for the ensemble

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
        if isinstance(ensemble, SiteCollection):
            output = self._output_model(
                structure=[ensemble],
                files=[file for file in [self.write_file(ensemble)] if file is not None],
                properties=[properties] if properties is not None else None,
            )
            return Response(output=output)  # type: ignore
        jobs: list[Response[type[self._output_model]]] = []

        def _is_base_ensemble(value: EnsembleSiteCollection) -> bool:
            return isinstance(value, list) and all(
                isinstance(item, SiteCollection) for item in value
            )

        if _is_base_ensemble(ensemble):
            return None

        for _ensemble, _properties in zip(
            ensemble, properties if properties is not None else [None] * len(ensemble), strict=False
        ):
            jobs.append(self.make(_ensemble, _properties))

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
        self, ensemble: Ensemble, properties: PropertyEnsemble
    ) -> tuple[Ensemble, PropertyEnsemble]:
        """Perform the computational operation on an ensemble.

        This method must be implemented by subclasses to define the specific
        operation to perform (e.g., ensemble generation, energy pre-screening,
        energy screening, conformer generation, and geometry optimization).

        Args:
            ensemble: List of Pymatgen SiteCollection (Molecule or Structure) with 3D coordinates.
            properties: List of properties for the ensemble.

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
        ensemble: EnsembleSiteCollection,
        properties: PropertyEnsembleCollection,
    ) -> Response[_output_model]:
        """Create a workflow job for processing an ensemble.

        Automatically handles job distribution for lists of structures. Each
        structure in a list is processed as a separate job for parallel execution.

        Args:
            ensemble: List of Pymatgen SiteCollection or list of SiteCollections.
            properties: List of properties for the ensemble.

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
        resp = self.handle_structures(ensemble, properties)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            ensemble, properties = self.operation(ensemble, properties)
            if type(ensemble) is list:
                files = [self.write_file(s) for s in ensemble]
            else:
                files = [self.write_file(ensemble)]
            if properties is not None:
                properties = [
                    Properties.model_validate(property, extra="allow", strict=False)
                    for property in properties
                ]
            return Response(
                output=self._output_model(
                    structure=ensemble,
                    files=files,
                    properties=properties,
                ),
            )
