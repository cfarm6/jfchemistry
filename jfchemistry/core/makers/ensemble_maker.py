"""Base class for operations on ensembles of structures."""

from dataclasses import dataclass
from typing import Type

from jobflow.core.job import Response
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import (
    JFChemistryBaseMaker,
    RecursiveMoleculeList,
    RecursiveStructureList,
)
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties

type PropertyEnsemble = list[Properties] | Properties
type PropertyEnsembleCollection = PropertyEnsemble | list[PropertyEnsembleCollection]


class EnsembleOutput(Output):
    """Output for an ensemble filter."""


@dataclass
class EnsembleMaker[
    InputType: RecursiveStructureList | RecursiveMoleculeList,  # type: ignore[type-var]
    OutputType: RecursiveStructureList | RecursiveMoleculeList,  # type: ignore[type-var]
](
    JFChemistryBaseMaker[InputType, OutputType]  # type: ignore[type-arg]
):
    """Base class for operations on ensembles of structures.

    This Maker processes a list of Pymatgen Structure objects. It is designed as the base class for filters on structure ensembles.

    Attributes:
        name: Descriptive name for the job/operation being performed.

    Examples:
        >>> from jfchemistry.filters import StructureEnsembleFilter # doctest: +SKIP
        >>> from pymatgen.core.structure import Structure # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>>
        >>> # Create a structure ensemble
        >>> structure1 = Structure.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> structure2 = Structure.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> structure3 = Structure.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> ensemble = [structure1, structure2, structure3] # doctest: +SKIP
        >>> # Create a filter
        >>> filter = StructureEnsembleFilter() # doctest: +SKIP
        >>> # Filter the ensemble
        >>> filtered_ensemble = filter.make(ensemble) # doctest: +SKIP
        >>> # Returns filtered ensemble
        >>> filtered_ensemble = filtered_ensemble.output["structure"] # doctest: +SKIP
    """

    name: str = "Ensemble Filter"
    distribute: bool = False
    _output_model: Type[EnsembleOutput] = EnsembleOutput
    _properties_model: Type[PropertyEnsemble] = PropertyEnsemble
    _ensemble_type = InputType | list[InputType]
    _ensemble_collection_type = _ensemble_type | list[_ensemble_type]

    def _distribute_ensembles(
        self,
        ensemble: _ensemble_collection_type,
        properties: PropertyEnsembleCollection | None = None,
    ) -> Response[_output_model] | None:
        jobs: list[Response] = []

        def is_base_ensemble(value):
            return isinstance(value, list) and all(
                isinstance(item, Molecule | Structure) for item in value
            )

        if is_base_ensemble(ensemble):
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

    def _operation(
        self,
        ensemble: _ensemble_collection_type,
        properties: PropertyEnsembleCollection,
    ) -> tuple[_ensemble_collection_type, PropertyEnsembleCollection]:
        raise NotImplementedError

    @jfchem_job()
    def make(
        self, ensemble: _ensemble_collection_type, properties: PropertyEnsembleCollection
    ) -> Response:
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
        """
        resp = self._distribute_ensembles(ensemble, properties)
        if resp is not None:
            return resp
        else:
            ensemble, properties = self._operation(ensemble, properties)
            files = [self._write_file(s) for s in ensemble if isinstance(s, Structure | Molecule)]
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
