"""Maker for single structure or molecule operations"""

from dataclasses import dataclass, field
from typing import Type

from jobflow.core.job import Response

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import (
    JFChemistryBaseMaker,
    RecursiveMoleculeList,
    RecursiveStructureList,
)
from jfchemistry.core.outputs import Output


@dataclass
class NewEnsembleMaker[
    InputType: RecursiveStructureList | RecursiveMoleculeList,
    OutputType: RecursiveStructureList | RecursiveMoleculeList,
](JFChemistryBaseMaker[InputType, OutputType]):
    """Base class for makers that process single structures or molecules."""

    _output_model: Type[Output] = Output
    _ensemble: bool = field(default=True)

    @jfchem_job()
    def make(
        self,
        structure: InputType | list[InputType],
        **kwargs,
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
        return self._run_job(structure, **kwargs)
