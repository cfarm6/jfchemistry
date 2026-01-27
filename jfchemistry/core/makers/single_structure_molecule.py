"""Base class for makers that process single structures."""

from dataclasses import dataclass

from jobflow.core.job import Response
from pymatgen.core import Molecule
from pymatgen.core.structure import Structure

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers.pymatgen_base_maker import PymatgenBaseMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


@dataclass
class SingleStructureMoleculeMaker[T: Molecule | Structure](PymatgenBaseMaker[T]):
    """Base class for operations on single structures."""

    name: str = "Single Structure Molecule Maker"
    _output_model: type[Output] = Output
    _properties_model: type[Properties] = Properties

    def _operation(self, structure: T) -> tuple[T | list[T], Properties | list[Properties]]:
        """Perform the computational operation on a structure."""
        raise NotImplementedError

    @jfchem_job()
    def make(
        self,
        structure: T | list[T],
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
        return self._run_job(structure)
