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
class SingleStructureMoleculeMaker(PymatgenBaseMaker):
    """Base class for operations on single structures.

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

    name: str = "Single Structure Molecule Maker"
    _output_model: type[Output] = Output
    _properties_model: type[Properties] = Properties

    def operation(
        self, structure: Molecule | Structure
    ) -> tuple[Molecule | Structure | list[Molecule] | list[Structure], Properties]:
        """Perform the computational operation on a structure."""
        raise NotImplementedError

    @jfchem_job()
    def make(
        self,
        structure: Molecule | Structure | list[Molecule] | list[Structure],
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
        return self.run_job(structure)
