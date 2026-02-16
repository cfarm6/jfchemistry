"""Base class for makers in JFChemistry."""

from dataclasses import dataclass, field
from typing import Type

from jobflow.core.job import Response
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers.core_maker import CoreMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


@dataclass
class JFChemMaker[InputType, OutputType](CoreMaker):
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

    def _write_file(self, structure: Structure | Molecule) -> str | None:
        """Write the structure to a file."""
        if isinstance(structure, Structure):
            return structure.to(fmt="cif")
        elif isinstance(structure, Molecule):
            return structure.to(fmt="xyz")

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Perform the computational operation on a structure."""
        raise NotImplementedError

    def _handle_structures(self, input_list: list, **kwargs) -> Response[_output_model]:
        """Distribute workflow jobs for Pymatgen structures.

        Creates individual jobs for each structure in a list. If a single structure
        is provided, returns None to indicate it should be processed directly.

        Args:
            input_list: Either a list of SiteCollection (Molecule/Structure) objects
                or a single SiteCollection.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Response containing distributed jobs if structures is a list, None if
            structures is a single SiteCollection to be processed directly.
        """
        jobs = [self.make(input, **kwargs) for input in input_list]

        output = self._output_model(
            structure=[job.output.structure for job in jobs],
            files=[job.output.files for job in jobs],
            properties=[job.output.properties for job in jobs],
        )

        return Response(
            output=output,
            detour=jobs,
        )

    def _run_job(self, input: InputType | list[InputType], **kwargs) -> Response[_output_model]:
        """Run the job for a single structure or a list of structures."""
        if (not self._ensemble) and isinstance(input, list):
            return self._handle_structures(input, **kwargs)
        output, properties = self._operation(input, **kwargs)
        if isinstance(output, list):
            files = [self._write_file(s) for s in output]
        else:
            files = [self._write_file(output)]
        return Response(
            output=self._output_model(
                structure=output,
                files=files,
                properties=properties,
            ),
        )

    @jfchem_job()
    def make(
        self,
        input: InputType | list[InputType],
        **kwargs,
    ) -> Response[_output_model]:
        """Create a workflow job for processing structure(s).

        Automatically handles job distribution for lists of structures. Each
        structure in a list is processed as a separate job for parallel execution.

        Args:
            input: Single Pymatgen SiteCollection or list of SiteCollections.
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
            >>> job = conformer_gen.make(input) # doctest: +SKIP
        """
        return self._run_job(input, **kwargs)
