"""Base class for makers that process single molecules."""

from dataclasses import dataclass
from typing import Annotated, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model
from pymatgen.core.structure import SiteCollection
from rdkit.Chem import rdchem

from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import RDMolMolecule


@dataclass
class SingleRDMoleculeMaker(Maker):
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

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        self.make_output_model(self._properties_model)

    def operation(
        self, mol: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], Properties | list[Properties]]:
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
        jobs: list[Response] = []
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
        jobs: list[Response] = []
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
        structures, properties = self.operation(cast("RDMolMolecule", molecule))
        if isinstance(structures, SiteCollection):
            files = [structures.to(fmt="cif")]
        else:
            files = [m.to(fmt="cif") for m in structures]
        return Response(
            output=self._output_model(
                structure=structures,
                files=files,
                properties=properties,
            ),
        )
