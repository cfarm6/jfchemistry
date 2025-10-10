"""Base class for structure generation."""

from dataclasses import dataclass, field
from typing import Any, Union, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from pymatgen.core.structure import IMolecule

from jfchemistry.jfchemistry import RDMolMolecule
from jfchemistry.utils.bulk_jobs import handle_molecule


@dataclass
class StructureGeneration(Maker):
    """Maker for generating a structure."""

    # Input parameters
    name: str = field(default="Structure Generation")
    # Check the structure with PoseBusters
    check_structure: bool = field(default=False)

    def generate_structure(self, structure: RDMolMolecule) -> Union[IMolecule, None]:
        """Generate a structure."""
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(self, molecule: RDMolMolecule | list[RDMolMolecule]) -> Response[dict[str, Any]]:
        """Make the job."""
        resp = handle_molecule(self, molecule)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            structure = self.generate_structure(cast("RDMolMolecule", molecule))
            if structure is None:
                return Response(stop_children=True)
            structure.to("log.xyz")
            return Response(
                output={
                    "structure": structure,
                    "files": structure.to(fmt="mol"),
                    "properties": None,
                }
            )
