"""Base class for structure modifications."""

from dataclasses import dataclass
from typing import Any, Optional, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from pymatgen.core.structure import IMolecule

from jfchemistry.utils.bulk_jobs import handle_structures


@dataclass
class StructureModification(Maker):
    """Maker for modifying a structure."""

    name: str = "Structure Modification"

    def modify_structure(
        self, structure: IMolecule
    ) -> tuple[Optional[list[IMolecule] | IMolecule], Optional[dict[str, Any]]]:
        """Generate a structure."""
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(self, molecule: IMolecule | list[IMolecule]) -> Response[dict[str, Any]]:
        """Make the job."""
        resp = handle_structures(self, molecule)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            print(len(molecule))
            structures, properties = self.modify_structure(cast("IMolecule", molecule))
            print(len(structures[0]))
            if structures is None:
                return Response(stop_children=True)

            if isinstance(structures, list):
                files = [structure.to(fmt="xyz") for structure in structures]
            else:
                files = [structures.to(fmt="xyz")]

            return Response(
                output={
                    "structure": structures,
                    "files": files,
                    "properties": properties,
                }
            )
