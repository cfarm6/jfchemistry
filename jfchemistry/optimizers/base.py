"""Base class for structure generation."""

from dataclasses import dataclass
from typing import Any, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from pymatgen.core.structure import IMolecule

from jfchemistry.utils.bulk_jobs import handle_structures


@dataclass
class GeometryOptimization(Maker):
    """Maker for generating a structure."""

    name: str = "Geometry Optimization"

    def optimize_structure(self, structure: IMolecule) -> tuple[IMolecule, dict[str, Any]]:
        """Generate a structure."""
        raise NotImplementedError

    def get_properties(self, structure: Any) -> dict[str, Any]:
        """Get the properties of the structure."""
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(self, molecule: IMolecule | list[IMolecule]) -> Response[dict[str, Any]]:
        """Make the job."""
        resp = handle_structures(self, molecule)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            structure, properties = self.optimize_structure(cast("IMolecule", molecule))
            if structure is None:
                return Response(stop_children=True)
            structure.to("log.xyz")
            return Response(
                output={
                    "structure": structure,
                    "files": structure.to(fmt="mol"),
                    "properties": properties,
                }
            )
