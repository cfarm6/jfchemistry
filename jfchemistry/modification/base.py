"""Base class for structure modifications."""

from dataclasses import dataclass
from typing import Any, Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry import SingleStructureMaker


@dataclass
class StructureModification(SingleStructureMaker):
    """Maker for modifying a structure."""

    name: str = "Structure Modification"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Generate a structure."""
        raise NotImplementedError
