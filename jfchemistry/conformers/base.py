"""Base class for conformer generation."""

from dataclasses import dataclass
from typing import Any, Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry import SingleStructureMaker


@dataclass
class ConformerGeneration(SingleStructureMaker):
    """Maker for generating a structure."""

    name: str = "Conformer Generation"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Generate conformers."""
        raise NotImplementedError
