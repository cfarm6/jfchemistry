"""Base class for structure generation."""

from dataclasses import dataclass, field
from typing import Any

from pymatgen.core.structure import SiteCollection

from jfchemistry import RDMolMolecule, SingleMoleculeMaker


@dataclass
class StructureGeneration(SingleMoleculeMaker):
    """Maker for generating a structure."""

    # Input parameters
    name: str = "Structure Generation"
    # Check the structure with PoseBusters
    check_structure: bool = field(default=False)

    def operation(
        self, structure: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], dict[str, Any]]:
        """Generate a structure."""
        raise NotImplementedError
