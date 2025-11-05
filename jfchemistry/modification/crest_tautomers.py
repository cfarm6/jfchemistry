"""CREST-based tautomerization for generating tautomers.

This module provides integration with CREST's automated tautomerization workflow
for generating low-energy tautomers at different sites.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

from pymatgen.core.structure import SiteCollection
from pymatgen.io.xyz import XYZ

from jfchemistry.calculators import CRESTCalculator
from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTTautomers(StructureModification, CRESTCalculator):
    """Generate tautomers using CREST.

    Uses CREST's automated tautomerization workflow to identify basic sites
    and generate low-energy tautomers. The method systematically
    explores different tautomer sites and optimizes the resulting structures.

    Attributes:
        name: Name of the job (default: "CREST Tautomers").
        runtype: Workflow type (default: "tautomerize").
        ewin: Energy window in kcal/mol for selecting tautomers
            (default: None, uses CREST default).
        ffopt: Perform force field pre-optimization (default: True).
        freezeopt: Freeze constraint string for optimization (default: None).
        finalopt: Perform final optimization (default: True).
        threads: Number of parallel threads (default: 1).

    References:
        - CREST Documentation: https://crest-lab.github.io/crest-docs/

    Examples:
        >>> from jfchemistry.modification import CRESTTautomers # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>> prot = CRESTTautomers(ewin=6.0, threads=4) # doctest: +SKIP
        >>> job = prot.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
        >>> prot_custom = CRESTTautomers( # doctest: +SKIP
        ...     ewin=8.0, # doctest: +SKIP
        ...     ffopt=True, # doctest: +SKIP
        ...     finalopt=True, # doctest: +SKIP
        ...     threads=8 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = prot_custom.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
    """

    name: str = "CREST Tautomers"
    # INTERNAL
    _runtype: Literal["tautomerize"] = "tautomerize"

    def make_commands(self):
        """Make the CLI for the CREST input."""
        super().make_commands()
        self._commands.append(f"--{self._runtype}")
        self._commands.append("--newversion")

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Generate protonated structures using CREST.

        Runs CREST's protonation workflow to identify basic sites and
        generate optimized protonated structures.

        Args:
            structure: Input molecular structure with 3D coordinates. The
                structure's charge is used for the CREST calculation.

        Returns:
            Tuple containing:
                - List of protonated structures sorted by energy
                - None (no additional properties)

        Examples:
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from jfchemistry.modification import CRESTTautomers # doctest: +SKIP
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> prot = CRESTTautomers(ewin=6.0, threads=4) # doctest: +SKIP
            >>> structures, properties = prot.operation(molecule) # doctest: +SKIP
        """
        structure.to("input.xyz", fmt="xyz")
        if self.charge is None and structure.charge is not None:
            self.charge = structure.charge
        super().make_dict()
        super().write_toml()
        self.make_commands()
        super().run()

        try:
            structures = XYZ.from_file("tautomers.xyz").all_molecules
        except IndexError:
            raise IndexError(
                "No tautomers found. Please check your CREST settings and log file."
            ) from None

        structures = cast("list[SiteCollection]", structures)
        return structures, None
