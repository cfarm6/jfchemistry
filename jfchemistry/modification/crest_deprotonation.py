"""CREST-based deprotonation for generating deprotonated structures.

This module provides integration with CREST's automated deprotonation workflow
for generating low-energy deprotonated structures and tautomers.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

from pymatgen.core.structure import Molecule
from pymatgen.io.xyz import XYZ

from jfchemistry.calculators import CRESTCalculator
from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTDeprotonation(StructureModification, CRESTCalculator):
    """Generate deprotonated structures using CREST.

    Uses CREST's automated deprotonation workflow to identify acidic sites
    and generate low-energy deprotonated structures. The method systematically
    explores different deprotonation sites and optimizes the resulting structures
    using GFN2-xTB.

    Attributes:
        name: Name of the job (default: "CREST Deprotonation").
        ewin: Energy window in kcal/mol for selecting deprotonated structures
            (default: None, uses CREST default). Structures within ewin of the
            lowest energy structure are retained.

    References:
        - CREST Documentation: https://crest-lab.github.io/crest-docs/

    Examples:
        >>> from jfchemistry.modification import CRESTDeprotonation # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> >>> from ase.build import molecule # doctest: +SKIP
        >>> ethane = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> # Deprotonate a ethane
        >>> deprot = CRESTDeprotonation(ewin=6.0) # doctest: +SKIP
        >>> job = deprot.make(ethane) # doctest: +SKIP
        >>> deprotonated_structures = job.output["structure"] # doctest: +SKIP
        >>>
        >>> deprot_default = CRESTDeprotonation() # doctest: +SKIP
        >>> job = deprot_default.make(ethane) # doctest: +SKIP
    """

    name: str = "CREST Deprotonation"

    # INTERNAL
    _runtype: Literal["deprotonate"] = "deprotonate"

    def make_commands(self):
        """Make the CLI for the CREST input."""
        super().make_commands()
        self._commands.append(f"--{self._runtype}")
        self._commands.append("--newversion")

    def operation(
        self, structure: Molecule
    ) -> tuple[Molecule | list[Molecule], Optional[dict[str, Any]]]:
        """Generate deprotonated structures using CREST.

        Runs CREST's deprotonation workflow to identify acidic sites and
        generate optimized deprotonated structures. The calculation uses
        GFN2-xTB with Wiberg bond order analysis.

        Args:
            structure: Input molecular structure with 3D coordinates. The
                structure's charge is used for the CREST calculation.

        Returns:
            Tuple containing:
                - List of deprotonated structures sorted by energy
                - None (no additional properties)

        Examples:
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from ase.build import molecule # doctest: +SKIP
            >>> ethane = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> deprot = CRESTDeprotonation(ewin=8.0) # doctest: +SKIP
            >>> structures, props = deprot.operation(ethane) # doctest: +SKIP
            >>> print(f"Generated {len(structures)} deprotonated structures") # doctest: +SKIP
        """
        structure.to("input.xyz", fmt="xyz")
        if self.charge is None and structure.charge is not None:
            self.charge = structure.charge
        super().make_dict()
        super().write_toml()
        self.make_commands()
        super().run()

        try:
            structures = XYZ.from_file("deprotonated.xyz").all_molecules
        except IndexError:
            raise IndexError(
                "No deprotonated structures found. Please check your CREST settings and log file."
            ) from None
        structures = cast("list[Molecule]", structures)
        for i, deprotonated_structure in enumerate(structures):
            structures[i] = deprotonated_structure.set_charge_and_spin(
                charge=deprotonated_structure.charge - 1,
                spin_multiplicity=int(deprotonated_structure.charge - 1) // 2 + 1,
            )
        return structures, None
