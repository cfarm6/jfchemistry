"""CREST-based deprotonation for generating deprotonated structures.

This module provides integration with CREST's automated deprotonation workflow
for generating low-energy deprotonated structures and tautomers.
"""

import os
from dataclasses import dataclass
from typing import Literal

from pymatgen.core.structure import Molecule

from jfchemistry.calculators.crest import CRESTCalculator
from jfchemistry.core.makers.single_molecule import SingleMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.modification.deprotonation.base import DeprotonationMaker
from jfchemistry.modification.molbar_screening import molbar_screening


@dataclass
class CRESTDeprotonation(DeprotonationMaker, CRESTCalculator, SingleMoleculeMaker):
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
    _output_filename: str = "deprotonated.xyz"

    def make_commands(self):
        """Make the CLI for the CREST input."""
        super().make_commands()
        self._commands.append(f"--{self._runtype}")
        self._commands.append("--newversion")

    def operation(
        self, molecule: Molecule
    ) -> tuple[Molecule | list[Molecule], Properties | list[Properties]]:
        """Generate deprotonated structures using CREST.

        Runs CREST's deprotonation workflow to identify acidic sites and
        generate optimized deprotonated structures. The calculation uses
        GFN2-xTB with Wiberg bond order analysis.

        Args:
            molecule: Input molecular structure with 3D coordinates. The
                molecule's charge is used for the CREST calculation.

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
        molecule.to("input.xyz", fmt="xyz")
        if self.charge is None and molecule.charge is not None:
            self.charge = molecule.charge
        super().make_dict()
        super().write_toml()
        self.make_commands()
        super().run()
        if not os.path.exists(self._output_filename):
            raise FileNotFoundError(
                "No deprotonated structures found. Please check your CREST settings and log file."
            ) from None
        molecules = molbar_screening(self._output_filename, self.threads)
        for i, deprotonated_structure in enumerate(molecules):
            molecules[i] = deprotonated_structure.set_charge_and_spin(
                charge=deprotonated_structure.charge - 1,
                spin_multiplicity=int(deprotonated_structure.charge - 1) // 2 + 1,
            )
        return molecules, Properties()
