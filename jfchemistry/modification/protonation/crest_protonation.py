"""CREST-based protonation for generating protonated structures.

This module provides integration with CREST's automated protonation workflow
for generating low-energy protonated structures at different sites.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pymatgen.core.structure import Molecule

from jfchemistry.calculators.crest import CRESTCalculator
from jfchemistry.modification.molbar_screening import molbar_screening
from jfchemistry.modification.protonation.base import ProtonationMaker


@dataclass
class CRESTProtonation(ProtonationMaker, CRESTCalculator):
    """Generate protonated structures using CREST.

    Uses CREST's automated protonation workflow to identify basic sites
    and generate low-energy protonated structures. The method systematically
    explores different protonation sites and optimizes the resulting structures.

    Attributes:
        name: Name of the job (default: "CREST Protonation").
        runtype: Workflow type (default: "protonate").
        ion: Ion to add for protonation (default: None, uses H+).
        ion_charge: Charge of the ion (default: 1).

    References:
        - CREST Documentation: https://crest-lab.github.io/crest-docs/

    Examples:
        >>> from jfchemistry.modification import CRESTProtonation # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH")) # doctest: +SKIP
        >>> prot = CRESTProtonation(ewin=6.0, threads=4) # doctest: +SKIP
        >>> job = prot.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
        >>> prot_custom = CRESTProtonation( # doctest: +SKIP
        ...     ewin=8.0, # doctest: +SKIP
        ...     ffopt=True, # doctest: +SKIP
        ...     finalopt=True, # doctest: +SKIP
        ...     threads=8 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = prot_custom.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
        >>> prot_solv = CRESTProtonation( # doctest: +SKIP
        ...     ewin=6.0, # doctest: +SKIP
        ...     solvation=("alpb", "water")  # ALPB with water # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = prot_solv.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
        >>> prot_gbsa = CRESTProtonation( # doctest: +SKIP
        ...     ewin=6.0, # doctest: +SKIP
        ...     solvation=("gbsa", "DMSO")  # GBSA with DMSO # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = prot_gbsa.make(molecule) # doctest: +SKIP
        >>> protonated_structures = job.output["structure"] # doctest: +SKIP
    """

    name: str = "CREST Protonation"
    ion: Optional[str] = field(
        default=None,
        metadata={"description": "the ion to add for protonation. Default is none for H+"},
    )
    ion_charge: int = field(
        default=1,
        metadata={
            "description": "the charge of the ion to add for protonation. Default is 1 for H+"
        },
    )
    # INTERNAL
    _runtype: Literal["protonate"] = "protonate"
    _output_filename: str = "protonated.xyz"

    def make_commands(self):
        """Make the CLI for the CREST input."""
        super().make_commands()
        self._commands.append(f"--{self._runtype}")
        if self.ion is not None:
            self._commands.append(f"--swel {self.ion}{self.ion_charge}+")
        self._commands.append("--newversion")

    def operation(
        self, structure: Molecule
    ) -> tuple[Molecule | list[Molecule], Optional[dict[str, Any]]]:
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
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> prot = CRESTProtonation(ewin=6.0, threads=4) # doctest: +SKIP
            >>> structures, properties = prot.operation(molecule) # doctest: +SKIP
        """
        structure.to("input.xyz", fmt="xyz")
        if self.charge is None and structure.charge is not None:
            self.charge = structure.charge
        super().make_dict()
        super().write_toml()
        self.make_commands()
        super().run()
        if not os.path.exists(self._output_filename):
            raise FileNotFoundError(
                "No tautomers found. Please check your CREST settings and log file."
            ) from None
        molecules = molbar_screening(self._output_filename, self.threads)
        for i in range(len(molecules)):
            molecules[i].set_charge_and_spin(
                charge=molecules[i].charge + 1,
                spin_multiplicity=int(molecules[i].charge + 1) // 2 + 2,
            )
        return molecules, None
