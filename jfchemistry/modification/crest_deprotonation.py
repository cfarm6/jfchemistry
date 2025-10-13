"""CREST-based deprotonation for generating deprotonated structures.

This module provides integration with CREST's automated deprotonation workflow
for generating low-energy deprotonated structures and tautomers.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, cast

import tomli_w
from pymatgen.core.structure import SiteCollection
from pymatgen.io.xyz import XYZ

from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTDeprotonation(StructureModification):
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
    ewin: Optional[float] = None

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
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
        structure.to("input.sdf", fmt="sdf")

        d = {"calculation": {"level": {"method": "gfn2", "rdwbo": True}}}
        with open("crest.toml", "wb") as f:
            tomli_w.dump(d, f)
        commands = ["crest", "input.sdf", "--deprotonate", "--input", "crest.toml"]
        if self.ewin is not None:
            commands.append(f"--ewin {self.ewin}")
        charge = structure.charge
        commands.append(f"--chrg {charge} --newversion")
        commands.append(" > log.out")

        # Save current working directory
        original_dir = os.getcwd()

        # Create temporary directory and run crest there
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy input files to temp directory
            shutil.copy("input.sdf", tmp_dir)
            shutil.copy("crest.toml", tmp_dir)

            # Change to temp directory
            os.chdir(tmp_dir)

            # Run crest command
            subprocess.call(" ".join(commands), shell=True)
            # Copy log.out back to original directory
            if os.path.exists("log.out"):
                shutil.copy("log.out", original_dir)

            # Copy all crest_conformers.* files back to original directory
            for file in glob.glob("deprotonated.xyz"):
                shutil.copy(file, original_dir)

            # Change back to original directory
            os.chdir(original_dir)

        structures = XYZ.from_file("deprotonated.xyz").all_molecules
        structures = cast("list[SiteCollection]", structures)
        return structures, None
