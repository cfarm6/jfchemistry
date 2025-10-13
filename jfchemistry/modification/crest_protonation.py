"""CREST-based protonation for generating protonated structures.

This module provides integration with CREST's automated protonation workflow
for generating low-energy protonated structures at different sites.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

import tomli_w
from pymatgen.core.structure import SiteCollection
from pymatgen.io.xyz import XYZ

from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTProtonation(StructureModification):
    """Generate protonated structures using CREST.

    Uses CREST's automated protonation workflow to identify basic sites
    and generate low-energy protonated structures. The method systematically
    explores different protonation sites and optimizes the resulting structures.

    Attributes:
        name: Name of the job (default: "CREST Protonation").
        runtype: Workflow type (default: "protonate").
        ion: Ion to add for protonation (default: None, uses H+).
        ion_charge: Charge of the ion (default: 1).
        ewin: Energy window in kcal/mol for selecting protonated structures
            (default: None, uses CREST default).
        ffopt: Perform force field pre-optimization (default: True).
        freezeopt: Freeze constraint string for optimization (default: None).
        finalopt: Perform final optimization (default: True).
        threads: Number of parallel threads (default: 1).

    References:
        - CREST Documentation: https://crest-lab.github.io/crest-docs/

    Examples:
        >>> from jfchemistry.modification import CRESTProtonation
        >>> from pymatgen.core import Molecule
        >>> from ase.build import molecule
        >>> molecule = Molecule.from_ase_atoms(molecule("CCH"))
        >>>
        >>> # Protonate an amine
        >>> prot = CRESTProtonation(ewin=6.0, threads=4)
        >>> job = prot.make(molecule)
        >>> protonated_structures = job.output["structure"]
        >>>
        >>> # Protonate with custom settings
        >>> prot_custom = CRESTProtonation(
        ...     ewin=8.0,
        ...     ffopt=True,
        ...     finalopt=True,
        ...     threads=8
        ... )
        >>> job = prot_custom.make(molecule)
        >>> protonated_structures = job.output["structure"]
    """

    name: str = "CREST Protonation"
    runtype: Literal["protonate"] = "protonate"
    ion: Optional[str] = None
    ion_charge: int = 1
    ewin: Optional[float] = None
    ffopt: bool = True
    freezeopt: Optional[str] = None
    finalopt: bool = True
    threads: int = 1

    def make_dict(self):
        """Create parameter dictionary for CREST configuration.

        Extracts relevant protonation parameters and packages them for
        the CREST TOML configuration file.

        Returns:
            Dictionary of non-None protonation parameters.

        Examples:
            >>> prot = CRESTProtonation(ewin=6.0, ffopt=True)
            >>> params = prot.make_dict()
            >>> print(params)
            {'ewin': 6.0, 'ffopt': True, 'finalopt': True}
        """
        keys = ["ion", "ewin", "ffopt", "freezeopt", "finalopt"]
        d = {}
        for k, v in vars(self).items():
            if v is None or k not in keys:
                continue
            d[k] = v
        return d

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
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> prot = CRESTProtonation(ewin=6.0, threads=4) # doctest: +SKIP
            >>> structures, properties = prot.operation(molecule) # doctest: +SKIP
        """
        structure.to("input.xyz", fmt="xyz")

        # Write the input file
        self.inputfile = "input.xyz"
        d = {"threads": self.threads, "runtype": self.runtype, "input": "input.xyz"}
        d["protonation"] = self.make_dict()
        with open("crest.toml", "wb") as f:
            tomli_w.dump(d, f)

        charge = structure.charge

        # Save current working directory
        original_dir = os.getcwd()

        # Create temporary directory and run crest there
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy input files to temp directory
            shutil.copy("input.xyz", tmp_dir)
            shutil.copy("crest.toml", tmp_dir)

            # Change to temp directory
            os.chdir(tmp_dir)

            # Run crest command
            subprocess.call(
                f"crest --input crest.toml --chrg {charge} --newversion > log.out",
                shell=True,
            )
            # Copy log.out back to original directory
            if os.path.exists("log.out"):
                shutil.copy("log.out", original_dir)

            # Copy all crest_conformers.* files back to original directory
            for file in glob.glob("protonated.xyz"):
                shutil.copy(file, original_dir)

            # Change back to original directory
            os.chdir(original_dir)

        structures = XYZ.from_file("protonated.xyz").all_molecules
        structures = cast("list[SiteCollection]", structures)
        return structures, None
