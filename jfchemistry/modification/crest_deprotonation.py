"""Structure deprotonation using CREST."""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, cast

import tomli_w
from pymatgen.core.structure import IMolecule
from pymatgen.io.xyz import XYZ

from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTDeprotonation(StructureModification):
    """Structure deprotonation using CREST.

    Parameters
    ----------
    - ewin: The energy window. Keep structures within this energy window.

    References
    ----------
    - https://crest-lab.github.io/crest-docs/
    """

    name: str = "CREST Deprotonation"
    ewin: Optional[float] = None

    def modify_structure(
        self, structure: IMolecule
    ) -> tuple[Optional[list[IMolecule] | IMolecule], Optional[dict[str, Any]]]:
        """Modify the structure."""
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
        structures = cast("list[IMolecule]", structures)
        return structures, None
