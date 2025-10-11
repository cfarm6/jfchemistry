"""Structure protonation using CREST."""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

import tomli_w
from pymatgen.core.structure import IMolecule
from pymatgen.io.xyz import XYZ

from jfchemistry.modification.base import StructureModification


@dataclass
class CRESTProtonation(StructureModification):
    """Structure protonation using CREST.

    Parameters
    ----------
    - ewin: The energy window. Keep structures within this energy window.

    References
    ----------
    - https://crest-lab.github.io/crest-docs/
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
        """Make a dictionary of the parameters."""
        keys = ["ion", "ewin", "ffopt", "freezeopt", "finalopt"]
        d = {}
        for k, v in vars(self).items():
            if v is None or k not in keys:
                continue
            d[k] = v
        return d

    def modify_structure(
        self, structure: IMolecule
    ) -> tuple[Optional[list[IMolecule] | IMolecule], Optional[dict[str, Any]]]:
        """Modify the structure."""
        structure.to("input.sdf", fmt="sdf")

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
            shutil.copy("input.sdf", tmp_dir)
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

        structures = XYZ.from_file("deprotonated.xyz").all_molecules
        structures = cast("list[IMolecule]", structures)
        return structures, None
