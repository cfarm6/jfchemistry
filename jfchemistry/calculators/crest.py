"""Base class for CREST Applications."""

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import tomli_w

from jfchemistry.calculators.base import Calculator


@dataclass
class CRESTCalculator(Calculator):
    """Base class for CREST Applications."""

    name: str = "CREST"
    executable: str = "crest"
    threads: int = 1  # TOML
    solvation: Optional[
        Union[
            tuple[
                Literal["alpb"],
                Literal[
                    "acetone",
                    "acetonitrile",
                    "aniline",
                    "benzaldehyde",
                    "benzene",
                    "ch2cl2",
                    "chcl3",
                    "cs2",
                    "dioxane",
                    "dmf",
                    "dmso",
                    "ether",
                    "ethylacetate",
                    "furane",
                    "hexandecane",
                    "hexane",
                    "methanol",
                    "nitromethane",
                    "octanol",
                    "woctanol",
                    "phenol",
                    "toluene",
                    "thf",
                    "water",
                ],
            ],
            tuple[
                Literal["gbsa"],
                Literal[
                    "acetone",
                    "acetonitrile",
                    "benzene",
                    "CH2Cl2",
                    "CHCl3",
                    "CS2",
                    "DMF",
                    "DMSO",
                    "ether",
                    "H2O",
                    "methanol",
                    "n-hexane",
                    "THF",
                    "toluene",
                ],
            ],
        ]
    ] = None  # CLI
    # CREGEN PARAMETERS
    # ------- TOML ------
    energy_window: Optional[float] = 6.0
    cartesian_rmsd_threshold: Optional[float] = 0.125
    energy_threshold: Optional[float] = 0.05
    rotational_rms_threshold: Optional[float] = 0.01
    preoptimization: Optional[bool] = True
    # ------ CLI ------
    z_matrix_sorting: bool = False

    # INTERNAL
    _input_dict: dict[str, Any] = field(default_factory=dict)
    _commands: list[str | int | float] = field(default_factory=list)
    _toml_filename: str = "crest.toml"
    _xyz_filename: str = "input.xyz"

    def make_dict(self):
        """Make the TOML dictionary for the CREST input."""
        self._input_dict["input"] = self._xyz_filename
        self._input_dict["threads"] = self.threads
        self._input_dict["preopt"] = self.preoptimization
        self._input_dict["cregen"] = {
            "ewin": self.energy_window,
            "rthr": self.cartesian_rmsd_threshold,
            "ethr": self.energy_threshold,
            "bthr": self.rotational_rms_threshold,
        }

    def write_toml(self):
        """Write the TOML file for the CREST input."""
        with open(self._toml_filename, "wb") as f:
            tomli_w.dump(self._input_dict, f)

    def make_commands(self):
        """Make the CLI for the CREST input."""
        self._commands.append(self.executable)
        self._commands.append("--input")
        self._commands.append(self._toml_filename)
        if self.charge is not None:
            self._commands.append("--chrg")
            self._commands.append(str(self.charge))
        if self.spin_multiplicity is not None:
            self._commands.append("--uhf")
            self._commands.append(str(self.spin_multiplicity))
        if self.solvation is not None:
            self._commands.append(f"--{self.solvation[0]}")
            self._commands.append(self.solvation[1])
        if self.z_matrix_sorting:
            self._commands.append("--zs")
        else:
            self._commands.append("--nosz")

    def run(self):
        """Run the CREST calculation."""
        # Save current working directory
        original_dir = os.getcwd()

        # Create temporary directory and run crest there
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy input files to temp directory
            shutil.copy(self._xyz_filename, tmp_dir)
            shutil.copy(self._toml_filename, tmp_dir)

            # Change to temp directory
            os.chdir(tmp_dir)

            # Run crest command
            subprocess.call(
                " ".join(str(cmd) for cmd in self._commands) + " > log.out",
                shell=True,
            )

            # Copy all files back to original directory
            for file in glob.glob("*"):
                shutil.copy(file, original_dir)

            # Change back to original directory
            os.chdir(original_dir)

            # Remove the temporary directory
            shutil.rmtree(tmp_dir)
