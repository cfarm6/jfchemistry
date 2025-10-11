"""Geometry Optimization using AimNet2."""

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

from jfchemistry.conformers.base import ConformerGeneration


@dataclass
class CRESTConformers(ConformerGeneration):
    """CREST Conformer Generation.

    Parameters
    ----------
    - runtype: The type of run to perform.
    - preopt: Whether to preoptimize the structure.
    - multilevelopt: Whether to use multiple optimization engines.
    - topo: Whether to use topology optimization.
    - parallel: The number of parallel threads to use.
    - opt_engine: The optimization engine to use.
    - hess_update: The hessian update method to use.
    - maxcycle: The maximum number of cycles to run.
    - optlev: The optimization level to use.
    - converge_e: The convergence energy.
    - converge_g: The convergence gradient.
    - freeze: The freeze string.
    - ewin: The energy window.
    - ethr: The energy threshold.
    - rthr: The root mean square gradient threshold.
    - bthr: The bond threshold.
    - calculation_energy_method: The energy calculation method to use.
    - calculation_energy_calcspace: The energy calculation calcspace.
    - calculation_energy_chrg: The energy calculation charge.
    - calculation_energy_uhf: The energy calculation UHF.
    - calculation_energy_rdwbo: The energy calculation RDWBO.
    - calculation_energy_rddip: The energy calculation RDDIP.
    - calculation_energy_dipgrad: The energy calculation DIPGRAD.
    - calculation_energy_gradfile: The energy calculation gradfile.
    - calculation_energy_gradtype: The energy calculation gradtype.
    - calculation_dynamics_method: The dynamics calculation method to use.
    - calculation_dynamics_calcspace: The dynamics calculation calcspace.
    - calculation_dynamics_chrg: The dynamics calculation charge.
    - calculation_dynamics_uhf: The dynamics calculation UHF.
    - calculation_dynamics_rdwbo: The dynamics calculation RDWBO.
    - calculation_dynamics_rddip: The dynamics calculation RDDIP.
    - calculation_dynamics_dipgrad: The dynamics calculation DIPGRAD.
    - calculation_dynamics_gradfile: The dynamics calculation gradfile.
    - calculation_dynamics_gradtype: The dynamics calculation gradtype.
    - dynamics_dump: The dynamics dump.

    References
    ----------
    - https://crest-lab.github.io/crest-docs/
    """

    name: str = "CREST Conformer Generation"
    runtype: Literal["imtd-gc", "nci-mtd", "imtd-smtd"] = "imtd-gc"
    preopt: bool = True
    multilevelopt: bool = True
    topo: bool = True
    parallel: int = 1
    opt_engine: Literal["ancopt", "rfo", "gd"] = "ancopt"
    hess_update: Literal["bfgs", "powell", "sd1", "bofill", "schlegel"] = "bfgs"
    maxcycle: Optional[int] = None
    optlev: Literal["crude", "vloose", "loose", "normal", "tight", "vtight", "extreme"] = "normal"
    converge_e: Optional[float] = None
    converge_g: Optional[float] = None
    freeze: Optional[str] = None
    # CREGEN Block
    ewin: float = 6.0
    ethr: float = 0.05
    rthr: float = 0.125
    bthr: float = 0.01
    # Optimization Calculation Block
    calculation_energy_method: Literal[
        "gfn2",
        "gfn1",
        "gfn0",
        "gfnff",
    ] = "gfn2"
    calculation_energy_calcspace: Optional[str] = None
    calculation_energy_chrg: Optional[int] = None
    calculation_energy_uhf: Optional[int] = None
    calculation_energy_rdwbo: bool = False
    calculation_energy_rddip: bool = False
    calculation_energy_dipgrad: bool = False
    calculation_energy_gradfile: Optional[str] = None
    calculation_energy_gradtype: Optional[Literal["engrad"]] = None

    # Metadynamics Block
    calculation_dynamics_method: Literal[
        "gfn2",
        "gfn1",
        "gfn0",
        "gfnff",
    ] = "gfn2"
    calculation_dynamics_calcspace: Optional[str] = None
    calculation_dynamics_chrg: Optional[int] = None
    calculation_dynamics_uhf: Optional[int] = None
    calculation_dynamics_rdwbo: bool = False
    calculation_dynamics_rddip: bool = False
    calculation_dynamics_dipgrad: bool = False
    calculation_dynamics_gradfile: Optional[str] = None
    calculation_dynamics_gradtype: Optional[Literal["engrad"]] = None
    dynamics_dump: float = 100.0

    def generate_conformers(
        self, structure: IMolecule
    ) -> tuple[Optional[list[IMolecule]], Optional[dict[str, Any]]]:
        """Generate conformers using CREST."""
        # Write structures to sdf file
        structure.to("input.sdf", fmt="sdf")

        if self.calculation_energy_chrg is None:
            self.calculation_energy_chrg = structure.charge
        if self.calculation_dynamics_chrg is None:
            self.calculation_dynamics_chrg = structure.charge
        self.input = "input.sdf"
        # Empty strucutres
        d = {"calculation": {}, "cregen": {}, "dynamics": {"active": [2]}}
        calculation_blocks = {"energy": {}, "dynamics": {}}
        # Fill in dictionaries for toml
        for k, v in vars(self).items():
            if v is None:
                continue
            if k == "name":
                continue
            if k.split("_")[0] == "calculation":
                if k.split("_")[1] == "energy" or k.split("_")[1] == "dynamics":
                    calculation_blocks[k.split("_")[1]][k.split("_")[2]] = v
            elif k.split("_")[0] == "dynamics":
                d["dynamics"][k.split("_")[1]] = v
            elif k in ["ewin", "ethr", "rthr", "bthr"]:
                d["cregen"][k] = v
            elif k in [
                "type",
                "elog",
                "eprint",
                "opt_engine",
                "hess_update",
                "maxcycle",
                "optlev",
                "converge_e",
                "converge_g",
                "freeze",
            ]:
                d["calculation"][k] = v
            else:
                d[k] = v
        d["calculation"]["level"] = [
            calculation_blocks["energy"],
            calculation_blocks["dynamics"],
        ]
        # Check for charges
        if self.calculation_energy_chrg is None or self.calculation_dynamics_chrg is None:
            self.calculation_energy_chrg = self.calculation_dynamics_chrg = structure.charge

        with open("crest.toml", "wb") as f:
            tomli_w.dump(d, f)

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
            subprocess.call("crest --input crest.toml --noreftopo --mquick > log.out", shell=True)

            # Copy log.out back to original directory
            if os.path.exists("log.out"):
                shutil.copy("log.out", original_dir)

            # Copy all crest_conformers.* files back to original directory
            for file in glob.glob("crest_conformers.xyz"):
                shutil.copy(file, original_dir)

            # Change back to original directory
            os.chdir(original_dir)

        conformers = cast("list[IMolecule]", XYZ.from_file("crest_conformers.xyz").all_molecules)
        return conformers, None
