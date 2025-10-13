"""CREST-based conformer generation using metadynamics.

This module provides integration with CREST (Conformer-Rotamer Ensemble
Sampling Tool) for comprehensive conformational searching using metadynamics
and GFN-xTB methods.
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

from jfchemistry.conformers.base import ConformerGeneration


@dataclass
class CRESTConformers(ConformerGeneration):
    """CREST conformer generation using metadynamics sampling.

    CREST (Conformer-Rotamer Ensemble Sampling Tool) performs automated
    conformational and rotameric searches using metadynamics simulations
    with GFN-xTB tight-binding methods. It efficiently explores conformational
    space to identify unique low-energy conformers.

    The implementation supports various metadynamics protocols and provides
    extensive control over optimization settings, energy calculations, and
    conformer filtering.

    Attributes:
        name: Name of the job (default: "CREST Conformer Generation").
        runtype: Metadynamics protocol to use:
            - "imtd-gc": Iterative metadynamics with genetic crossing (default)
            - "nci-mtd": Non-covalent interaction metadynamics
            - "imtd-smtd": Iterative metadynamics with static metadynamics
        preopt: Pre-optimize structure before conformer search (default: True).
        multilevelopt: Use multi-level optimization (default: True).
        topo: Enable topology-based filtering (default: True).
        parallel: Number of parallel threads (default: 1).
        opt_engine: Optimization algorithm:
            - "ancopt": Approximate normal coordinate optimizer (default)
            - "rfo": Rational function optimizer
            - "gd": Gradient descent
        hess_update: Hessian update method:
            - "bfgs": BFGS update (default)
            - "powell": Powell update
            - "sd1": Steepest descent
            - "bofill": Bofill update
            - "schlegel": Schlegel update
        maxcycle: Maximum optimization cycles (default: None, auto).
        optlev: Optimization convergence level:
            - "crude", "vloose", "loose", "normal" (default), "tight", "vtight", "extreme"
        converge_e: Energy convergence threshold (default: None, auto).
        converge_g: Gradient convergence threshold (default: None, auto).
        freeze: Freeze constraints string (default: None).
        ewin: Energy window for conformer selection in kcal/mol (default: 6.0).
        ethr: Energy threshold for duplicate detection in kcal/mol (default: 0.05).
        rthr: RMSD threshold for structural similarity in Angstrom (default: 0.125).
        bthr: Rotational constant threshold for duplicate detection (default: 0.01).
        calculation_energy_method: Method for energy calculations:
            - "gfn2" (default), "gfn1", "gfn0", "gfnff"
        calculation_energy_calcspace: Calculation space setting (default: None).
        calculation_energy_chrg: Charge for energy calculations (default: None, from structure).
        calculation_energy_uhf: Unpaired electrons for energy calc (default: None).
        calculation_energy_rdwbo: Read Wiberg bond orders (default: False).
        calculation_energy_rddip: Read dipole moments (default: False).
        calculation_energy_dipgrad: Compute dipole gradients (default: False).
        calculation_energy_gradfile: External gradient file (default: None).
        calculation_energy_gradtype: Gradient file type (default: None).
        calculation_dynamics_method: Method for metadynamics:
            - "gfn2" (default), "gfn1", "gfn0", "gfnff"
        calculation_dynamics_calcspace: Calculation space for dynamics (default: None).
        calculation_dynamics_chrg: Charge for dynamics (default: None, from structure).
        calculation_dynamics_uhf: Unpaired electrons for dynamics (default: None).
        calculation_dynamics_rdwbo: Read Wiberg bond orders in dynamics (default: False).
        calculation_dynamics_rddip: Read dipole in dynamics (default: False).
        calculation_dynamics_dipgrad: Compute dipole gradients in dynamics (default: False).
        calculation_dynamics_gradfile: External gradient file for dynamics (default: None).
        calculation_dynamics_gradtype: Gradient file type for dynamics (default: None).
        dynamics_dump: Dynamics trajectory dump frequency (default: 100.0).

    References:
        - CREST Documentation: https://crest-lab.github.io/crest-docs/
        - Pracht et al., PCCP 2020, 22, 7169-7192

    Examples:
        >>> from pymatgen.core import Molecule
        >>> from ase.build import molecule
        >>> from jfchemistry.conformers import CRESTConformers
        >>>
        >>> mol = Molecule.from_ase_atoms(molecule("C2H6"))
        >>> mol = mol.set_charge_and_spin(0, 1)
        >>> # Basic conformer search
        >>> conf_gen = CRESTConformers(
        ...     ewin=6.0,
        ...     calculation_energy_method="gfnff",
        ...     calculation_dynamics_method="gfnff"
        ... )
        >>> structures, properties = conf_gen.operation(mol)
        >>> type(structures[0])
        <class 'pymatgen.core.structure.Molecule'>
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

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Generate conformers using CREST metadynamics search.

        Performs a conformational search using CREST with the configured
        metadynamics protocol and GFN-xTB method. The calculation runs in
        a temporary directory and returns the unique low-energy conformers.

        Args:
            structure: Input molecular structure with 3D coordinates. The
                structure's charge property is used if calculation charges
                are not explicitly set.

        Returns:
            Tuple containing:
                - List of conformer structures sorted by energy
                - None (no additional properties returned)

        Examples:
            >>> from jfchemistry.conformers import CRESTConformers
            >>> from pymatgen.core import Molecule
            >>> from ase.build import molecule
            >>> mol = Molecule.from_ase_atoms(molecule("C2H6"))
            >>> mol = mol.set_charge_and_spin(0, 1)
            >>> gen = CRESTConformers(ewin=6.0, parallel=4)
            >>> conformers, props = gen.operation(mol)
            >>> len(conformers)
            1
        """
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

        conformers = cast(
            "list[SiteCollection]", XYZ.from_file("crest_conformers.xyz").all_molecules
        )
        return conformers, {}
