"""CREST-based conformer generation using metadynamics.

This module provides integration with CREST (Conformer-Rotamer Ensemble
Sampling Tool) for comprehensive conformational searching using metadynamics
and GFN-xTB methods.
"""

import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, cast

import tomli_w
from pydantic import BaseModel
from pymatgen.core.structure import SiteCollection
from pymatgen.io.xyz import XYZ

from jfchemistry.conformers.base import ConformerGeneration


class CRESTProperties(BaseModel):
    """Properties of the CREST conformer generation."""


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
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from jfchemistry.conformers import CRESTConformers # doctest: +SKIP
        >>> mol = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> mol = mol.set_charge_and_spin(0, 1) # doctest: +SKIP
        >>> # Basic conformer search
        >>> conf_gen = CRESTConformers( # doctest: +SKIP
        ...     ewin=6.0, # doctest: +SKIP
        ...     calculation_energy_method="gfnff", # doctest: +SKIP
        ...     calculation_dynamics_method="gfnff" # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> structures, properties = conf_gen.operation(mol) # doctest: +SKIP
    """

    name: str = "CREST Conformer Generation"
    executable: str = "crest"
    # Command line options
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
    ] = None
    charge: Optional[int] = None
    spin_multiplicity: Optional[int] = None
    # General Settings Block
    threads: int = 1
    runtype: Literal["imtd-gc", "nci-mtd", "imtd-smtd"] = "imtd-gc"
    preopt: bool = True
    topo: bool = True

    # Calculation Main Block
    opt_engine: Literal["ancopt", "rfo", "gd"] = "ancopt"
    hess_update: Literal["bfgs", "powell", "sd1", "bofill", "schlegel"] = "bfgs"
    optlev: Literal["crude", "vloose", "loose", "normal", "tight", "vtight", "extreme"] = "normal"

    # Optimization Calculation Block
    calculation_energy_method: Literal[
        "gfn2",
        "gfn1",
        "gfn0",
        "gfnff",
    ] = "gfn2"

    # Metadynamics Calculation Block
    calculation_dynamics_method: Literal[
        "gfn2",
        "gfn1",
        "gfn0",
        "gfnff",
    ] = "gfn2"

    # Dynamics Block
    dynamics_dump: float = 100.0
    dynamics_dump_frequency: Optional[float] = 100

    # CREGEN Block
    ewin: float = 6.0
    ethr: float = 0.05
    rthr: float = 0.125
    bthr: float = 0.01

    # INTERNAL
    _input_dict: dict[str, Any] = field(default_factory=dict)
    _commands: list[str | int | float] = field(default_factory=list)
    _toml_filename: str = "crest.toml"
    _xyz_filename: str = "input.xyz"
    _properties_model: type[CRESTProperties] = CRESTProperties

    def make_dict(self):
        """Make the dictionary for the CREST input."""
        self._input_dict["threads"] = self.threads
        self._input_dict["runtype"] = self.runtype
        self._input_dict["preopt"] = self.preopt
        self._input_dict["topo"] = self.topo
        self._input_dict["input"] = self._xyz_filename
        self._input_dict["calculation"] = {
            "opt_engine": self.opt_engine,
            "hess_update": self.hess_update,
            "optlev": self.optlev,
            "level": [
                {
                    "method": self.calculation_energy_method,
                },
                {
                    "method": self.calculation_dynamics_method,
                },
            ],
        }

        self._input_dict["cregen"] = {
            "ewin": self.ewin,
            "ethr": self.ethr,
            "rthr": self.rthr,
            "bthr": self.bthr,
        }

        self._input_dict["dynamics"] = {
            "active": [2],
            "dump": self.dynamics_dump_frequency,
        }

    def write_toml(self):
        """Write the TOML file for the CREST input."""
        # Create a copy of the input nested dictionary without an key-value pairs with None values
        with open(self._toml_filename, "wb") as f:
            tomli_w.dump(self._input_dict, f)

    def make_commands(self):
        """Make the CLI for the CREST input."""
        self._commands.append(self.executable)
        self._commands.append("--input")
        self._commands.append(self._toml_filename)
        if self.solvation is not None:
            self._commands.append(f"--{self.solvation[0]}")
            self._commands.append(self.solvation[1])
        if self.charge is not None:
            self._commands.append("--chrg")
            self._commands.append(str(self.charge))
        if self.spin_multiplicity is not None:
            self._commands.append("--uhf")
            self._commands.append(str(self.spin_multiplicity))

    def run(self):
        """Run the CREST calculation."""
        # Save current working directory
        # original_dir = os.getcwd()
        subprocess.call(
            args=" ".join(str(x) for x in self._commands) + " > log.out",
            shell=True,
        )

        # # Create temporary directory and run crest there
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     # Copy input files to temp directory
        #     shutil.copy(self._xyz_filename, tmp_dir)
        #     shutil.copy(self._toml_filename, tmp_dir)

        #     # Change to temp directory
        #     os.chdir(tmp_dir)

        #     # Run crest command

        #     # Copy all files back to original directory
        #     for file in glob.glob("*"):
        #         shutil.copy(file, original_dir)

        #     # Change back to original directory
        #     os.chdir(original_dir)

        #     # Remove the temporary directory
        #     shutil.rmtree(tmp_dir)

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], None]:
        """Generate conformers using CREST metadynamics search.

        Performs a conformational search using CREST with the configured
        metadynamics protocol and GFN-xTB method. The calculation runs in
        a temporary directory and returns the unique low-energy conformers.

        Args:
            structure: Input molecular structure with 3D coordinates. The
         1       structure's charge property is used if calculation charges
                are not explicitly set.

        Returns:
            Tuple containing:
                - List of conformer structures sorted by energy
                - None (no additional properties returned)

        Examples:
            >>> from jfchemistry.conformers import CRESTConformers # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from ase.build import molecule # doctest: +SKIP
            >>> mol = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> mol = mol.set_charge_and_spin(0, 1) # doctest: +SKIP
            >>> gen = CRESTConformers(ewin=6.0, parallel=4) # doctest: +SKIP
            >>> conformers, props = gen.operation(mol) # doctest: +SKIP
        """
        # Write structures to sdf file
        structure.to(self._xyz_filename, fmt="xyz")

        self.input = self._xyz_filename

        self.make_dict()
        self.write_toml()
        self.make_commands()
        self.run()

        conformers = cast(
            "list[SiteCollection]", XYZ.from_file("crest_conformers.xyz").all_molecules
        )

        return (conformers, None)
