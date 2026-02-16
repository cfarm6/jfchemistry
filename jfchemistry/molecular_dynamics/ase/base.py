"""ASE-based molecular dynamics framework.

This module provides the base framework for molecular dynamics using
ASE (Atomic Simulation Environment) MD integrators with various calculators.
"""

import glob
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from ase import Atoms
from ase.io import Trajectory
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass, SystemProperty
from jfchemistry.molecular_dynamics.base import MolecularDynamics, MolecularDynamicsOutput


class ASEMDSystemProperties(PropertyClass):
    """System properties for ASE Molecular Dynamics."""

    potential_energy: Optional[SystemProperty] = None
    kinetic_energy: Optional[SystemProperty] = None
    temperature: Optional[SystemProperty] = None
    pressure: Optional[SystemProperty] = None
    volume: Optional[SystemProperty] = None


class ASEMDProperties(Properties):
    """Properties for ASE Molecular Dynamics."""

    system: ASEMDSystemProperties


@dataclass
class ASEMolecularDynamics[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    PymatGenMaker[InputType, OutputType], MolecularDynamics
):
    """Base class for molecular dynamics simulations using ASE calculators.

    Combines molecular dynamics simulations with ASE calculator interfaces.
    This class provides the framework for running MD simulations
    of a structure using various ASE calculators (neural networks, machine learning,
    semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Molecular Dynamics").
        calculator: The calculator to use for the calculation.
        integrator: The integrator to use for the simulation.
        duration: The duration of the simulation in fs.
        timestep: The timestep of the simulation in fs.
        temperature: The temperature of the simulation in K.
        logfile: The filename prefix to log the trajectory of the simulation.
        progress_bar: Whether to show a progress bar in the simulation.
        log_interval: The interval at which to log the simulation in fs.
        log_potential_energy: Whether to log the potential energy in the simulation.
        log_kinetic_energy: Whether to log the kinetic energy in the simulation.
        log_temperature: Whether to log the temperature in the simulation.
        log_pressure: Whether to log the pressure in the simulation.
        log_volume: Whether to log the volume in the simulation.
        log_trajectory: Whether to log the trajectory in the simulation.
    """

    name: str = "ASE Molecular Dynamics"
    calculator: ASECalculator = field(
        default_factory=lambda: ASECalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )
    integrator: str = field(
        default="nve", metadata={"description": "The integrator to use for the simulation"}
    )
    duration: float = field(
        default=1.0, metadata={"description": "The duration of the simulation in fs"}
    )
    timestep: float = field(
        default=0.5, metadata={"description": "The timestep of the simulation in fs"}
    )
    temperature: float = field(
        default=300.0, metadata={"description": "The temperature of the simulation in K"}
    )
    logfile: str = field(
        default="md",
        metadata={
            "description": "The filename prefix to log the trajectory of the simulation.\
                 The filename will be appended with the system index and the extension .traj"
        },
    )
    progress_bar: bool = field(
        default=True,
        metadata={"description": "Whether to show a progress bar in the simulation"},
    )
    log_interval: float = field(
        default=1.0, metadata={"description": "The interval at which to log the simulation in fs"}
    )
    log_potential_energy: bool = field(
        default=False,
        metadata={"description": "Whether to log the potential energy in the simulation"},
    )
    log_kinetic_energy: bool = field(
        default=False,
        metadata={"description": "Whether to log the kinetic energy in the simulation"},
    )
    log_temperature: bool = field(
        default=False,
        metadata={"description": "Whether to log the temperature in the simulation"},
    )
    log_pressure: bool = field(
        default=False,
        metadata={"description": "Whether to log the pressure in the simulation"},
    )
    log_volume: bool = field(
        default=False,
        metadata={"description": "Whether to log the volume in the simulation"},
    )
    log_trajectory: bool = field(
        default=False,
        metadata={"description": "Whether to log the trajectory in the simulation"},
    )

    _output_model: type[MolecularDynamicsOutput] = MolecularDynamicsOutput
    _properties_model: type[ASEMDProperties] = ASEMDProperties

    def __post_init__(self):
        """Post initialization hook."""
        super().__post_init__()

    def _create_dynamics(self, atoms: Atoms) -> Any:
        """Create the dynamics object for the simulation.

        This method must be implemented by subclasses to create the specific
        ASE dynamics integrator.

        Args:
            atoms: ASE Atoms object with calculator attached.

        Returns:
            ASE dynamics object.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def _parse_trajectory(self) -> list[list[Atoms]]:
        """Parse the trajectory from the simulation."""
        trajfiles = glob.glob(f"{self.logfile}*.traj")
        if len(trajfiles) == 0:
            raise FileNotFoundError("No trajectory files found in the current directory")
        trajectories: list[list[Atoms]] = []
        # Trajectory files are named like <self.logfile>_<system_index>.traj - sort by system index
        trajfiles.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else 0)
        for trajfile in trajfiles:
            trajectory_atoms = []
            traj = Trajectory(trajfile, "r")
            for atoms in traj:
                trajectory_atoms.append(atoms.copy())
            traj.close()
            trajectories.append(trajectory_atoms)
        return trajectories

    def _make_logger(self, atoms: Atoms, logfile: str) -> Optional[MDLogger]:
        """Make a logger for the simulation."""
        if not any(
            [
                self.log_potential_energy,
                self.log_kinetic_energy,
                self.log_temperature,
                self.log_pressure,
                self.log_volume,
            ]
        ):
            return None

        # Determine what to log
        peratom = False
        header = True
        mode = "a"

        return MDLogger(
            dyn=None,
            atoms=atoms,
            logfile=logfile,
            header=header,
            stress=False,  # We'll handle pressure separately if needed
            peratom=peratom,
            mode=mode,
        )

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Run molecular dynamics simulation using ASE.

        Performs molecular dynamics simulation by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Creating the dynamics integrator
        4. Running the simulation
        5. Converting back to Pymatgen Molecule
        6. Extracting properties from the calculation

        Args:
            input: Input molecular structure with 3D coordinates.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Final Pymatgen Molecule/Structure
                - Dictionary of computed properties from simulation
        """
        final_structures = []

        n_steps = int(self.duration // self.timestep)
        log_interval = int(self.log_interval // self.timestep)

        atoms = input.to_ase_atoms()
        charge = int(input.charge)
        if isinstance(input, Molecule):
            spin_multiplicity = int(input.spin_multiplicity)
        else:
            spin_multiplicity = 1
        # Set up calculator
        self.calculator._set_calculator(atoms, charge=charge, spin_multiplicity=spin_multiplicity)
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        # Set up trajectory and logging
        traj_file = None
        if self.log_trajectory:
            traj_file = f"{self.logfile}.traj"
            traj = Trajectory(traj_file, "w", atoms)
        else:
            traj = None
        log_file = f"{self.logfile}.log"
        logger = self._make_logger(atoms, log_file)
        # Create dynamics object
        dyn = self._create_dynamics(atoms)
        # Attach trajectory and logger
        if traj is not None:
            dyn.attach(traj.write, interval=log_interval)
        if logger is not None:
            dyn.attach(logger, interval=log_interval)
        # Run simulation
        if self.progress_bar:
            from tqdm import tqdm

            pbar = tqdm(total=n_steps, desc="Running MD simulation")
            step_count = [0]  # Use list to allow modification in closure

            def update_pbar():
                step_count[0] += 1
                pbar.update(1)

            dyn.attach(update_pbar, interval=1)
            dyn.run(n_steps)
            pbar.close()
        else:
            dyn.run(n_steps)
        # Close trajectory
        if traj is not None:
            traj.close()
        # Convert back to pymatgen
        final_structures.append(cast("OutputType", type(input).from_ase_atoms(atoms)))

        # Parse properties
        # properties_list = self._parse_properties()
        # if len(properties_list) == 0:
        #     # Create empty properties if none were logged
        #     properties_list = [
        #         ASEMDProperties(system=ASEMDSystemProperties())
        #     ]

        # Return as list to match return type annotation
        return cast("list[OutputType]", final_structures), cast(
            "list[Properties]",
            [ASEMDProperties(system=ASEMDSystemProperties())] * len(final_structures),
        )
