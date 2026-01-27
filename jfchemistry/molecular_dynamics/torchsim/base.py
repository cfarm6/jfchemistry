"""TorchSim-based molecular dynamics framework.

This module provides the base framework for molecular dynamics using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

import glob
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, cast

import torch
import torch_sim as ts
from ase.atoms import Atoms
from pymatgen.core.structure import Molecule, Structure
from torch_sim.models.interface import ModelInterface
from torch_sim.units import UnitConversion as Uc

from jfchemistry import ureg
from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core import PymatgenBaseMaker
from jfchemistry.core.properties import Properties, PropertyClass, SystemProperty
from jfchemistry.molecular_dynamics.base import MolecularDynamics, MolecularDynamicsOutput

UNITS = ts.units.UnitSystem.metal


class TSMDSystemProperties(PropertyClass):
    """System properties for TorchSim Molecular Dynamics."""

    potential_energy: Optional[SystemProperty] = None
    kinetic_energy: Optional[SystemProperty] = None
    temperature: Optional[SystemProperty] = None
    pressure: Optional[SystemProperty] = None
    volume: Optional[SystemProperty] = None


class TSMDProperties(Properties):
    """Properties for TorchSim Molecular Dynamics."""

    system: TSMDSystemProperties


@dataclass
class TorchSimMolecularDynamics[InputType: Structure | Molecule, OutputType: Structure | Molecule](
    PymatgenBaseMaker[InputType, OutputType], MolecularDynamics
):
    """Base class for single point energy calculations using TorchSim calculators.

    Combines single point energy calculations with TorchSim calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various TorchSim calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "TorchSim Molecular Dynamics").
        calculator: The calculator to use for the calculation.
        integrator: The integrator to use for the simulation.
        duration: The duration of the simulation in fs.
        timestep: The timestep of the simulation in fs.
        temperature: The temperature of the simulation in K.
        autobatcher: Whether to enable autobatching.
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

    name: str = "TorchSim Molecular Dynamics"
    calculator: TorchSimCalculator = field(
        default_factory=lambda: TorchSimCalculator,
        metadata={"description": "the calculator to use for the calculation"},
    )
    integrator: Literal[
        "nve", "nvt_nose_hoover", "nvt_langevin", "npt_langevin", "npt_nose_hoove"
    ] = field(default="nve", metadata={"description": "The integrator to use for the simulation"})
    duration: float = field(
        default=1.0, metadata={"description": "The duration of the simulation in fs"}
    )
    timestep: float = field(
        default=0.5, metadata={"description": "The timestep of the simulation in fs"}
    )
    temperature: float = field(
        default=300.0, metadata={"description": "The temperature of the simulation in K"}
    )
    autobatcher: bool = field(default=False, metadata={"description": "Enable autobatching"})
    logfile: str = field(
        default="trajectory",
        metadata={
            "description": "The filename prefix to log the trajectory of the simulation.\
                 The filename will be appended with the system index and the extension .h5"
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
    _properties_model: type[TSMDProperties] = TSMDProperties

    def __post_init__(self):
        """Post initialization hook."""
        self.init_kwargs = {}
        self.step_kwargs = {}
        self._make_output_model(self._properties_model)

    def _setup_dicts(self, model: ModelInterface):
        """Setup the dictionaries for the integrator."""
        raise NotImplementedError("This method is not implemented for TorchSimMolecularDynamics")

    def _parse_properties(self) -> list[TSMDProperties]:
        """Parse the properties from the simulation."""
        logfiles = glob.glob("*.h5")
        if len(logfiles) == 0:
            raise FileNotFoundError("No log files found in the current directory")
        property_objects: list[TSMDProperties] = []
        # Log files are named like <self.logfile>_<system_index>.h5 - sort by system index
        logfiles.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for logfile in logfiles:
            properties: dict[str, SystemProperty] = {}
            with ts.TorchSimTrajectory(logfile) as trajectory:
                if len(trajectory.array_registry.keys()) == 0:
                    continue
                if self.log_potential_energy:
                    properties["potential_energy"] = SystemProperty(
                        name="potential_energy",
                        value=trajectory.get_array("potential_energy").flatten().tolist() * ureg.eV,
                    )
                if self.log_kinetic_energy:
                    properties["kinetic_energy"] = SystemProperty(
                        name="kinetic_energy",
                        value=trajectory.get_array("kinetic_energy").flatten().tolist() * ureg.eV,
                    )
                if self.log_temperature:
                    properties["temperature"] = SystemProperty(
                        name="temperature",
                        value=trajectory.get_array("temperature").flatten().tolist() * ureg.K,
                    )
                if self.log_pressure:
                    properties["pressure"] = SystemProperty(
                        name="pressure",
                        value=trajectory.get_array("pressure")
                        * Uc.bar_to_pa
                        / Uc.atm_to_pa
                        * ureg.atm,
                    )
                if self.log_volume:
                    properties["volume"] = SystemProperty(
                        name="volume",
                        value=trajectory.get_array("volume").flatten().tolist() * ureg.angstrom**3,
                    )
            property = TSMDProperties(system=TSMDSystemProperties(**properties))
            property_objects.append(property)
        return property_objects

    def _parse_trajectory(self) -> list[list[Atoms]]:
        """Parse the trajectory from the simulation."""
        logfiles = glob.glob("*.h5")
        if len(logfiles) == 0:
            raise FileNotFoundError("No log files found in the current directory")
        trajectories: list[list[Atoms]] = []
        # Log files are named like <self.logfile>_<system_index>.h5 - sort by system index
        logfiles.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for logfile in logfiles:
            trajectory_atoms = []
            with ts.TorchSimTrajectory(logfile) as trajectory:
                for i in range(len(trajectory)):
                    trajectory.get_atoms(i)
        trajectories.append(trajectory_atoms)

        return trajectories

    def _make_reporter(self) -> dict[str, Callable[[ts.SimState], Any]]:
        """Make a reporter for the simulation."""
        log_dict: dict[str, Callable[[ts.SimState], Any]] = {}

        if self.log_potential_energy:
            log_dict["potential_energy"] = lambda state: state.energy
        if self.log_kinetic_energy:
            log_dict["kinetic_energy"] = lambda state: ts.calc_kinetic_energy(
                momenta=state.momenta, masses=state.masses, system_idx=state.system_idx
            )
        if self.log_temperature:
            log_dict["temperature"] = lambda state: ts.quantities.calc_temperature(
                masses=state.masses,
                velocities=state.velocities,
                system_idx=state.system_idx,
            )
        if self.log_pressure:
            log_dict["pressure"] = lambda state: ts.quantities.get_pressure(
                state.stress,
                ts.calc_kinetic_energy(
                    masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
                ),
                torch.det(state.cell),
            )
        if self.log_volume:
            log_dict["volume"] = lambda state: torch.det(state.cell) * UNITS.distance**3
        return log_dict

    def _operation(
        self, structure: InputType
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Optimize molecular structure using ASE.

        Performs molecular dynamics simulation by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator
        """
        if not isinstance(structure, list):
            structure_list: list[InputType] = [structure]
        else:
            structure_list = structure
        model = self.calculator._get_model()
        dtype = model.dtype
        device = model.device
        self._setup_dicts(model)
        log_interval = int(self.log_interval // self.timestep)
        log_dict = self._make_reporter()
        trajectory_reporter = ts.TrajectoryReporter(
            [f"{self.logfile}_{i}.h5" for i in range(len(structure_list))],
            state_frequency=log_interval,
            prop_calculators={log_interval: log_dict},
        )

        initial_state = ts.initialize_state(
            [s.to_ase_atoms() for s in structure_list], device=device, dtype=dtype
        )
        if self.progress_bar:
            from tqdm import tqdm

            pbar_kwargs = {}
            pbar_kwargs.setdefault("desc", "Integrate")
            pbar_kwargs.setdefault("disable", None)
            pbar = tqdm(total=initial_state.n_systems, **pbar_kwargs)
        n_steps = int(self.duration // self.timestep)
        temps = [self.temperature] * n_steps
        kTs = torch.tensor(temps, dtype=dtype, device=device) * UNITS.temperature  # kbT
        dt = torch.tensor(self.timestep / 1000, dtype=dtype, device=device)  # ps
        init_func, step_func = ts.INTEGRATOR_REGISTRY[
            getattr(ts.Integrator, self.integrator.lower())
        ]
        batch_iterator = ts.runners._configure_batches_iterator(
            initial_state, model, autobatcher=self.autobatcher
        )
        _final_states: list[ts.SimState] | ts.SimState = []
        og_filenames = trajectory_reporter.filenames if trajectory_reporter else None
        for state, system_indices in batch_iterator:
            # Pass correct parameters based on integrator type
            state = init_func(state=state, model=model, kT=kTs[0], dt=dt, **self.init_kwargs)  # noqa: PLW2901
            # set up trajectory reporters
            if self.autobatcher and trajectory_reporter is not None and og_filenames is not None:
                # we must remake the trajectory reporter for each system
                trajectory_reporter.load_new_trajectories(
                    filenames=[og_filenames[i] for i in system_indices]
                )
            # run the simulation
            if self.progress_bar:
                step_pbar = tqdm(total=n_steps, desc="Steps", leave=False)
            for step in range(1, n_steps + 1):
                state = step_func(  # noqa: PLW2901
                    state=state, model=model, dt=dt, kT=kTs[step - 1], **self.step_kwargs
                )
                trajectory_reporter.report(state, step, model=model)
                if self.progress_bar:
                    step_pbar.update(1)
            if self.progress_bar:
                step_pbar.close()
                pbar.update(state.n_systems)
            # finish the trajectory reporter
            _final_states.append(state)
        if trajectory_reporter:
            trajectory_reporter.finish()
        if isinstance(batch_iterator, ts.BinningAutoBatcher):
            reordered_states = batch_iterator.restore_original_order(_final_states)
            final_states = [ts.concatenate_states(reordered_states)]  # type: ignore
        else:
            final_states = _final_states
        properties = self._parse_properties()
        final_structures = []
        for final_state in final_states:
            for atoms in final_state.to_atoms():
                # print(len(atoms))
                final_structures.append(
                    cast("OutputType", type(structure_list[0]).from_ase_atoms(atoms))
                )
        # Return as list to match return type annotation
        return cast("list[OutputType]", final_structures), cast("list[Properties]", properties)
