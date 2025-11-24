"""TorchSim-based molecular dynamics framework.

This module provides the base framework for molecular dynamics using
ASE (Atomic Simulation Environment) optimizers with various calculators.
"""

import glob
import importlib
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Optional

import torch
import torch_sim as ts
from ase.atoms import Atoms
from jobflow.core.job import Response
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model
from pymatgen.core import SiteCollection, Structure
from torch_sim.models.interface import ModelInterface
from torch_sim.units import UnitConversion as Uc

from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.jfchem_job import jfchem_job
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


class TorchSimMolecularDynamicsOutput(MolecularDynamicsOutput):
    """Output for TorchSim Molecular Dynamics."""

    properties: TSMDProperties | list[TSMDProperties]


@dataclass
class TorchSimMolecularDynamics(MolecularDynamics):
    """Base class for single point energy calculations using TorchSim calculators.

    Combines single point energy calculations with TorchSim calculator interfaces.
    This class provides the framework for calculating the single point energy
    of a structure using various TorchSim calculators (neural networks, machine learning
    , semi-empirical, etc.).

    Attributes:
        name: Name of the calculator (default: "ASE Single Point Calculator").

    """

    name: str = "TorchSim Molecular Dynamics"
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

    _output_model: type[TorchSimMolecularDynamicsOutput] = TorchSimMolecularDynamicsOutput
    _properties_model: type[TSMDProperties] = TSMDProperties

    def make_output_model(self, properties_model: type[Properties]):
        """Make a properties model for the job."""
        fields = {}
        if isinstance(self._output_model, dict):
            module = self._output_model["@module"]
            class_name = self._output_model["@callable"]
            self._output_model = getattr(importlib.import_module(module), class_name)
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    properties_model
                    | list[properties_model]  # type: ignore[type-arg]
                    | OutputReference
                    | list[OutputReference]
                )  # type: ignore

            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
                    *f_dict["metadata"],  # type: ignore
                    Field(**f_dict["attributes"]),
                ],  # type: ignore
                None,
            )

        self._output_model = create_model(
            f"{self._output_model.__name__}",
            __base__=self._output_model,
            **fields,
        )

    def __post_init__(self):
        """Post initialization hook."""
        self.init_kwargs = {}
        self.step_kwargs = {}
        self.make_output_model(self._properties_model)

    def write_file(self, structure: SiteCollection) -> str | None:
        """Write the structure to a file."""
        if isinstance(structure, Structure):
            return structure.to(fmt="cif")
        else:
            return structure.to(fmt="xyz")

    def setup_dicts(self, model: ModelInterface):
        """Setup the dictionaries for the integrator."""
        raise NotImplementedError("This method is not implemented for TorchSimMolecularDynamics")

    def parse_properties(self) -> list[TSMDProperties]:
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
                        value=trajectory.get_array("potential_energy").flatten().tolist(),
                        units="eV",
                    )
                if self.log_kinetic_energy:
                    properties["kinetic_energy"] = SystemProperty(
                        name="kinetic_energy",
                        value=trajectory.get_array("kinetic_energy").flatten().tolist(),
                        units="eV",
                    )
                if self.log_temperature:
                    properties["temperature"] = SystemProperty(
                        name="temperature",
                        value=trajectory.get_array("temperature").flatten().tolist(),
                        units="K",
                    )
                if self.log_pressure:
                    properties["pressure"] = SystemProperty(
                        name="pressure",
                        value=trajectory.get_array("pressure") * Uc.bar_to_pa / Uc.atm_to_pa,
                        units="atm",
                    )
                if self.log_volume:
                    properties["volume"] = SystemProperty(
                        name="volume",
                        value=trajectory.get_array("volume").flatten().tolist(),
                        units="A^3",
                    )
            property = TSMDProperties(system=TSMDSystemProperties(**properties))
            property_objects.append(property)
        return property_objects

    def parse_trajectory(self) -> list[list[Atoms]]:
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

    def make_reporter(self) -> dict[str, Callable[[ts.SimState], Any]]:
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

    def operation(
        self,
        structure: SiteCollection | list[SiteCollection],
        calculator: TorchSimCalculator,
        **kwargs: Any,
    ) -> tuple[
        SiteCollection | list[SiteCollection],
        TSMDProperties | list[TSMDProperties],
    ]:
        """Optimize molecular structure using ASE.

        Performs molecular dynamics simulation by:
        1. Converting structure to ASE Atoms
        2. Setting up the calculator with charge and spin
        3. Running the calculator
        4. Converting back to Pymatgen Molecule
        5. Extracting properties from the calculation

        Args:
            structure: Input molecular structure with 3D coordinates.
            calculator: TorchSimCalculator to use for the calculation.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Tuple containing:
                - Optimized Pymatgen Molecule
                - Dictionary of computed properties from calculator

        Examples:
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from jfchemistry.optimizers import TBLiteOptimizer # doctest: +SKIP
            >>> ethane = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> opt = TBLiteOptimizer(optimizer="LBFGS", fmax=0.01) # doctest: +SKIP
            >>> structures, properties = opt.operation(ethane) # doctest: +SKIP
        """
        if not isinstance(structure, list):
            structure = [structure]
        model = calculator.get_model()
        dtype = model.dtype
        device = model.device
        self.setup_dicts(model)
        log_interval = int(self.log_interval // self.timestep)
        log_dict = self.make_reporter()
        trajectory_reporter = ts.TrajectoryReporter(
            [f"{self.logfile}_{i}.h5" for i in range(len(structure))],
            state_frequency=log_interval,
            prop_calculators={log_interval: log_dict},
        )

        if not isinstance(structure, list):
            structure = [structure]
        initial_state = ts.initialize_state(
            [structure.to_ase_atoms() for structure in structure], device=device, dtype=dtype
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
        properties = self.parse_properties()
        final_structures: list[SiteCollection] = []
        for final_state in final_states:
            for atoms in final_state.to_atoms():
                # print(len(atoms))
                final_structures.append(type(structure[0]).from_ase_atoms(atoms))
        return final_structures, properties

    @jfchem_job()
    def make(
        self,
        structure: SiteCollection | list[SiteCollection],
        calculator: TorchSimCalculator,
        **kwargs: Any,
    ) -> Response[_output_model]:
        """Create a workflow job for processing structure(s).

        Automatically handles job distribution for lists of structures. Each
        structure in a list is processed as a separate job for parallel execution.

        Args:
            structure: Single Pymatgen SiteCollection or list of SiteCollections.
            calculator: TorchSimCalculator to use for the calculation.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Response containing:
                - structure: Processed structure(s)
                - files: XYZ format file(s) of the structure(s)
                - properties: Computed properties from the operation

        Examples:
            >>> from jfchemistry.conformers import CRESTConformers # doctest: +SKIP
            >>> from pymatgen.core import Molecule # dokctest: +SKIP
            >>> molecule = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> # Generate conformers
            >>> conformer_gen = CRESTConformers(ewin=6.0) # doctest: +SKIP
            >>> job = conformer_gen.make(molecule) # doctest: +SKIP
        """
        if isinstance(structure, SiteCollection):
            structure = [structure]
        structures, properties = self.operation(structure, calculator, **kwargs)
        if isinstance(structures, list):
            files = [self.write_file(s) for s in structures]
        else:
            files = [self.write_file(structures)]
        if self.log_trajectory:
            trajectories = self.parse_trajectory() if self.log_trajectory else None
            trajectory: list[list[SiteCollection]] = [
                [type(structures[i]).from_ase_atoms(atoms) for atoms in trajectory]
                for i, trajectory in enumerate(trajectories)
            ]
        return Response(
            output=self._output_model(
                structure=structures,
                files=files,
                properties=properties,
                trajectory=trajectory if self.log_trajectory else None,
            ),
        )
