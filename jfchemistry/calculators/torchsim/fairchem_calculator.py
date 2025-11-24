"""Base class for using TorchSim with FairChem Models."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch_sim as ts
from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS
from fairchem.core.units.mlip_unit.api.inference import UMATask
from monty.json import MSONable
from pymatgen.core import Structure
from torch_sim.models.fairchem import FairChemModel

from jfchemistry.calculators.base import MachineLearnedInteratomicPotentialCalculator
from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.properties import AtomicProperty, Properties, PropertyClass, SystemProperty

model_types = Literal[
    *[model for model in _MODEL_CKPTS.checkpoints.keys() if isinstance(model, str)]  # type: ignore
]
task_type = Literal[*[x.value for x in list(UMATask)]]  # type: ignore


class FairChemAtomicProperties(PropertyClass):
    """Properties of the FairChem model calculation."""

    forces: AtomicProperty


class FairChemSystemProperties(PropertyClass):
    """System properties of the FairChem model calculation."""

    total_energy: SystemProperty
    stress: Optional[SystemProperty] = None


class FairChemProperties(Properties):
    """Properties of the FairChem model calculation."""

    atomic: FairChemAtomicProperties
    system: FairChemSystemProperties


@dataclass
class FairChemCalculator(
    TorchSimCalculator, MachineLearnedInteratomicPotentialCalculator, MSONable
):
    """Base class for using TorchSim with FairChem Models."""

    name: str = "FairChem TorchSim Calculator"
    model: model_types = field(
        default="uma-s-1", metadata={"description": "The FairChem model to use"}
    )
    task: task_type = field(default="omol", metadata={"description": "The task to use"})
    compute_stress: bool = field(
        default=False, metadata={"description": "Whether to compute the stress"}
    )

    def get_model(self) -> FairChemModel:
        """Get the FairChem model."""
        model = FairChemModel(
            None,
            model_name=self.model,
            task_name=self.task,
            cpu=self.device == "cpu",  # type: ignore
            compute_stress=self.compute_stress,  # type: ignore
        )
        self._model = model
        return model

    def get_properties(self, structure: Structure) -> FairChemProperties:
        """Get the properties of the FairChem model."""
        if not hasattr(self, "_model"):
            self.get_model()
        prop_calculators = {
            10: {"potential_energy": lambda state: state.energy},
            20: {"forces": lambda state: state.forces},
        }

        if self.compute_stress:
            prop_calculators[30] = {"stress": lambda state: state.stress}

        """Get the properties of the FairChem model"""
        final_results = ts.static(
            system=structure.to_ase_atoms(),
            model=self._model,
            # we don't want to save any trajectories this time, just get the properties
            trajectory_reporter={"filenames": None, "prop_calculators": prop_calculators},
        )
        forces = final_results[0]["forces"]
        energy = final_results[0]["potential_energy"]
        if self.compute_stress:
            stress = final_results[0]["stress"]
        properties = FairChemProperties(
            atomic=FairChemAtomicProperties(
                forces=AtomicProperty(
                    name="FairChem Forces",
                    value=forces.tolist(),
                    units="eV/Å",
                    description=f"Forces predicted by the {self.model} model and {self.task}",
                ),
            ),
            system=FairChemSystemProperties(
                total_energy=SystemProperty(
                    name="Total Energy",
                    value=energy.tolist(),
                    units="eV",
                    description=f"Total energy predicted by the {self.model} model and {self.task}",
                ),
                stress=SystemProperty(
                    name="Stress",
                    value=stress.tolist(),
                    units="eV/Å^3",
                    description=f"Stress predicted by the {self.model} model and {self.task}",
                )
                if self.compute_stress
                else None,
            ),
        )
        return properties
