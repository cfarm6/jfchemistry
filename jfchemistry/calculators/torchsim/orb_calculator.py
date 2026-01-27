"""Base class for using TorchSim with Orb Models."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch_sim as ts
from monty.json import MSONable
from orb_models.forcefield import pretrained
from pymatgen.core import SiteCollection
from torch_sim.models.orb import OrbModel

from jfchemistry import ureg
from jfchemistry.calculators.base import MachineLearnedInteratomicPotentialCalculator
from jfchemistry.calculators.torchsim.torchsim_calculator import TorchSimCalculator
from jfchemistry.core.properties import AtomicProperty, Properties, PropertyClass, SystemProperty


class OrbAtomicProperties(PropertyClass):
    """Properties of the Orb model calculation."""

    forces: AtomicProperty


class OrbSystemProperties(PropertyClass):
    """System properties of the Orb model calculation."""

    total_energy: SystemProperty
    stress: Optional[SystemProperty] = None


class OrbProperties(Properties):
    """Properties of the Orb model calculation."""

    atomic: OrbAtomicProperties
    system: OrbSystemProperties


@dataclass
class OrbCalculator(TorchSimCalculator, MachineLearnedInteratomicPotentialCalculator, MSONable):
    """Orb TorchSim Calculator.

    Attributes:
        name: Name of the calculator (default: "Orb TorchSim Calculator").
        model: The Orb model to use (default: "orb_v3_conservative_omol").
        device: The device to use for the model (default: "cpu").
        conservative: Whether to use the conservative model (default: True).
        precision: The precision to use for the model (default: "float32-high").
    """

    name: str = "Orb TorchSim Calculator"
    model: Literal[
        "orb_v3_conservative_omol",
        "orb_v3_direct_omol",
        "orb_v3_direct_20_omat",
        "orb_v3_direct_20_mpa",
        "orb_v3_direct_inf_omat",
        "orb_v3_direct_inf_mpa",
        "orb_v3_conservative_20-omat",
        "orb_v3_conservative_20_mpa",
        "orb_v3_conservative_inf_omat",
        "orb_v3_conservative_inf_mpa",
    ] = field(default="orb_v3_conservative_omol", metadata={"description": "The ORB model to use"})
    device: Literal["cpu", "cuda"] = field(
        default="cpu", metadata={"description": "The device to use for the model"}
    )
    conservative: bool = field(
        default=True, metadata={"description": "Whether to use the conservative model"}
    )
    precision: Literal["float32-high", "float32-highest", "float64"] = field(
        default="float32-high", metadata={"description": "The precision to use for the model"}
    )
    compile: bool = field(default=True, metadata={"description": "Whether to compile the model"})
    compute_stress: bool = field(
        default=False, metadata={"description": "Whether to compute the stress"}
    )

    def _get_model(self) -> OrbModel:
        """Get the Orb model."""
        orb_model = getattr(pretrained, self.model)(
            device=self.device, precision=self.precision, compile=self.compile
        )
        model = OrbModel(
            model=orb_model,
            device=self.device,
            compute_stress=self.compute_stress,
        )
        self._model = model
        return model

    def _get_properties(self, system: SiteCollection) -> Properties:
        """Get the properties of the Orb model."""
        if not hasattr(self, "_model"):
            self._get_model()
        prop_calculators = {
            10: {"potential_energy": lambda state: state.energy},
            20: {"forces": lambda state: state.forces},
        }

        if self.compute_stress:
            prop_calculators[30] = {"stress": lambda state: state.stress}

        """Get the properties of the Orb model"""
        final_results = ts.static(
            system=system.to_ase_atoms(),
            model=self._model,
            # we don't want to save any trajectories this time, just get the properties
            trajectory_reporter={"filenames": None, "prop_calculators": prop_calculators},
        )
        forces = final_results[0]["forces"]
        energy = final_results[0]["potential_energy"]
        if self.compute_stress:
            stress = final_results[0]["stress"]
        properties = OrbProperties(
            atomic=OrbAtomicProperties(
                forces=AtomicProperty(
                    name="Orb Forces",
                    value=forces.tolist() * ureg.eV / ureg.angstrom,
                    description=f"Forces predicted by the {self.model} model",
                ),
            ),
            system=OrbSystemProperties(
                total_energy=SystemProperty(
                    name="Total Energy",
                    value=energy.tolist() * ureg.eV,
                    description=f"Total energy predicted by the {self.model} model",
                ),
                stress=SystemProperty(
                    name="Stress",
                    value=stress.tolist() * ureg.eV / ureg.angstrom**3,
                    description=f"Stress predicted by the {self.model} model",
                )
                if self.compute_stress
                else None,
            ),
        )
        return properties
