"""FairChem machine learning force field calculator.

This module provides integration with FairChemital Materials' FairChem machine learning
force field models for molecular energy calculations.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, UMATask
from monty.json import MSONable

from jfchemistry.base_classes import AtomicProperty, SystemProperty
from jfchemistry.base_jobs import Properties, PropertyClass
from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.calculators.base import MachineLearnedInteratomicPotentialCalculator

model_types = Literal[
    *[model for model in _MODEL_CKPTS.checkpoints.keys() if isinstance(model, str)]  # type: ignore
]
task_type = Literal[*[x.value for x in list(UMATask)]]  # type: ignore


class FairChemAtomicProperties(PropertyClass):
    """Properties of the FairChem model calculation."""

    FairChem_forces: AtomicProperty


class FairChemSystemProperties(PropertyClass):
    """System properties of the FairChem model calculation."""

    total_energy: SystemProperty


class FairChemProperties(Properties):
    """Properties of the FairChem model calculation."""

    atomic: FairChemAtomicProperties
    system: FairChemSystemProperties


@dataclass
class FairChemCalculator(ASECalculator, MachineLearnedInteratomicPotentialCalculator, MSONable):
    """FairChemital Materials FairChem machine learning force field calculator.

    FairChem models are graph neural network-based force fields developed by FairChem
    Materials for fast and accurate molecular property predictions. The calculator
    supports both conservative and direct versions of the FairChem-v3 model.

    Attributes:
        name: Name of the calculator (default: "FairChem Model Calculator").
        model: FairChem model variant to use. Options:
            - "FairChem-v3-conservative-omol": Conservative model (recommended)
            - "FairChem-v3-direct-omol": Direct model
        charge: Molecular charge override. If None, uses charge from structure.
        multiplicity: Spin multiplicity override. If None, uses spin from structure.
        device: Computation device ("cpu" or "cuda"). Default: "cpu".
        precision: Numerical precision for calculations. Options:
            - "float32-high": Standard precision (default)
            - "float32-highest": Higher precision float32
            - "float64": Double precision
        compile: Whether to compile the model for faster inference (default: False).

    Examples:
        >>> from jfchemistry.calculators import FairChemModelCalculator # doctest: +SKIP
        >>>
        >>> # Create calculator with GPU acceleration
        >>> calc = FairChemModelCalculator(
        ...     model="FairChem-v3-conservative-omol", # doctest: +SKIP
        ...     device="cuda", # doctest: +SKIP
        ...     precision="float32-highest" # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>>
        >>> # Setup on structure
        >>> atoms = molecule.to_ase_atoms() # doctest: +SKIP
        >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
        >>>
        >>> # Get properties
        >>> props = calc.get_properties(atoms) # doctest: +SKIP
        >>> energy = props["Global"]["Total Energy [eV]"] # doctest: +SKIP
    """

    name: str = "FairChem Model Calculator"
    model: model_types = field(
        default="uma-s-1",
        metadata={"description": "The FairChem model to use"},
    )
    device: Literal["cpu", "cuda"] = field(
        default="cpu", metadata={"description": "The device to use"}
    )
    task: task_type = field(default="omol", metadata={"description": "The task to use"})
    workers: int = field(default=1, metadata={"description": "The number of workers to use"})
    turbo: bool = field(
        default=False,
        metadata={
            "description": "For long rollout trajectory\
         use-cases, such as molecular dynamics (MD) or relaxations, we provide a special mode\
          called turbo, which optimizes for speed but restricts the user to using a single system\
             where the atomic composition is held constant. Turbo mode is approximately 1.5-2x\
                 faster than default mode, depending on the situation. However, batching is not\
                         supported in this mode. It can be easily activated as shown below."
        },
    )
    tf32: bool = field(
        default=False,
        metadata={
            "description": "Flag to enable or disable the use of\
         tf32 data type for inference. TF32 will slightly reduce accuracy compared to FP32 but will\
             still keep energy conservation in most cases."
        },
    )
    activation_checkpointing: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Flag to enable or disable activation checkpointing\
             during inference. This will dramatically decrease the memory footprint especially for\
                 large number of atoms (ie 10+) at a slight cost to inference speed. If set to None\
                    , the setting from the model checkpoint will be used."
        },
    )
    merge_mole: bool = field(
        default=False,
        metadata={
            "description": "Flag to enable or disable the merging of MOLE experts during inference.\
                 If this is used, the input composition, total charge and spin MUST remain constant\
                     throughout the simulation this will slightly increase speed and reduce memory\
                         footprint used by the parameters significantly"
        },
    )
    compile: bool = field(
        default=False,
        metadata={
            "description": "Flag to enable or disable the compilation of the inference model."
        },
    )

    _properties_model: type[FairChemProperties] = FairChemProperties

    def set_calculator(self, atoms: Atoms, charge: float = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the FairChem model calculator on the atoms object.

        Loads the specified FairChem model and attaches it as an ASE calculator to the
        atoms object. Stores charge and spin information in atoms.info dictionary.

        Args:
            atoms: ASE Atoms object to attach calculator to.
            charge: Total molecular charge (default: 0). Overridden by self.charge if set.
            spin_multiplicity: Spin multiplicity 2S+1 (default: 1). Overridden by
                self.multiplicity if set.

        Returns:
            ASE Atoms object with FairChem calculator attached and charge/spin set.

        Raises:
            ImportError: If the 'FairChem-models' package is not installed.

        Examples:
            >>> calc = FairChemModelCalculator(device="cuda", compile=True) # doctest: +SKIP
            >>> atoms = molecule.to_ase_atoms() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> # Charge and spin are stored in atoms.info
            >>> print(atoms.info["charge"]) # doctest: +SKIP
            0
        """
        atoms.info.update({"charge": charge})
        atoms.info.update({"spin": spin_multiplicity})

        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model,
            device=self.device,
            workers=self.workers,
            inference_settings=InferenceSettings(
                tf32=self.tf32,
                activation_checkpointing=self.activation_checkpointing,
                merge_mole=self.merge_mole,
                compile=self.compile,
            )
            if not self.turbo
            else "turbo",
        )

        atoms.calc = FAIRChemCalculator(predictor, task_name=self.task)
        return atoms

    def get_properties(self, atoms: Atoms) -> FairChemProperties:
        """Extract computed properties from the FairChem calculation.

        Retrieves the total energy from the FairChem model calculation.

        Args:
            atoms: ASE Atoms object with FairChem calculator attached and calculation
                completed.

        Returns:
            Dictionary with structure:
                - "Global": {"Total Energy [eV]": float}

        Examples:
            >>> calc = FairChemModelCalculator() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> atoms.get_potential_energy()  # Trigger calculation # doctest: +SKIP
            >>> props = calc.get_properties(atoms) # doctest: +SKIP
            >>> print(props["Global"]["Total Energy [eV]"]) # doctest: +SKIP
            -234.567
        """
        print(atoms.info)
        energy = atoms.get_total_energy()  # type: ignore
        forces = atoms.get_forces()  # type: ignore
        atomic_properties = FairChemAtomicProperties(
            FairChem_forces=AtomicProperty(
                name="FairChem Forces",
                value=forces,
                units="eV/Å",
                description=f"Forces predicted by {self.model} model and {self.task} task",
            ),
        )
        system_properties = FairChemSystemProperties(
            total_energy=SystemProperty(
                name="Total Energy",
                value=energy,
                units="eV",
                description=f"Total energy prediction from {self.model} model and {self.task} task",
            ),
        )
        return FairChemProperties(
            atomic=atomic_properties,
            system=system_properties,
        )
