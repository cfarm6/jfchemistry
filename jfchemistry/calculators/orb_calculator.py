"""ORB machine learning force field calculator.

This module provides integration with Orbital Materials' ORB machine learning
force field models for molecular energy calculations.
"""

from dataclasses import dataclass
from typing import Literal

from ase import Atoms
from pydantic import BaseModel

from jfchemistry.base_classes import AtomicProperty, SystemProperty

from .ase_calculator import ASECalculator


class OrbAtomicProperties(BaseModel):
    """Properties of the ORB model calculation."""

    orb_forces: AtomicProperty


class OrbSystemProperties(BaseModel):
    """System properties of the ORB model calculation."""

    total_energy: SystemProperty


class OrbProperties(BaseModel):
    """Properties of the ORB model calculation."""

    atomic: OrbAtomicProperties
    system: OrbSystemProperties


@dataclass
class ORBModelCalculator(ASECalculator):
    """Orbital Materials ORB machine learning force field calculator.

    ORB models are graph neural network-based force fields developed by Orbital
    Materials for fast and accurate molecular property predictions. The calculator
    supports both conservative and direct versions of the ORB-v3 model.

    The calculator requires the 'orb-models' package from:
    https://github.com/orbital-materials/orb-models

    Attributes:
        name: Name of the calculator (default: "ORB Model Calculator").
        model: ORB model variant to use. Options:
            - "orb-v3-conservative-omol": Conservative model (recommended)
            - "orb-v3-direct-omol": Direct model
        charge: Molecular charge override. If None, uses charge from structure.
        multiplicity: Spin multiplicity override. If None, uses spin from structure.
        device: Computation device ("cpu" or "cuda"). Default: "cpu".
        precision: Numerical precision for calculations. Options:
            - "float32-high": Standard precision (default)
            - "float32-highest": Higher precision float32
            - "float64": Double precision
        compile: Whether to compile the model for faster inference (default: False).

    Examples:
        >>> from jfchemistry.calculators import ORBModelCalculator # doctest: +SKIP
        >>>
        >>> # Create calculator with GPU acceleration
        >>> calc = ORBModelCalculator(
        ...     model="orb-v3-conservative-omol", # doctest: +SKIP
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

    name: str = "ORB Model Calculator"
    model: Literal["orb-v3-conservative-omol", "orb-v3-direct-omol"] = "orb-v3-conservative-omol"
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32-high", "float32-highest", "float64"] = "float32-high"
    compile: bool = False

    _properties_model: type[OrbProperties] = OrbProperties

    def set_calculator(self, atoms: Atoms, charge: float = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the ORB model calculator on the atoms object.

        Loads the specified ORB model and attaches it as an ASE calculator to the
        atoms object. Stores charge and spin information in atoms.info dictionary.

        Args:
            atoms: ASE Atoms object to attach calculator to.
            charge: Total molecular charge (default: 0). Overridden by self.charge if set.
            spin_multiplicity: Spin multiplicity 2S+1 (default: 1). Overridden by
                self.multiplicity if set.

        Returns:
            ASE Atoms object with ORB calculator attached and charge/spin set.

        Raises:
            ImportError: If the 'orb-models' package is not installed.

        Examples:
            >>> calc = ORBModelCalculator(device="cuda", compile=True) # doctest: +SKIP
            >>> atoms = molecule.to_ase_atoms() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> # Charge and spin are stored in atoms.info
            >>> print(atoms.info["charge"]) # doctest: +SKIP
            0
        """
        try:
            from orb_models.forcefield import pretrained  # type: ignore
            from orb_models.forcefield.calculator import ORBCalculator  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'orb-models' package is required to use ORBCalculator but is not available."
                "Please install it from: https://github.com/orbital-materials/orb-models"
            ) from e
        if self.charge is not None:
            charge = self.charge
        if self.spin_multiplicity is not None:
            spin_multiplicity = self.spin_multiplicity

        orbff = getattr(pretrained, self.model.replace("-", "_"))(
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )

        atoms.calc = ORBCalculator(orbff, device=self.device)
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin_multiplicity
        return atoms

    def get_properties(self, atoms: Atoms) -> OrbProperties:
        """Extract computed properties from the ORB calculation.

        Retrieves the total energy from the ORB model calculation.

        Args:
            atoms: ASE Atoms object with ORB calculator attached and calculation
                completed.

        Returns:
            Dictionary with structure:
                - "Global": {"Total Energy [eV]": float}

        Examples:
            >>> calc = ORBModelCalculator() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> atoms.get_potential_energy()  # Trigger calculation # doctest: +SKIP
            >>> props = calc.get_properties(atoms) # doctest: +SKIP
            >>> print(props["Global"]["Total Energy [eV]"]) # doctest: +SKIP
            -234.567
        """
        energy = atoms.get_total_energy()  # type: ignore
        forces = atoms.get_forces()  # type: ignore
        atomic_properties = OrbAtomicProperties(
            orb_forces=AtomicProperty(
                name="ORB Forces",
                value=forces,
                units="eV/Å",
                description=f"Forces predicted by {self.model} model",
            ),
        )
        system_properties = OrbSystemProperties(
            total_energy=SystemProperty(
                name="Total Energy",
                value=energy,
                units="eV",
                description=f"Total energy prediction from {self.model} model",
            ),
        )
        return OrbProperties(
            atomic=atomic_properties,
            system=system_properties,
        )
