"""AimNet2 neural network calculator for molecular properties.

This module provides integration with the AimNet2 neural network potential
for fast and accurate calculation of molecular energies and partial charges.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
from ase import Atoms
from monty.json import MSONable

from jfchemistry import ureg
from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.calculators.base import MachineLearnedInteratomicPotentialCalculator
from jfchemistry.core.properties import AtomicProperty, Properties, PropertyClass, SystemProperty


class AimNet2AtomicProperties(PropertyClass):
    """Properties of the AimNet2 calculator.

    Attributes:
        aimnet2_partial_charges: Partial charges of the atoms.
        aimnet2_forces: Forces of the atoms.
    """

    aimnet2_partial_charges: AtomicProperty
    forces: AtomicProperty


class AimNet2SystemProperties(PropertyClass):
    """System properties of the AimNet2 calculator.

    Attributes:
        total_energy: Total energy of the system.
    """

    total_energy: SystemProperty


class AimNet2Properties(Properties):
    """Properties of the AimNet2 calculator.

    Attributes:
        atomic: Atomic properties of the AimNet2 calculator.
        system: System properties of the AimNet2 calculator.
    """

    atomic: AimNet2AtomicProperties
    system: AimNet2SystemProperties


@dataclass
class AimNet2Calculator(ASECalculator, MachineLearnedInteratomicPotentialCalculator, MSONable):
    """AimNet2 neural network potential calculator.

    AimNet2 is a neural network-based calculator for computing molecular energies
    and atomic partial charges. It provides fast predictions for molecules containing
    H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, and I atoms.

    The calculator requires the 'aimnet' package from:
    https://github.com/cfarm6/aimnetcentral.git

    Attributes:
        name: Name of the calculator (default: "AimNet2 Calculator").
        model: AimNet2 model to use (default: "aimnet2").
        charge: Molecular charge override. If None, uses charge from structure.
        multiplicity: Spin multiplicity override. If None, uses spin from structure.

    """

    name: str = "AimNet2 Calculator"
    model: Literal["aimnet2_wb97m", "aimnet2_2025", "aimnet2nse", "aimnet2pd"] = field(
        default="aimnet2_2025", metadata={"description": "AimNet2 model to use"}
    )
    nb_threshold: int = field(
        default=120, metadata={"description": "Number of neighbors threshold for the model"}
    )
    needs_coulomb: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Whether the model needs Coulomb interaction. \
            If None, the model will use what is defined in the model metadata."
        },
    )
    needs_dispersion: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Whether the model needs dispersion interaction. \
            If None, the model will use what is defined in the model metadata."
        },
    )
    device: Literal["cpu", "cuda"] = field(
        default="cpu", metadata={"description": "The device to use for the calculator"}
    )
    compile_model: bool = field(
        default=False, metadata={"description": "Whether to compile the model"}
    )
    compile_kwargs: dict[str, str] = field(
        default={}, metadata={"description": "Keyword arguments for the model compilation"}
    )
    _properties_model: type[AimNet2Properties] = AimNet2Properties

    def _set_calculator(self, atoms: Atoms, charge: float = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the AimNet2 calculator on the atoms object.

        Attaches the AimNet2 ASE calculator to the atoms object with the specified
        charge and spin multiplicity. Validates that all atoms are supported by AimNet2.

        Args:
            atoms: ASE Atoms object to attach calculator to.
            charge: Total molecular charge (default: 0). Overridden by self.charge if set.
            spin_multiplicity: Spin multiplicity 2S+1 (default: 1). Overridden by
                self.multiplicity if set.

        Returns:
            ASE Atoms object with AimNet2 calculator attached.

        Raises:
            ImportError: If the 'aimnet' package is not installed.
            ValueError: If molecule contains atoms not supported by AimNet2.

        Examples:
            >>> from ase import Atoms # doctest: +SKIP
            >>> calc = AimNet2Calculator() # doctest: +SKIP
            >>> atoms = Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]]) # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> energy = atoms.get_potential_energy() # doctest: +SKIP
        """
        if self.charge is not None:
            charge = self.charge
        if self.spin_multiplicity is not None:
            spin_multiplicity = self.spin_multiplicity
        model = AIMNet2Calculator(
            self.model,
            nb_threshold=self.nb_threshold,
            needs_coulomb=self.needs_coulomb,
            needs_dispersion=self.needs_dispersion,
            device=self.device,
            compile_model=self.compile_model,
            compile_kwargs=self.compile_kwargs,
        )
        atoms.calc = AIMNet2ASE(model, charge, spin_multiplicity)

        aimnet2_atomtypes = [1, 6, 7, 8, 9, 17, 16, 5, 14, 15, 33, 34, 35, 53]
        atomic_nums = atoms.get_atomic_numbers()
        if not all(atom in aimnet2_atomtypes for atom in atomic_nums):
            raise ValueError(
                f"Unsupport atomtype by AimNet2. Supported atom types are {aimnet2_atomtypes}"
            )

        return atoms

    def _get_properties(self, atoms: Atoms) -> AimNet2Properties:
        """Extract computed properties from the AimNet2 calculation.

        Retrieves the total energy and atomic partial charges from the AimNet2
        calculation on the atoms object.

        Args:
            atoms: ASE Atoms object with AimNet2 calculator attached and calculation
                completed.

        Returns:
            Dictionary with structure:
                - "Global": {"Total Energy [eV]": float}
                - "Atomic": {"AimNet2 Partial Charges [e]": array}

        Examples:
            >>> calc = AimNet2Calculator() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> atoms.get_potential_energy()  # Trigger calculation # doctest: +SKIP
            >>> props = calc.get_properties(atoms) # doctest: +SKIP
        """
        energy = atoms.get_total_energy()
        system_property = AimNet2SystemProperties(
            total_energy=SystemProperty(
                name="Total Energy",
                value=energy * ureg.eV,
                description=f"Total energy prediction from {self.model} model",
            ),
        )

        charge = atoms.get_charges()
        atomic_property = AimNet2AtomicProperties(
            aimnet2_partial_charges=AtomicProperty(
                name="AimNet2 Partial Charges",
                value=charge * ureg.e,
                description=f"Partial charges predicted by {self.model} model",
            ),
            forces=AtomicProperty(
                name="AimNet2 Forces",
                value=atoms.get_forces() * ureg.eV / ureg.angstrom,
                description=f"Forces predicted by {self.model} model",
            ),
        )

        return AimNet2Properties(
            atomic=atomic_property,
            system=system_property,
        )
