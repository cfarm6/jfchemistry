"""Base Class for ORCA DFT Calculations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from monty.json import MSONable
from opi.core import Calculator
from opi.input.simple_keywords.base import SimpleKeyword
from opi.input.simple_keywords.basis_set import BasisSet
from opi.input.simple_keywords.dft import Dft
from opi.input.simple_keywords.dispersion_correction import DispersionCorrection
from opi.input.simple_keywords.ecp import Ecp
from opi.input.simple_keywords.solvation import Solvation
from opi.input.simple_keywords.solvation_model import SolvationModel
from opi.input.simple_keywords.solvent import Solvent
from opi.output.core import Output

from jfchemistry import AtomicProperty, SystemProperty, ureg
from jfchemistry.calculators.base import WavefunctionCalculator

# Import fully typed Literal definitions
from jfchemistry.calculators.orca.orca_keywords import (
    BasisSetType,
    DispersionCorrectionType,
    ECPType,
    SolvationModelType,
    SolvationType,
    SolventType,
    XCFunctionalType,
)
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.core.solvation import ImplicitSolventConfig, to_orca

# Re-export types for external use
__all__ = [
    "BasisSetType",
    "DispersionCorrectionType",
    "ECPType",
    "ORCACalculator",
    "SolvationModelType",
    "SolvationType",
    "SolventType",
    "XCFunctionalType",
]


class ORCASystemProperties(PropertyClass):
    """System properties of the ORCA calculation."""

    total_energy: SystemProperty
    solvation_energy: Optional[SystemProperty] = None


class ORCAAtomicProperties(PropertyClass):
    """Atomic properties of the ORCA calculation."""

    homo_participation_ratio: Optional[AtomicProperty] = None
    lumo_participation_ratio: Optional[AtomicProperty] = None


class ORCAProperties(Properties):
    """Properties of the ORCA calculation."""

    system: ORCASystemProperties
    atomic: ORCAAtomicProperties


@dataclass
class ORCACalculator(WavefunctionCalculator, MSONable):
    """ORCA DFT Calculator with full type support.

    This calculator wraps the ORCA Python Interface (OPI) package to provide
    DFT calculation capabilities. It supports various basis sets, exchange-correlation
    functionals, effective core potentials (ECPs), and solvation models.

    All keyword parameters are fully typed with Literal types extracted from the
    OPI package, providing complete IDE autocompletion and type checking for
    937 available options.

    Attributes:
        name: Name of the calculator (default: "ORCA").
        cores: Number of CPU cores to use for parallel calculations (default: 1).
        maxcore: ORCA MaxCore setting (MB per core).
        basis_set: Basis set to use for the calculation (488 options available).
        xc_functional: Exchange-correlation functional to use (195 options available).
        ecp: Effective core potential to use for heavy atoms (13 options available).
        solvation: Solvation model type to use (3 options available).
        solvent: Solvent specification for solvation calculations (236 options available).
        solvation_model: Specific solvation model implementation (2 options available).
        working_dir: Working directory for ORCA files.
        profile: OPI profile name/path used to resolve executable settings.
        launch_command: Launch command used to execute ORCA.
        command_arg: Additional command argument passed to ORCA.
        additional_keywords: Raw ORCA keywords appended to the input.
        additional_arbitrary_strings: Raw input strings/blocks added verbatim.
        charge: Molecular charge override. If None, uses charge from structure.
        spin_multiplicity: Spin multiplicity override. If None, uses spin from structure.
    """

    name: str = "ORCA"
    cores: int = field(
        default=1,
        metadata={"description": "The number of CPU cores to use for parallel calculations"},
    )
    basis_set: Optional[BasisSetType] = field(
        default=None, metadata={"description": "The basis set to use for the calculation"}
    )
    xc_functional: Optional[XCFunctionalType] = field(
        default=None,
        metadata={"description": "The exchange-correlation functional to use for the calculation"},
    )
    ecp: Optional[ECPType] = field(
        default=None,
        metadata={"description": "The effective core potential to use for heavy atoms"},
    )
    dispersion_correction: Optional[DispersionCorrectionType] = field(
        default=None,
        metadata={"description": "The dispersion correction to use for the calculation"},
    )
    solvation: Optional[SolvationType] = field(
        default=None, metadata={"description": "The solvation model to use for the calculation"}
    )
    solvent: Optional[SolventType] = field(
        default=None, metadata={"description": "The solvent to use for the calculation"}
    )
    solvation_model: Optional[SolvationModelType] = field(
        default=None,
        metadata={
            "description": "The specific solvation model implementation to use for the calculation"
        },
    )
    implicit_solvent: Optional[ImplicitSolventConfig] = field(
        default=None,
        metadata={"description": "Unified implicit-solvent configuration override."},
    )
    maxcore: Optional[int] = field(
        default=None,
        metadata={"description": "ORCA MaxCore setting in MB per core"},
    )
    working_dir: str = field(
        default=".",
        metadata={"description": "Working directory for ORCA files"},
    )
    profile: Optional[str] = field(
        default=None, metadata={"description": "OPI profile name or profile path"}
    )
    launch_command: Optional[str] = field(
        default=None, metadata={"description": "Launch command used to execute ORCA"}
    )
    command_arg: Optional[str] = field(
        default=None, metadata={"description": "Additional command argument passed to ORCA"}
    )
    additional_keywords: list[str] = field(
        default_factory=list,
        metadata={"description": "Raw ORCA keywords appended after typed simple keywords"},
    )
    additional_arbitrary_strings: list[str] = field(
        default_factory=list,
        metadata={"description": "Raw ORCA input lines or blocks added verbatim"},
    )
    additional_input_files: list[str] = field(
        default_factory=list,
        metadata={"description": "Extra files staged for the ORCA run"},
    )
    _properties_model: type[ORCAProperties] = ORCAProperties

    def __post_init__(self):
        """Apply unified implicit-solvent overrides when provided."""
        if self.implicit_solvent is None:
            return
        mapped = to_orca(self.implicit_solvent)
        self.solvation = mapped["solvation"]  # type: ignore[assignment]
        self.solvation_model = mapped["solvation_model"]  # type: ignore[assignment]
        self.solvent = mapped["solvent"]  # type: ignore[assignment]

    def _set_keywords(self) -> list[SimpleKeyword]:
        """Construct OPI simple keywords from calculator settings.

        Builds a list of SimpleKeyword objects from the configured calculator
        parameters. Handles both simple keywords (basis set, xc functional, etc.)
        and complex keywords (solvation models with solvents).

        Returns:
            List of SimpleKeyword objects ready for OPI input generation.
        """
        keywords: list[SimpleKeyword] = []

        # Add simple keywords: basis set, xc functional, ecp, solvation
        simple_keyword_classes = {
            "basis_set": BasisSet,
            "xc_functional": Dft,
            "ecp": Ecp,
            "solvation": Solvation,
            "dispersion_correction": DispersionCorrection,
        }

        for attr_name, keyword_class in simple_keyword_classes.items():
            keyword_value = getattr(self, attr_name)
            if keyword_value is not None:
                keyword_obj = getattr(keyword_class, keyword_value.upper())
                keywords.append(keyword_obj)

        # Add complex solvation keyword: model(solvent)
        if self.solvation_model is not None and self.solvent is not None:
            model_class = getattr(SolvationModel, self.solvation_model.upper())
            solvent_obj = getattr(Solvent, self.solvent.upper())
            keywords.append(model_class(solvent_obj))
            if self.solvation is not None:
                solvation_obj = getattr(Solvation, self.solvation.upper())
                keywords.append(solvation_obj)
        return keywords

    def _build_calculator(self, basename: str) -> Calculator:
        """Create an OPI Calculator configured with runtime execution options."""
        calc_kwargs: dict[str, Any] = {
            "basename": basename,
            "working_dir": Path(self.working_dir).as_posix(),
        }
        if self.profile is not None:
            calc_kwargs["profile"] = self.profile
        if self.launch_command is not None:
            calc_kwargs["launch_command"] = self.launch_command
        if self.command_arg is not None:
            calc_kwargs["command_arg"] = self.command_arg
        return Calculator(**calc_kwargs)

    def _configure_calculator_input(
        self, calc: Calculator, simple_keywords: list[SimpleKeyword]
    ) -> None:
        """Apply simple keywords and advanced input controls to the calculator."""
        calc.input.add_simple_keywords(*simple_keywords)
        if self.maxcore is not None:
            calc.input.memory = self.maxcore
        for raw_input in self.additional_arbitrary_strings:
            calc.input.add_arbitrary_string(raw_input)
        calc.input.ncores = self.cores

    def _set_structure_charge_and_spin(
        self,
        calc: Calculator,
        default_charge: int,
        default_spin_multiplicity: int | None,
    ) -> None:
        """Set charge and multiplicity with calculator-level overrides when provided."""
        calc.charge = int(self.charge) if self.charge is not None else int(default_charge)  # type: ignore[attr-defined]
        calc.multiplicity = (  # type: ignore[attr-defined]
            int(self.spin_multiplicity)
            if self.spin_multiplicity is not None
            else int(default_spin_multiplicity)
            if default_spin_multiplicity is not None
            else 1
        )

    def _parse_output(self, output: Output) -> ORCAProperties:
        """Extract molecular properties from ORCA output.

        Parses the OPI Output object to extract computed molecular properties
        including total energy and solvation energy. This method accesses the
        nested results structure provided by OPI.

        Args:
            output: OPI Output object containing calculation results.

        Returns:
            Properties object with system-level properties extracted from the output.

        Raises:
            AttributeError: If expected output structure is not present.

        """
        # Extract geometry and energies from the complex nested structure
        output.parse()
        geometry = output.results_properties.geometries[0]  # type: ignore[attr-defined]
        total_energy = geometry.energy[0].totalenergy[0][0]  # type: ignore[attr-defined]
        if self.solvation_model is not None:
            solvation_energy = geometry.solvation_details.cpcmdielenergy  # type: ignore[attr-defined]
        else:
            solvation_energy = None

        # Construct system property list
        total_energy = SystemProperty(
            name="Total Energy",
            value=float(total_energy) * ureg.hartree,
            description="Total electronic energy from ORCA calculation",
        )
        properties = ORCASystemProperties(total_energy=total_energy)
        if self.solvation_model is not None:
            solvation_energy = SystemProperty(
                name="Solvation Energy",
                value=(float(solvation_energy) if solvation_energy is not None else 0.0)
                * ureg.hartree,
                description=f"Solvation energy contribution from  \
                {self.solvation_model} {self.solvent} model with {self.solvation} solvation model",
            )
            properties.solvation_energy = solvation_energy

        homo_pr_property = None
        lumo_pr_property = None
        atomic_properties = ORCAAtomicProperties(
            homo_participation_ratio=homo_pr_property,
            lumo_participation_ratio=lumo_pr_property,
        )
        return ORCAProperties(system=properties, atomic=atomic_properties)
