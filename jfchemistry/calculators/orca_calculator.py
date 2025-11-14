"""Base Class for ORCA DFT Calculations."""

from dataclasses import dataclass, field
from typing import Optional

from opi.input.simple_keywords.base import SimpleKeyword
from opi.input.simple_keywords.basis_set import BasisSet
from opi.input.simple_keywords.dft import Dft
from opi.input.simple_keywords.dispersion_correction import DispersionCorrection
from opi.input.simple_keywords.ecp import Ecp
from opi.input.simple_keywords.solvation import Solvation
from opi.input.simple_keywords.solvation_model import SolvationModel
from opi.input.simple_keywords.solvent import Solvent
from opi.output.core import Output
from pydantic import BaseModel

from jfchemistry.base_classes import SystemProperty

# Import fully typed Literal definitions
from jfchemistry.calculators.orca_keywords import (
    BasisSetType,
    DispersionCorrectionType,
    ECPType,
    SolvationModelType,
    SolvationType,
    SolventType,
    XCFunctionalType,
)

from .base import WavefunctionCalculator

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


class ORCASystemProperties(BaseModel):
    """System properties of the ORCA calculation."""

    total_energy: SystemProperty
    solvation_energy: Optional[SystemProperty] = None


class ORCAProperties(BaseModel):
    """Properties of the ORCA calculation."""

    system: ORCASystemProperties


@dataclass
class ORCACalculator(WavefunctionCalculator):
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
        basis_set: Basis set to use for the calculation (488 options available).
        xc_functional: Exchange-correlation functional to use (195 options available).
        ecp: Effective core potential to use for heavy atoms (13 options available).
        solvation: Solvation model type to use (3 options available).
        solvent: Solvent specification for solvation calculations (236 options available).
        solvation_model: Specific solvation model implementation (2 options available).
        charge: Molecular charge override. If None, uses charge from structure.
        spin_multiplicity: Spin multiplicity override. If None, uses spin from structure.

    Examples:
        >>> from jfchemistry.calculators import ORCACalculator
        >>>
        >>> # Create a simple calculator with a basis set and functional
        >>> calc = ORCACalculator(
        ...     basis_set="DEF2_SVP",  # IDE will autocomplete all 488 options
        ...     xc_functional="B3LYP",  # IDE will autocomplete all 195 options
        ...     cores=4
        ... )
        >>>
        >>> # Set up keywords
        >>> keywords = calc.set_keywords()
        >>> print(len(keywords))
        2
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
    _properties_model: type[ORCAProperties] = ORCAProperties

    def set_keywords(self) -> list[SimpleKeyword]:
        """Construct OPI simple keywords from calculator settings.

        Builds a list of SimpleKeyword objects from the configured calculator
        parameters. Handles both simple keywords (basis set, xc functional, etc.)
        and complex keywords (solvation models with solvents).

        Returns:
            List of SimpleKeyword objects ready for OPI input generation.

        Examples:
            >>> calc = ORCACalculator(basis_set="DEF2_SVP", xc_functional="B3LYP")
            >>> keywords = calc.set_keywords()
            >>> len(keywords)
            2
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

    def parse_output(self, output: Output) -> ORCAProperties:
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
            value=float(total_energy),
            units="Hartree",
            description="Total electronic energy from ORCA calculation",
        )
        properties = ORCASystemProperties(total_energy=total_energy)
        if self.solvation_model is not None:
            solvation_energy = SystemProperty(
                name="Solvation Energy",
                value=float(solvation_energy),
                units="Hartree",
                description=f"Solvation energy contribution from  \
                {self.solvation_model} {self.solvent} model with {self.solvation} solvation model",
            )
            properties.solvation_energy = solvation_energy

        return ORCAProperties(system=properties)
