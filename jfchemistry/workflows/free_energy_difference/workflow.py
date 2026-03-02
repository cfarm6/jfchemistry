"""Free energy difference workflow based on Nelson's four-point method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from jobflow.core.job import OutputReference, Response
from pydantic import ConfigDict

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.nelsons_four_point_method.workflow import NelsonsFourPointMethod

if TYPE_CHECKING:
    from jobflow.core.flow import Flow
    from pymatgen.core.structure import Molecule


class FreeEnergyDifferenceSystemProperty(PropertyClass):
    """Free energy difference terms."""

    free_energy_difference: SystemProperty | OutputReference
    donor_free_energy_difference: SystemProperty | OutputReference
    acceptor_free_energy_difference: SystemProperty | OutputReference


class FreeEnergyDifferenceProperties(Properties):
    """Properties for free energy difference workflow."""

    system: FreeEnergyDifferenceSystemProperty


class FreeEnergyDifferenceOutput(Output):
    """Output of the free energy difference workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: list[Any]
    files: Optional[Any] = None
    properties: FreeEnergyDifferenceProperties


@dataclass
class FreeEnergyDifferenceCalculation(PymatGenMaker):
    """Compute donor/acceptor and total free energy differences."""

    name: str = "Free Energy Difference Calculation"
    _properties_model: type[FreeEnergyDifferenceProperties] = FreeEnergyDifferenceProperties
    _output_model: type[Output] = Output

    @staticmethod
    def _extract_total_energy_ev(properties: Properties, label: str) -> float:
        _properties = Properties.model_validate(properties, extra="allow", strict=False)
        if (
            _properties.system is None
            or not hasattr(_properties.system, "total_energy")
            or (te := getattr(_properties.system, "total_energy", None)) is None
            or getattr(te, "value", None) is None
        ):
            raise ValueError(f"Missing system.total_energy for {label}.")
        energy = te.value
        if hasattr(energy, "to"):
            return float(energy.to(ureg.eV).magnitude)
        if isinstance(energy, (int, float)):
            return float(energy)
        raise ValueError(f"Could not parse system.total_energy for {label}.")

    @classmethod
    def _calculate_free_energy_differences(
        cls,
        donor_relaxed_initial_properties: Properties,
        donor_relaxed_final_properties: Properties,
        acceptor_relaxed_initial_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
    ) -> tuple[float, float, float]:
        donor_initial = cls._extract_total_energy_ev(
            donor_relaxed_initial_properties, "donor_relaxed_initial_properties"
        )
        donor_final = cls._extract_total_energy_ev(
            donor_relaxed_final_properties, "donor_relaxed_final_properties"
        )
        acceptor_initial = cls._extract_total_energy_ev(
            acceptor_relaxed_initial_properties, "acceptor_relaxed_initial_properties"
        )
        acceptor_final = cls._extract_total_energy_ev(
            acceptor_relaxed_final_properties, "acceptor_relaxed_final_properties"
        )

        donor_free_energy_difference = donor_final - donor_initial
        acceptor_free_energy_difference = acceptor_final - acceptor_initial
        total_free_energy_difference = (
            donor_free_energy_difference + acceptor_free_energy_difference
        )
        return (
            donor_free_energy_difference,
            acceptor_free_energy_difference,
            total_free_energy_difference,
        )

    @jfchem_job()
    def make(
        self,
        donor_relaxed_initial_properties: Properties,
        donor_relaxed_final_properties: Properties,
        acceptor_relaxed_initial_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
    ) -> Response[_output_model]:
        """Calculate free energy difference from four-point properties."""
        donor_term, acceptor_term, total_term = self._calculate_free_energy_differences(
            donor_relaxed_initial_properties=donor_relaxed_initial_properties,
            donor_relaxed_final_properties=donor_relaxed_final_properties,
            acceptor_relaxed_initial_properties=acceptor_relaxed_initial_properties,
            acceptor_relaxed_final_properties=acceptor_relaxed_final_properties,
        )
        return Response(
            output=Output(
                properties=self._properties_model(
                    system=FreeEnergyDifferenceSystemProperty(
                        free_energy_difference=SystemProperty(
                            name="Free Energy Difference",
                            value=total_term * ureg.eV,
                        ),
                        donor_free_energy_difference=SystemProperty(
                            name="Donor Free Energy Difference",
                            value=donor_term * ureg.eV,
                        ),
                        acceptor_free_energy_difference=SystemProperty(
                            name="Acceptor Free Energy Difference",
                            value=acceptor_term * ureg.eV,
                        ),
                    )
                )
            )
        )


@dataclass
class FreeEnergyDifferenceWorkflow(PymatGenMaker):
    """Wrapper over NelsonsFourPointMethod exposing only free-energy terms."""

    name: str = "Free Energy Difference Workflow"
    optimizer: PymatGenMaker | None = None
    donor_initial_charge: int | None = None
    donor_initial_spin_multiplicity: int | None = None
    donor_final_charge: int | None = None
    donor_final_spin_multiplicity: int | None = None
    acceptor_initial_charge: int | None = None
    acceptor_initial_spin_multiplicity: int | None = None
    acceptor_final_charge: int | None = None
    acceptor_final_spin_multiplicity: int | None = None
    _properties_model: type[FreeEnergyDifferenceProperties] = FreeEnergyDifferenceProperties
    _output_model: type[FreeEnergyDifferenceOutput] = FreeEnergyDifferenceOutput

    def _build_nelsons_four_point_method(self) -> NelsonsFourPointMethod:
        if self.optimizer is None:
            raise ValueError("FreeEnergyDifferenceWorkflow requires an `optimizer` attribute.")
        return NelsonsFourPointMethod(
            name=self.name,
            optimizer=self.optimizer,
            donor_initial_charge=self.donor_initial_charge,
            donor_initial_spin_multiplicity=self.donor_initial_spin_multiplicity,
            donor_final_charge=self.donor_final_charge,
            donor_final_spin_multiplicity=self.donor_final_spin_multiplicity,
            acceptor_initial_charge=self.acceptor_initial_charge,
            acceptor_initial_spin_multiplicity=self.acceptor_initial_spin_multiplicity,
            acceptor_final_charge=self.acceptor_final_charge,
            acceptor_final_spin_multiplicity=self.acceptor_final_spin_multiplicity,
        )

    def _build_flow(
        self,
        donor: Molecule,
        acceptor: Molecule,
    ) -> tuple[Flow, FreeEnergyDifferenceOutput]:
        flow, combined_output = self._build_nelsons_four_point_method()._build_flow(donor, acceptor)
        system = combined_output.properties.system
        properties = self._properties_model(
            system=FreeEnergyDifferenceSystemProperty(
                free_energy_difference=system.free_energy_difference,
                donor_free_energy_difference=system.donor_free_energy_difference,
                acceptor_free_energy_difference=system.acceptor_free_energy_difference,
            )
        )
        output = FreeEnergyDifferenceOutput(
            structure=[donor, acceptor],
            properties=properties,
            files=combined_output.files,
        )
        return flow, output

    @jfchem_job()
    def make(self, donor: Molecule, acceptor: Molecule) -> Response[_output_model]:
        """Create the free energy difference workflow."""
        flow, output = self._build_flow(donor, acceptor)
        return Response(output=output, detour=flow)
