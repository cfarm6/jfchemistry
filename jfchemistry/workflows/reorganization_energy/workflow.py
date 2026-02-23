"""Reorganization energy workflow based on Nelson's four-point method."""

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


class ReorganizationEnergySystemProperty(PropertyClass):
    """Reorganization energy terms."""

    reorganization_energy: SystemProperty | OutputReference
    donor_reorganization_energy: SystemProperty | OutputReference
    acceptor_reorganization_energy: SystemProperty | OutputReference


class ReorganizationEnergyProperties(Properties):
    """Properties for reorganization energy workflow."""

    system: ReorganizationEnergySystemProperty


class ReorganizationEnergyOutput(Output):
    """Output of the reorganization energy workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: list[Molecule]
    files: Optional[Any] = None
    properties: Optional[ReorganizationEnergyProperties] = None


@dataclass
class ReorganizationEnergyCalculation(PymatGenMaker):
    """Compute donor/acceptor and total reorganization energies."""

    name: str = "Reorganization Energy Calculation"
    _properties_model: type[ReorganizationEnergyProperties] = ReorganizationEnergyProperties
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
    def _calculate_reorganization_energies(
        cls,
        donor_cross_properties: Properties,
        donor_relaxed_final_properties: Properties,
        acceptor_cross_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
    ) -> tuple[float, float, float]:
        donor_cross = cls._extract_total_energy_ev(donor_cross_properties, "donor_cross_properties")
        donor_relaxed_final = cls._extract_total_energy_ev(
            donor_relaxed_final_properties, "donor_relaxed_final_properties"
        )
        acceptor_cross = cls._extract_total_energy_ev(
            acceptor_cross_properties, "acceptor_cross_properties"
        )
        acceptor_relaxed_final = cls._extract_total_energy_ev(
            acceptor_relaxed_final_properties, "acceptor_relaxed_final_properties"
        )

        donor_reorganization_energy = donor_cross - donor_relaxed_final
        acceptor_reorganization_energy = acceptor_cross - acceptor_relaxed_final
        total_reorganization_energy = donor_reorganization_energy + acceptor_reorganization_energy
        return (
            donor_reorganization_energy,
            acceptor_reorganization_energy,
            total_reorganization_energy,
        )

    @jfchem_job()
    def make(
        self,
        donor_cross_properties: Properties,
        donor_relaxed_final_properties: Properties,
        acceptor_cross_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
    ) -> Response[_output_model]:
        """Calculate reorganization energy from four-point properties."""
        donor_term, acceptor_term, total_term = self._calculate_reorganization_energies(
            donor_cross_properties=donor_cross_properties,
            donor_relaxed_final_properties=donor_relaxed_final_properties,
            acceptor_cross_properties=acceptor_cross_properties,
            acceptor_relaxed_final_properties=acceptor_relaxed_final_properties,
        )
        return Response(
            output=Output(
                properties=self._properties_model(
                    system=ReorganizationEnergySystemProperty(
                        reorganization_energy=SystemProperty(
                            name="Reorganization Energy",
                            value=total_term * ureg.eV,
                        ),
                        donor_reorganization_energy=SystemProperty(
                            name="Donor Reorganization Energy",
                            value=donor_term * ureg.eV,
                        ),
                        acceptor_reorganization_energy=SystemProperty(
                            name="Acceptor Reorganization Energy",
                            value=acceptor_term * ureg.eV,
                        ),
                    )
                )
            )
        )


@dataclass
class ReorganizationEnergyWorkflow(PymatGenMaker):
    """Wrapper over NelsonsFourPointMethod exposing only reorganization terms."""

    name: str = "Reorganization Energy Workflow"
    optimizer: PymatGenMaker | None = None
    donor_initial_charge: int | None = None
    donor_initial_spin_multiplicity: int | None = None
    donor_final_charge: int | None = None
    donor_final_spin_multiplicity: int | None = None
    acceptor_initial_charge: int | None = None
    acceptor_initial_spin_multiplicity: int | None = None
    acceptor_final_charge: int | None = None
    acceptor_final_spin_multiplicity: int | None = None
    _properties_model: type[ReorganizationEnergyProperties] = ReorganizationEnergyProperties
    _output_model: type[ReorganizationEnergyOutput] = ReorganizationEnergyOutput

    def _build_nelsons_four_point_method(self) -> NelsonsFourPointMethod:
        if self.optimizer is None:
            raise ValueError("ReorganizationEnergyWorkflow requires an `optimizer` attribute.")
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
        self, donor: Molecule, acceptor: Molecule
    ) -> tuple[Flow, ReorganizationEnergyOutput]:
        flow, combined_output = self._build_nelsons_four_point_method()._build_flow(donor, acceptor)
        system = combined_output.properties.system
        properties = self._properties_model(
            system=ReorganizationEnergySystemProperty(
                reorganization_energy=system.reorganization_energy,
                donor_reorganization_energy=system.donor_reorganization_energy,
                acceptor_reorganization_energy=system.acceptor_reorganization_energy,
            )
        )
        output = ReorganizationEnergyOutput(
            structure=[donor, acceptor],
            properties=properties,
            files=combined_output.files,
        )
        return flow, output

    @jfchem_job()
    def make(self, donor: Molecule, acceptor: Molecule) -> Response[_output_model]:
        """Create the reorganization energy workflow."""
        flow, output = self._build_flow(donor, acceptor)
        return Response(output=output, detour=flow)
