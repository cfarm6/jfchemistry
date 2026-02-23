"""Combined Nelson's four-point workflow for reorganization and free energy terms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

from jobflow.core.flow import Flow
from jobflow.core.job import OutputReference, Response
from pydantic import ConfigDict

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

RoleType = Literal["donor", "acceptor"]
StateType = Literal["initial", "final"]


class NelsonsFourPointSystemProperty(PropertyClass):
    """Combined four-point properties for donor-acceptor electron transfer."""

    reorganization_energy: SystemProperty | OutputReference
    donor_reorganization_energy: SystemProperty | OutputReference
    acceptor_reorganization_energy: SystemProperty | OutputReference
    free_energy_difference: SystemProperty | OutputReference
    donor_free_energy_difference: SystemProperty | OutputReference
    acceptor_free_energy_difference: SystemProperty | OutputReference


class NelsonsFourPointProperties(Properties):
    """Properties for the combined Nelson's four-point workflow."""

    system: NelsonsFourPointSystemProperty


class NelsonsFourPointOutput(Output):
    """Output of the Nelsons Four Point Method."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: list[Molecule]
    files: Optional[Any] = None
    properties: NelsonsFourPointProperties


@dataclass
class NelsonsFourPointCalculation(PymatGenMaker):
    """Compute reorganization and free energy difference terms from four-point energies."""

    name: str = "Nelsons Four Point Calculation"
    _properties_model: type[NelsonsFourPointProperties] = NelsonsFourPointProperties
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
    def _calculate_terms(  # noqa: PLR0913
        cls,
        donor_relaxed_initial_properties: Properties,
        donor_relaxed_final_properties: Properties,
        donor_cross_properties: Properties,
        acceptor_relaxed_initial_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
        acceptor_cross_properties: Properties,
    ) -> tuple[float, float, float, float, float, float]:
        donor_initial = cls._extract_total_energy_ev(
            donor_relaxed_initial_properties, "donor_relaxed_initial_properties"
        )
        donor_final = cls._extract_total_energy_ev(
            donor_relaxed_final_properties, "donor_relaxed_final_properties"
        )
        donor_cross = cls._extract_total_energy_ev(donor_cross_properties, "donor_cross_properties")
        acceptor_initial = cls._extract_total_energy_ev(
            acceptor_relaxed_initial_properties, "acceptor_relaxed_initial_properties"
        )
        acceptor_final = cls._extract_total_energy_ev(
            acceptor_relaxed_final_properties, "acceptor_relaxed_final_properties"
        )
        acceptor_cross = cls._extract_total_energy_ev(
            acceptor_cross_properties, "acceptor_cross_properties"
        )

        donor_reorganization_energy = donor_cross - donor_final
        acceptor_reorganization_energy = acceptor_cross - acceptor_final
        reorganization_energy = donor_reorganization_energy + acceptor_reorganization_energy

        donor_free_energy_difference = donor_final - donor_initial
        acceptor_free_energy_difference = acceptor_final - acceptor_initial
        free_energy_difference = donor_free_energy_difference + acceptor_free_energy_difference

        return (
            donor_reorganization_energy,
            acceptor_reorganization_energy,
            reorganization_energy,
            donor_free_energy_difference,
            acceptor_free_energy_difference,
            free_energy_difference,
        )

    @jfchem_job()
    def make(  # noqa: PLR0913
        self,
        donor_relaxed_initial_properties: Properties,
        donor_relaxed_final_properties: Properties,
        donor_cross_properties: Properties,
        acceptor_relaxed_initial_properties: Properties,
        acceptor_relaxed_final_properties: Properties,
        acceptor_cross_properties: Properties,
    ) -> Response[_output_model]:
        """Calculate all Nelson four-point terms from six required energies."""
        (
            donor_reorg,
            acceptor_reorg,
            total_reorg,
            donor_dg,
            acceptor_dg,
            total_dg,
        ) = self._calculate_terms(
            donor_relaxed_initial_properties=donor_relaxed_initial_properties,
            donor_relaxed_final_properties=donor_relaxed_final_properties,
            donor_cross_properties=donor_cross_properties,
            acceptor_relaxed_initial_properties=acceptor_relaxed_initial_properties,
            acceptor_relaxed_final_properties=acceptor_relaxed_final_properties,
            acceptor_cross_properties=acceptor_cross_properties,
        )
        return Response(
            output=Output(
                properties=self._properties_model(
                    system=NelsonsFourPointSystemProperty(
                        reorganization_energy=SystemProperty(
                            name="Reorganization Energy",
                            value=total_reorg * ureg.eV,
                        ),
                        donor_reorganization_energy=SystemProperty(
                            name="Donor Reorganization Energy",
                            value=donor_reorg * ureg.eV,
                        ),
                        acceptor_reorganization_energy=SystemProperty(
                            name="Acceptor Reorganization Energy",
                            value=acceptor_reorg * ureg.eV,
                        ),
                        free_energy_difference=SystemProperty(
                            name="Free Energy Difference",
                            value=total_dg * ureg.eV,
                        ),
                        donor_free_energy_difference=SystemProperty(
                            name="Donor Free Energy Difference",
                            value=donor_dg * ureg.eV,
                        ),
                        acceptor_free_energy_difference=SystemProperty(
                            name="Acceptor Free Energy Difference",
                            value=acceptor_dg * ureg.eV,
                        ),
                    )
                )
            )
        )


@dataclass
class NelsonsFourPointMethod(PymatGenMaker):
    """Run Nelson's four-point workflow for donor/acceptor and return all terms."""

    name: str = "Nelsons Four Point Method"
    optimizer: PymatGenMaker | None = None
    donor_initial_charge: int | None = None
    donor_initial_spin_multiplicity: int | None = None
    donor_final_charge: int | None = None
    donor_final_spin_multiplicity: int | None = None
    acceptor_initial_charge: int | None = None
    acceptor_initial_spin_multiplicity: int | None = None
    acceptor_final_charge: int | None = None
    acceptor_final_spin_multiplicity: int | None = None
    _properties_model: type[NelsonsFourPointProperties] = NelsonsFourPointProperties
    _output_model: type[Output] = Output

    def _resolve_state(self, molecule: Molecule, role: RoleType) -> tuple[int, int, int, int]:
        base_charge = int(molecule.charge)
        base_spin = int(molecule.spin_multiplicity) if molecule.spin_multiplicity is not None else 1

        if role == "donor":
            initial_charge = (
                int(self.donor_initial_charge)
                if self.donor_initial_charge is not None
                else base_charge
            )
            initial_spin = (
                int(self.donor_initial_spin_multiplicity)
                if self.donor_initial_spin_multiplicity is not None
                else base_spin
            )
            final_charge = (
                int(self.donor_final_charge)
                if self.donor_final_charge is not None
                else initial_charge + 1
            )
            final_spin = (
                int(self.donor_final_spin_multiplicity)
                if self.donor_final_spin_multiplicity is not None
                else initial_spin
            )
            return initial_charge, initial_spin, final_charge, final_spin

        initial_charge = (
            int(self.acceptor_initial_charge)
            if self.acceptor_initial_charge is not None
            else base_charge
        )
        initial_spin = (
            int(self.acceptor_initial_spin_multiplicity)
            if self.acceptor_initial_spin_multiplicity is not None
            else base_spin
        )
        final_charge = (
            int(self.acceptor_final_charge)
            if self.acceptor_final_charge is not None
            else initial_charge - 1
        )
        final_spin = (
            int(self.acceptor_final_spin_multiplicity)
            if self.acceptor_final_spin_multiplicity is not None
            else initial_spin
        )
        return initial_charge, initial_spin, final_charge, final_spin

    @staticmethod
    def _build_state_molecule(molecule: Molecule, charge: int, spin_multiplicity: int) -> Molecule:
        state_molecule = molecule.copy()
        if hasattr(state_molecule, "_charge_spin_check"):
            state_molecule._charge_spin_check = False
        updated = state_molecule.set_charge_and_spin(
            charge=charge, spin_multiplicity=spin_multiplicity
        )
        if updated is not None:
            state_molecule = updated
        return state_molecule

    def _get_optimizer(
        self,
        role: RoleType,
        state: StateType,
        charge: int,
        spin_multiplicity: int,
        relax_geometry: bool,
    ) -> PymatGenMaker:
        if self.optimizer is None:
            raise ValueError("NelsonsFourPointMethod requires an `optimizer` attribute.")
        maker = deepcopy(self.optimizer)
        if not isinstance(maker, PymatGenMaker):
            raise TypeError("`optimizer` must be a PymatGenMaker instance.")

        calculator = getattr(maker, "calculator", None)
        if calculator is not None:
            if hasattr(calculator, "charge"):
                calculator.charge = charge
            if hasattr(calculator, "spin_multiplicity"):
                calculator.spin_multiplicity = spin_multiplicity

        if hasattr(maker, "charge"):
            maker.charge = charge  # type: ignore[attr-defined]
        if hasattr(maker, "spin_multiplicity"):
            maker.spin_multiplicity = spin_multiplicity  # type: ignore[attr-defined]

        if (not relax_geometry) and hasattr(maker, "steps"):
            maker.steps = 0  # type: ignore[attr-defined]

        if hasattr(maker, "name"):
            maker.name = f"{maker.name} ({role}-{state}, relax={relax_geometry})"

        return maker

    def _build_flow(
        self, donor: Molecule, acceptor: Molecule
    ) -> tuple[Flow, NelsonsFourPointOutput]:
        donor_initial_charge, donor_initial_spin, donor_final_charge, donor_final_spin = (
            self._resolve_state(donor, "donor")
        )
        (
            acceptor_initial_charge,
            acceptor_initial_spin,
            acceptor_final_charge,
            acceptor_final_spin,
        ) = self._resolve_state(acceptor, "acceptor")

        donor_initial = self._build_state_molecule(donor, donor_initial_charge, donor_initial_spin)
        donor_final = self._build_state_molecule(donor, donor_final_charge, donor_final_spin)
        acceptor_initial = self._build_state_molecule(
            acceptor, acceptor_initial_charge, acceptor_initial_spin
        )
        acceptor_final = self._build_state_molecule(
            acceptor, acceptor_final_charge, acceptor_final_spin
        )

        donor_relaxed_initial_job = self._get_optimizer(
            "donor", "initial", donor_initial_charge, donor_initial_spin, True
        ).make(donor_initial)
        donor_relaxed_final_job = self._get_optimizer(
            "donor", "final", donor_final_charge, donor_final_spin, True
        ).make(donor_final)
        donor_cross_job = self._get_optimizer(
            "donor", "final", donor_final_charge, donor_final_spin, False
        ).make(donor_relaxed_initial_job.output.structure)

        acceptor_relaxed_initial_job = self._get_optimizer(
            "acceptor", "initial", acceptor_initial_charge, acceptor_initial_spin, True
        ).make(acceptor_initial)
        acceptor_relaxed_final_job = self._get_optimizer(
            "acceptor", "final", acceptor_final_charge, acceptor_final_spin, True
        ).make(acceptor_final)
        acceptor_cross_job = self._get_optimizer(
            "acceptor", "final", acceptor_final_charge, acceptor_final_spin, False
        ).make(acceptor_relaxed_initial_job.output.structure)

        reducer = NelsonsFourPointCalculation()
        final_job = reducer.make(
            donor_relaxed_initial_properties=donor_relaxed_initial_job.output.properties,
            donor_relaxed_final_properties=donor_relaxed_final_job.output.properties,
            donor_cross_properties=donor_cross_job.output.properties,
            acceptor_relaxed_initial_properties=acceptor_relaxed_initial_job.output.properties,
            acceptor_relaxed_final_properties=acceptor_relaxed_final_job.output.properties,
            acceptor_cross_properties=acceptor_cross_job.output.properties,
        )
        flow = Flow(
            [
                donor_relaxed_initial_job,
                donor_relaxed_final_job,
                donor_cross_job,
                acceptor_relaxed_initial_job,
                acceptor_relaxed_final_job,
                acceptor_cross_job,
                final_job,
            ],
            name=self.name,
        )
        properties = self._properties_model(
            system=NelsonsFourPointSystemProperty(
                reorganization_energy=final_job.output.properties.system.reorganization_energy,
                donor_reorganization_energy=(
                    final_job.output.properties.system.donor_reorganization_energy
                ),
                acceptor_reorganization_energy=(
                    final_job.output.properties.system.acceptor_reorganization_energy
                ),
                free_energy_difference=final_job.output.properties.system.free_energy_difference,
                donor_free_energy_difference=(
                    final_job.output.properties.system.donor_free_energy_difference
                ),
                acceptor_free_energy_difference=(
                    final_job.output.properties.system.acceptor_free_energy_difference
                ),
            )
        )
        output = NelsonsFourPointOutput(
            structure=[donor, acceptor],
            properties=properties,
            files=[donor_relaxed_final_job.output.files, acceptor_relaxed_final_job.output.files],
        )
        return flow, output

    @jfchem_job()
    def make(self, donor: Molecule, acceptor: Molecule) -> Response[_output_model]:
        """Create the combined Nelson four-point workflow."""
        flow, output = self._build_flow(donor, acceptor)
        return Response(output=output, detour=flow)
