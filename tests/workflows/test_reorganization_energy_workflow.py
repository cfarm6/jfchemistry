"""Tests for the reorganization energy workflow."""

from dataclasses import dataclass, field

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.nelsons_four_point_method.workflow import NelsonsFourPointMethod
from jfchemistry.workflows.reorganization_energy.workflow import (
    ReorganizationEnergyCalculation,
    ReorganizationEnergyWorkflow,
)

EXPECTED_JOB_COUNT = 7


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


@dataclass
class _TrackingNelsons(NelsonsFourPointMethod):
    calls: list[tuple[str, str, int, int, bool]] = field(default_factory=list)

    def _get_optimizer(
        self, role: str, state: str, charge: int, spin_multiplicity: int, relax_geometry: bool
    ):
        self.calls.append((role, state, charge, spin_multiplicity, relax_geometry))
        return _ConstantEnergyMaker(energy_ev=0.0)


@dataclass
class _TrackingWorkflow(ReorganizationEnergyWorkflow):
    calls: list[tuple[str, str, int, int, bool]] = field(default_factory=list)

    def _build_nelsons_four_point_method(self) -> NelsonsFourPointMethod:
        if self.optimizer is None:
            raise ValueError("ReorganizationEnergyWorkflow requires an `optimizer` attribute.")
        return _TrackingNelsons(
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
            calls=self.calls,
        )


@dataclass
class _ConstantEnergyMaker(PymatGenMaker[Molecule, Molecule]):
    energy_ev: float = 0.0
    _properties_model: type[_EnergyProperties] = _EnergyProperties

    def _operation(self, input: Molecule, **kwargs):
        properties = _EnergyProperties(
            system=_SystemProperties(
                total_energy=SystemProperty(name="Total Energy", value=self.energy_ev * ureg.eV)
            )
        )
        return input, properties


def _single_hydrogen_molecule(charge: int, spin: int) -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=charge,
        spin_multiplicity=spin,
        charge_spin_check=False,
    )


def _energy_properties(energy_ev: float) -> _EnergyProperties:
    return _EnergyProperties(
        system=_SystemProperties(
            total_energy=SystemProperty(name="Total Energy", value=energy_ev * ureg.eV)
        )
    )


def test_workflow_dispatches_expected_six_jobs_plus_reducer() -> None:
    """Workflow should dispatch six factory jobs and one reducer job."""
    workflow = _TrackingWorkflow(optimizer=_ConstantEnergyMaker(energy_ev=0.0))
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)

    flow, _ = workflow._build_flow(donor, acceptor)

    assert flow.name == "Reorganization Energy Workflow"
    assert len(flow.jobs) == EXPECTED_JOB_COUNT
    assert [(role, state, relax) for role, state, _, _, relax in workflow.calls] == [
        ("donor", "initial", True),
        ("donor", "final", True),
        ("donor", "final", False),
        ("acceptor", "initial", True),
        ("acceptor", "final", True),
        ("acceptor", "final", False),
    ]


def test_energy_reduction_charged_state_formula() -> None:
    """Charged-state Nelson expression should match expected terms."""
    donor_term, acceptor_term, total_term = (
        ReorganizationEnergyCalculation._calculate_reorganization_energies(
            donor_cross_properties=_energy_properties(1.5),
            donor_relaxed_final_properties=_energy_properties(1.0),
            acceptor_cross_properties=_energy_properties(-0.2),
            acceptor_relaxed_final_properties=_energy_properties(-0.6),
        )
    )

    assert donor_term == pytest.approx(0.5)
    assert acceptor_term == pytest.approx(0.4)
    assert total_term == pytest.approx(0.9)


def test_default_state_resolution_from_input_and_charge_rules() -> None:
    """Default charge/spin resolution should follow donor/acceptor rules."""
    workflow = _TrackingWorkflow(optimizer=_ConstantEnergyMaker(energy_ev=0.0))
    donor = _single_hydrogen_molecule(charge=2, spin=3)
    acceptor = _single_hydrogen_molecule(charge=-1, spin=2)

    workflow._build_flow(donor, acceptor)

    assert workflow.calls == [
        ("donor", "initial", 2, 3, True),
        ("donor", "final", 3, 3, True),
        ("donor", "final", 3, 3, False),
        ("acceptor", "initial", -1, 2, True),
        ("acceptor", "final", -2, 2, True),
        ("acceptor", "final", -2, 2, False),
    ]


def test_output_contains_all_three_properties() -> None:
    """Workflow output should expose total, donor, and acceptor properties."""
    workflow = ReorganizationEnergyWorkflow(optimizer=_ConstantEnergyMaker(energy_ev=0.0))
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)

    _, output = workflow._build_flow(donor, acceptor)

    assert output.properties is not None
    system = output.properties.system
    assert system.reorganization_energy is not None
    assert system.donor_reorganization_energy is not None
    assert system.acceptor_reorganization_energy is not None


def test_missing_total_energy_raises_clear_error() -> None:
    """Reducer should raise a clear error when total energy is missing."""

    class _MissingEnergyProperties(Properties):
        system: PropertyClass

    missing = _MissingEnergyProperties(system=PropertyClass())
    valid = _energy_properties(0.0)

    with pytest.raises(
        ValueError, match=r"Missing system\.total_energy for donor_cross_properties\."
    ):
        ReorganizationEnergyCalculation._calculate_reorganization_energies(
            donor_cross_properties=missing,
            donor_relaxed_final_properties=valid,
            acceptor_cross_properties=valid,
            acceptor_relaxed_final_properties=valid,
        )


def test_missing_optimizer_raises_clear_error() -> None:
    """Workflow should require optimizer as an attribute."""
    workflow = ReorganizationEnergyWorkflow()
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)
    with pytest.raises(
        ValueError, match=r"ReorganizationEnergyWorkflow requires an `optimizer` attribute\."
    ):
        workflow._build_flow(donor, acceptor)
