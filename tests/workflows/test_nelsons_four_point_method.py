"""Tests for the combined Nelson's four-point workflow."""

from dataclasses import dataclass, field

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.nelsons_four_point_method.workflow import (
    NelsonsFourPointCalculation,
    NelsonsFourPointMethod,
)

EXPECTED_JOB_COUNT = 7


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


@dataclass
class _TrackingWorkflow(NelsonsFourPointMethod):
    calls: list[tuple[str, str, int, int, bool]] = field(default_factory=list)

    def _get_optimizer(
        self, role: str, state: str, charge: int, spin_multiplicity: int, relax_geometry: bool
    ):
        self.calls.append((role, state, charge, spin_multiplicity, relax_geometry))
        return _ConstantEnergyMaker(energy_ev=0.0)


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
    """Workflow should dispatch six optimizer jobs and one reducer job."""
    workflow = _TrackingWorkflow(optimizer=_ConstantEnergyMaker(energy_ev=0.0))
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)

    flow, _ = workflow._build_flow(donor, acceptor)

    assert flow.name == "Nelsons Four Point Method"
    assert len(flow.jobs) == EXPECTED_JOB_COUNT
    assert [(role, state, relax) for role, state, _, _, relax in workflow.calls] == [
        ("donor", "initial", True),
        ("donor", "final", True),
        ("donor", "final", False),
        ("acceptor", "initial", True),
        ("acceptor", "final", True),
        ("acceptor", "final", False),
    ]


def test_reducer_calculates_reorg_and_free_energy_terms() -> None:
    """Reducer should return both reorganization and free-energy terms."""
    terms = NelsonsFourPointCalculation._calculate_terms(
        donor_relaxed_initial_properties=_energy_properties(1.0),
        donor_relaxed_final_properties=_energy_properties(1.4),
        donor_cross_properties=_energy_properties(1.8),
        acceptor_relaxed_initial_properties=_energy_properties(-0.5),
        acceptor_relaxed_final_properties=_energy_properties(-0.2),
        acceptor_cross_properties=_energy_properties(0.1),
    )

    donor_reorg, acceptor_reorg, total_reorg, donor_dg, acceptor_dg, total_dg = terms

    assert donor_reorg == pytest.approx(0.4)
    assert acceptor_reorg == pytest.approx(0.3)
    assert total_reorg == pytest.approx(0.7)
    assert donor_dg == pytest.approx(0.4)
    assert acceptor_dg == pytest.approx(0.3)
    assert total_dg == pytest.approx(0.7)


def test_output_contains_all_six_terms() -> None:
    """Workflow output should expose combined total and component terms."""
    workflow = NelsonsFourPointMethod(optimizer=_ConstantEnergyMaker(energy_ev=0.0))
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)

    _, output = workflow._build_flow(donor, acceptor)
    system = output.properties.system

    assert system.reorganization_energy is not None
    assert system.donor_reorganization_energy is not None
    assert system.acceptor_reorganization_energy is not None
    assert system.free_energy_difference is not None
    assert system.donor_free_energy_difference is not None
    assert system.acceptor_free_energy_difference is not None


def test_missing_total_energy_raises_clear_error() -> None:
    """Reducer should raise a clear error when total energy is missing."""

    class _MissingEnergyProperties(Properties):
        system: PropertyClass

    missing = _MissingEnergyProperties(system=PropertyClass())
    valid = _energy_properties(0.0)

    with pytest.raises(
        ValueError, match=r"Missing system\.total_energy for donor_relaxed_initial_properties\."
    ):
        NelsonsFourPointCalculation._calculate_terms(
            donor_relaxed_initial_properties=missing,
            donor_relaxed_final_properties=valid,
            donor_cross_properties=valid,
            acceptor_relaxed_initial_properties=valid,
            acceptor_relaxed_final_properties=valid,
            acceptor_cross_properties=valid,
        )


def test_missing_optimizer_raises_clear_error() -> None:
    """Workflow should require optimizer as an attribute."""
    workflow = NelsonsFourPointMethod()
    donor = _single_hydrogen_molecule(charge=0, spin=1)
    acceptor = _single_hydrogen_molecule(charge=0, spin=1)
    with pytest.raises(
        ValueError, match=r"NelsonsFourPointMethod requires an `optimizer` attribute\."
    ):
        workflow._build_flow(donor, acceptor)
