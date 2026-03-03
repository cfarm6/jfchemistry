"""Tests for redox property workflow."""

from dataclasses import dataclass, field

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.redox.workflow import RedoxPropertyCalculation, RedoxPropertyWorkflow


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


@dataclass
class _ConstantStateEnergyMaker(PymatGenMaker[Molecule, Molecule]):
    """Maker that returns state-dependent fixed energies for workflow tests."""

    energy_by_charge: dict[int, float] = field(default_factory=dict)
    charge: int | None = None
    spin_multiplicity: int | None = None
    steps: int = 100
    _properties_model: type[_EnergyProperties] = _EnergyProperties

    def _operation(self, input: Molecule, **kwargs):
        if self.charge is None:
            raise ValueError("charge must be set by workflow before running")
        energy = self.energy_by_charge[self.charge]
        properties = _EnergyProperties(
            system=_SystemProperties(
                total_energy=SystemProperty(name="Total Energy", value=energy * ureg.eV)
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


def test_vertical_ip_ea_equations() -> None:
    """Vertical IP/EA should follow neutral-geometry expressions."""
    v_ip, v_ea, _, _ = RedoxPropertyCalculation._compute_redox_terms(
        neutral_relaxed=_energy_properties(-5.0),
        cation_relaxed=_energy_properties(-4.0),
        anion_relaxed=_energy_properties(-5.5),
        cation_on_neutral_geom=_energy_properties(-3.8),
        anion_on_neutral_geom=_energy_properties(-5.3),
    )
    assert v_ip == pytest.approx(1.2)
    assert v_ea == pytest.approx(0.3)


def test_adiabatic_ip_ea_equations() -> None:
    """Adiabatic IP/EA should use relaxed charged-state energies."""
    _, _, a_ip, a_ea = RedoxPropertyCalculation._compute_redox_terms(
        neutral_relaxed=_energy_properties(-5.0),
        cation_relaxed=_energy_properties(-4.2),
        anion_relaxed=_energy_properties(-5.7),
        cation_on_neutral_geom=_energy_properties(-4.0),
        anion_on_neutral_geom=_energy_properties(-5.4),
    )
    assert a_ip == pytest.approx(0.8)
    assert a_ea == pytest.approx(0.7)


def test_charge_spin_validation() -> None:
    """Charge relation checks should raise clear errors."""
    with pytest.raises(ValueError, match="cation charge"):
        RedoxPropertyCalculation.validate_charge_spin_states(
            neutral_charge=0,
            cation_charge=0,
            anion_charge=-1,
            neutral_spin=1,
            cation_spin=2,
            anion_spin=2,
        )

EXPECTED_JOB_COUNT = 6


def test_workflow_accepts_single_molecule_and_builds_jobs() -> None:
    """Workflow make interface should take one Molecule and build 6 jobs."""
    optimizer = _ConstantStateEnergyMaker(energy_by_charge={0: -5.0, 1: -4.2, -1: -5.7})
    single_point = _ConstantStateEnergyMaker(energy_by_charge={0: -5.0, 1: -4.0, -1: -5.4})
    wf = RedoxPropertyWorkflow(optimizer=optimizer, single_point=single_point)
    molecule = _single_hydrogen_molecule(charge=0, spin=1)

    flow, output = wf._build_flow(molecule)

    assert len(flow.jobs) == EXPECTED_JOB_COUNT
    assert output.properties is not None
    assert output.properties.system.vertical_ip is not None
    assert output.properties.system.vertical_ea is not None
    assert output.properties.system.adiabatic_ip is not None
    assert output.properties.system.adiabatic_ea is not None


def test_missing_optimizer_or_single_point_raises_clear_error() -> None:
    """Workflow should require both optimizer and single-point makers."""
    molecule = _single_hydrogen_molecule(charge=0, spin=1)
    with pytest.raises(ValueError, match="requires an `optimizer` attribute"):
        RedoxPropertyWorkflow(
            single_point=_ConstantStateEnergyMaker(energy_by_charge={0: -1.0})
        )._build_flow(molecule)
    with pytest.raises(ValueError, match="requires a `single_point` attribute"):
        RedoxPropertyWorkflow(
            optimizer=_ConstantStateEnergyMaker(energy_by_charge={0: -1.0})
        )._build_flow(molecule)
