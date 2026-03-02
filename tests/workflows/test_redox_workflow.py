"""Tests for redox property workflow."""

import pytest

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.redox.workflow import RedoxPropertyCalculation, RedoxPropertyWorkflow


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


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


def test_workflow_output_contains_all_redox_terms() -> None:
    """Workflow output should include all vertical/adiabatic IP/EA fields."""
    wf = RedoxPropertyWorkflow()
    response = wf.make.original(
        wf,
        neutral_relaxed=_energy_properties(-5.0),
        cation_relaxed=_energy_properties(-4.2),
        anion_relaxed=_energy_properties(-5.7),
        cation_on_neutral_geom=_energy_properties(-4.0),
        anion_on_neutral_geom=_energy_properties(-5.4),
    )
    out = response.output
    assert out is not None
    assert out.properties is not None
    system = out.properties.system
    assert system.vertical_ip is not None
    assert system.vertical_ea is not None
    assert system.adiabatic_ip is not None
    assert system.adiabatic_ea is not None
