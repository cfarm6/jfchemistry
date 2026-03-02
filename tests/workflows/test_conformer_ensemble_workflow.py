"""Tests for conformer ensemble thermochemistry workflow."""

import pytest

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.conformer_ensemble.workflow import (
    ConformerEnsembleCalculation,
    ConformerEnsembleWorkflow,
)


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


def test_boltzmann_weights_normalize() -> None:
    """Weights should sum to one and prefer low-energy conformers."""
    w = ConformerEnsembleCalculation._boltzmann_weights([0.0, 0.05, 0.10], 298.15)
    assert float(w.sum()) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_ensemble_energetics_bounds() -> None:
    """Weighted mean/free energy should not exceed the lowest conformer energy."""
    weighted, free, _ = ConformerEnsembleCalculation._ensemble_energetics([0.0, 0.04, 0.08], 298.15)
    assert weighted >= 0.0
    assert free <= weighted
    assert free <= 0.0 + 1e-12


def test_missing_total_energy_raises_clear_error() -> None:
    """Reducer should raise clear error when a conformer misses total_energy."""

    class _MissingEnergyProperties(Properties):
        system: PropertyClass

    missing = _MissingEnergyProperties(system=PropertyClass())
    valid = _energy_properties(0.0)
    calc = ConformerEnsembleCalculation()

    with pytest.raises(
        ValueError,
        match=r"Missing system\.total_energy for conformer_properties\[0\]\.",
    ):
        calc.make.original(calc, [missing, valid])


def test_workflow_output_contains_properties_and_weights() -> None:
    """Workflow should emit ensemble properties and weight file payload."""
    wf = ConformerEnsembleWorkflow(temperature=298.15)
    result = wf.make.original(wf, [_energy_properties(0.0), _energy_properties(0.02)])
    out = result.output
    assert out is not None
    assert out.properties is not None
    assert out.files is not None
    expected_conformer_count = 2
    assert "boltzmann_weights" in out.files
    assert len(out.files["boltzmann_weights"]) == expected_conformer_count
