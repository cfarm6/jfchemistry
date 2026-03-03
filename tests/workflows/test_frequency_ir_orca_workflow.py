"""Tests for ORCA-specific frequency/IR workflow."""

from dataclasses import dataclass

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.workflows.frequency_ir.orca import FrequencyIRORCACalculation


@dataclass
class _DummyORCACalculator(ORCACalculator):
    """Minimal ORCA calculator stub for workflow wiring tests."""


def _single_hydrogen_molecule() -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=0,
        spin_multiplicity=1,
        charge_spin_check=False,
    )


def test_orca_frequency_make_with_mocked_frequency_data(monkeypatch) -> None:
    """ORCA workflow should build output from parsed ORCA frequency payload."""
    calc = FrequencyIRORCACalculation(calculator=_DummyORCACalculator())
    mol = _single_hydrogen_molecule()

    def _fake_run(_molecule):
        return {
            "frequencies_cm1": [120.0, 540.0, 1620.0],
            "intensities_km_mol": [11.0, 51.0, 121.0],
            "energy_ev": -1.0,
        }

    monkeypatch.setattr(calc, "_run_orca_frequency_data", _fake_run)
    response = calc.make.original(calc, mol)

    out = response.output
    assert out is not None
    assert out.properties is not None
    expected_modes = 3
    assert out.files is not None
    assert out.files["backend"] == "orca"
    assert len(out.files["frequencies_cm1"]) == expected_modes
    assert out.properties.system.zpe.value is not None


def test_orca_frequency_requires_calculator() -> None:
    """ORCA frequency workflow should require an ORCA calculator."""
    calc = FrequencyIRORCACalculation(calculator=None)
    mol = _single_hydrogen_molecule()

    with pytest.raises(ValueError, match="requires `calculator`"):
        calc._run_orca_frequency_data(mol)
