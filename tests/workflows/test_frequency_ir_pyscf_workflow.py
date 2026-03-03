"""Tests for PySCF-specific frequency/IR workflow."""

from dataclasses import dataclass

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.pyscfgpu import PySCFCalculator
from jfchemistry.workflows.frequency_ir.pyscf import FrequencyIRPySCFCalculation


@dataclass
class _DummyPySCFCalculator(PySCFCalculator):
    """Minimal PySCF calculator stub for workflow wiring tests."""



def _single_hydrogen_molecule() -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=0,
        spin_multiplicity=1,
        charge_spin_check=False,
    )


def test_pyscf_frequency_make_with_mocked_frequency_data(monkeypatch) -> None:
    """PySCF workflow should build output from PySCF frequency payload."""
    calc = FrequencyIRPySCFCalculation(calculator=_DummyPySCFCalculator(mode="cpu"))
    mol = _single_hydrogen_molecule()

    def _fake_run(_molecule):
        return {
            "frequencies_cm1": [150.0, 550.0, 1650.0],
            "intensities_km_mol": [0.0, 0.0, 0.0],
            "energy_ev": -1.0,
            "zpe_ev": 0.12,
            "e_corr_ev": 0.15,
            "h_corr_ev": 0.18,
            "g_corr_ev": 0.10,
        }

    monkeypatch.setattr(calc, "_run_pyscf_frequency_data", _fake_run)
    response = calc.make.original(calc, mol)

    out = response.output
    assert out is not None
    assert out.properties is not None
    expected_modes = 3
    assert out.files is not None
    assert out.files["backend"] == "pyscf"
    assert len(out.files["frequencies_cm1"]) == expected_modes
    assert out.properties.system.zpe.value is not None


def test_pyscf_frequency_requires_calculator() -> None:
    """PySCF frequency workflow should require a PySCF calculator."""
    calc = FrequencyIRPySCFCalculation(calculator=None)
    mol = _single_hydrogen_molecule()

    with pytest.raises(ValueError, match="requires `calculator`"):
        calc._run_pyscf_frequency_data(mol)
