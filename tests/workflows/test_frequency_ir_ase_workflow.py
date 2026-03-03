"""Tests for ASE-specific frequency/IR workflow."""

from dataclasses import dataclass

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.workflows.frequency_ir.ase import FrequencyIRASECalculation


@dataclass
class _DummyASECalculator(ASECalculator):
    """Minimal ASE calculator stub for workflow wiring tests."""

    def _set_calculator(self, atoms, charge: float = 0, spin_multiplicity: int = 1):
        return atoms

    def _get_properties(self, atoms):
        return None


def _single_hydrogen_molecule() -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=0,
        spin_multiplicity=1,
        charge_spin_check=False,
    )


def test_ase_frequency_make_with_mocked_frequency_data(monkeypatch) -> None:
    """ASE workflow should build output from ASE frequency payload."""
    calc = FrequencyIRASECalculation(calculator=_DummyASECalculator())
    mol = _single_hydrogen_molecule()

    def _fake_run(_molecule):
        atoms = _molecule.to_ase_atoms()
        return {
            "frequencies_cm1": [100.0, 500.0, 1500.0],
            "intensities_km_mol": [10.0, 50.0, 120.0],
            "energy_ev": -1.0,
            "atoms": atoms,
        }

    monkeypatch.setattr(calc, "_run_ase_frequency_data", _fake_run)
    response = calc.make.original(calc, mol)

    out = response.output
    assert out is not None
    assert out.properties is not None
    assert out.files is not None
    expected_modes = 3
    assert out.files["backend"] == "ase"
    assert len(out.files["frequencies_cm1"]) == expected_modes
    assert out.properties.system.zpe.value is not None


def test_ase_frequency_requires_calculator() -> None:
    """ASE frequency workflow should require an ASE calculator."""
    calc = FrequencyIRASECalculation(calculator=None)
    mol = _single_hydrogen_molecule()

    with pytest.raises(ValueError, match="requires `calculator`"):
        calc._run_ase_frequency_data(mol)
