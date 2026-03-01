"""Tests for MACE-Polar-1 ASE calculator integration."""

from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from jfchemistry.calculators.ase import MACEPolar1Calculator


class _DummyLoadedModel:
    def get_calculator(self, dtype: str = "float64"):
        assert dtype in {"float32", "float64"}
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        return SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]),
        )


def test_mace_polar1_set_calculator(monkeypatch):
    """Calculator should attach ASE calc from mace_models.load."""
    fake_module = ModuleType("mace_models")
    fake_module.load = lambda model: _DummyLoadedModel()  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "mace_models", fake_module)

    calc = MACEPolar1Calculator(model="MACE-Polar-1", dtype="float64")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    out = calc._set_calculator(atoms)
    assert out.calc is not None


def test_mace_polar1_get_properties():
    """Properties should contain energy + forces with expected names."""
    calc = MACEPolar1Calculator(model="MACE-Polar-1")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.23,
        forces=np.array([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]),
    )

    props = calc._get_properties(atoms)
    assert props.system.total_energy.name == "Total Energy"
    assert props.atomic.forces.name == "MACE-Polar-1 Forces"


def test_mace_polar1_missing_dependency(monkeypatch):
    """Missing mace_models should raise a clear ImportError."""
    monkeypatch.setitem(__import__("sys").modules, "mace_models", None)

    calc = MACEPolar1Calculator()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    with pytest.raises(ImportError, match="mace-models"):
        calc._set_calculator(atoms)
