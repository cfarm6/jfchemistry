"""Tests for lazy import behavior of PySCF-GPU package."""

import importlib


def test_pyscfgpu_package_import_is_lazy() -> None:
    """Importing package should not eagerly import CuPy-bound calculator module."""
    mod = importlib.import_module("jfchemistry.calculators.pyscfgpu")
    assert hasattr(mod, "__getattr__")
    assert "PySCFCalculator" in getattr(mod, "__all__", [])
