"""Tests for PySCF calculator CPU/GPU mode selection."""

import pytest

from jfchemistry.calculators.pyscfgpu.pyscfgpu_calculator import PySCFCalculator


def test_mode_cpu_always_selects_cpu() -> None:
    """CPU mode should always select CPU backend."""
    calc = PySCFCalculator(mode="cpu")
    assert calc._selected_backend() == "cpu"


def test_mode_auto_falls_back_to_cpu(monkeypatch) -> None:
    """Auto mode should fall back to CPU when GPU backend is unavailable."""
    calc = PySCFCalculator(mode="auto")
    monkeypatch.setattr(calc, "_gpu_available", lambda: False)
    assert calc._selected_backend() == "cpu"


def test_mode_gpu_raises_when_unavailable(monkeypatch) -> None:
    """GPU mode should raise clear error if GPU backend is unavailable."""
    calc = PySCFCalculator(mode="gpu")
    monkeypatch.setattr(calc, "_gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU mode selected"):
        calc._selected_backend()
