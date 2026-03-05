"""Tests for unified implicit-solvent adapters."""

import pytest

from jfchemistry.core.solvation import ImplicitSolventConfig, to_crest, to_orca, to_tblite


def test_tblite_adapter_alpb() -> None:
    """TBLite adapter should map ALPB model and solvent."""
    cfg = ImplicitSolventConfig(model="alpb", solvent="Water")
    solvation, solvent = to_tblite(cfg)
    assert solvation == "alpb"
    assert solvent == "Water"


def test_crest_adapter_gbsa() -> None:
    """CREST adapter should map GBSA model tuple."""
    cfg = ImplicitSolventConfig(model="gbsa", solvent="water")
    mapped = to_crest(cfg)
    assert mapped == ("gbsa", "water")


def test_orca_adapter_cpcm() -> None:
    """ORCA adapter should map CPCM fields."""
    cfg = ImplicitSolventConfig(model="cpcm", solvent="Water")
    mapped = to_orca(cfg)
    assert mapped["solvation_model"] == "CPCM"
    assert mapped["solvent"] == "Water"


def test_validate_requires_solvent() -> None:
    """Non-none models should require an explicit solvent."""
    with pytest.raises(ValueError, match="solvent must be provided"):
        to_orca(ImplicitSolventConfig(model="smd", solvent=None))
