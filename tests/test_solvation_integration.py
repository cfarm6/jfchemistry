"""Integration tests for unified implicit-solvent wiring into calculators."""

import pytest

from jfchemistry.calculators.ase.tblite_calculator import TBLiteCalculator
from jfchemistry.calculators.crest.crest_calculator import CRESTCalculator
from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.core.solvation import ImplicitSolventConfig, to_pyscfgpu


def test_orca_implicit_solvent_override() -> None:
    """ORCA calculator should map unified CPCM solvent fields."""
    calc = ORCACalculator(implicit_solvent=ImplicitSolventConfig(model="cpcm", solvent="Water"))
    assert calc.solvation_model == "CPCM"
    assert calc.solvent == "Water"


def test_tblite_implicit_solvent_override() -> None:
    """TBLite calculator should map unified ALPB solvent fields."""
    calc = TBLiteCalculator(implicit_solvent=ImplicitSolventConfig(model="alpb", solvent="Water"))
    assert calc.solvation == "alpb"
    assert calc.solvent == "Water"


def test_crest_implicit_solvent_override() -> None:
    """CREST calculator should map unified GBSA tuple."""
    calc = CRESTCalculator(implicit_solvent=ImplicitSolventConfig(model="gbsa", solvent="water"))
    assert calc.solvation == ("gbsa", "water")


def test_pyscf_unsupported_solvent_model_raises() -> None:
    """PySCF-GPU adapter should reject unsupported implicit-solvent models."""
    with pytest.raises(ValueError, match="supports only model='none'"):
        to_pyscfgpu(ImplicitSolventConfig(model="cpcm", solvent="Water"))
