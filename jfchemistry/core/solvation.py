"""Unified implicit-solvent schema and backend adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SolvationModel = Literal["none", "alpb", "gbsa", "cpcm", "smd"]


@dataclass
class ImplicitSolventConfig:
    """Unified implicit-solvent input for calculator backends."""

    model: SolvationModel = "none"
    solvent: Optional[str] = None

    def validate(self) -> None:
        """Validate model/solvent consistency."""
        if self.model == "none":
            return
        if not self.solvent:
            raise ValueError("solvent must be provided when model is not 'none'")


def to_tblite(config: ImplicitSolventConfig) -> tuple[Optional[str], Optional[str]]:
    """Map unified config to TBLite fields (solvation, solvent)."""
    config.validate()
    if config.model == "none":
        return None, None
    if config.model != "alpb":
        raise ValueError("TBLite supports only ALPB in current adapter")
    return "alpb", config.solvent


def to_crest(config: ImplicitSolventConfig) -> Optional[tuple[str, str]]:
    """Map unified config to CREST tuple format."""
    config.validate()
    if config.model == "none":
        return None
    if config.model not in {"alpb", "gbsa"}:
        raise ValueError("CREST adapter supports ALPB/GBSA only")
    return config.model, str(config.solvent)


def to_pyscfgpu(config: ImplicitSolventConfig) -> None:
    """Validate unified config for current PySCF-GPU support.

    Current adapter supports only model='none'.
    """
    config.validate()
    if config.model != "none":
        raise ValueError("PySCF-GPU implicit solvent adapter currently supports only model='none'")


def to_orca(config: ImplicitSolventConfig) -> dict[str, Optional[str]]:
    """Map unified config to ORCA-style solvation fields."""
    config.validate()
    if config.model == "none":
        return {"solvation": None, "solvation_model": None, "solvent": None}
    # ORCA uses CPCM/SMD model keyword plus solvent.
    if config.model == "cpcm":
        return {"solvation": "CPCM", "solvation_model": "CPCM", "solvent": config.solvent}
    if config.model == "smd":
        return {"solvation": "SMD", "solvation_model": "SMD", "solvent": config.solvent}
    raise ValueError("ORCA adapter supports CPCM/SMD only")
