"""Provenance models and helpers for reproducible outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel


class ProvenanceRecord(BaseModel):
    """Run provenance metadata for auditability and reproducibility."""

    engine: Optional[str] = None
    method: Optional[str] = None
    basis: Optional[str] = None
    model: Optional[str] = None
    model_hash: Optional[str] = None
    software_version: Optional[str] = None
    random_seed: Optional[int] = None
    timestamp_utc: str
    extras: dict[str, Any] = {}


def make_provenance(  # noqa: PLR0913
    engine: Optional[str] = None,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    model: Optional[str] = None,
    model_hash: Optional[str] = None,
    software_version: Optional[str] = None,
    random_seed: Optional[int] = None,
    extras: Optional[dict[str, Any]] = None,
) -> ProvenanceRecord:
    """Create a provenance record with UTC timestamp."""
    return ProvenanceRecord(
        engine=engine,
        method=method,
        basis=basis,
        model=model,
        model_hash=model_hash,
        software_version=software_version,
        random_seed=random_seed,
        timestamp_utc=datetime.now(UTC).isoformat(),
        extras=extras or {},
    )
