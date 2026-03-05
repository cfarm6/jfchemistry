"""Tests for provenance schema and Output integration."""

from jfchemistry.core.outputs import Output
from jfchemistry.core.provenance import make_provenance


def test_make_provenance_stamps_key_fields() -> None:
    """Provenance helper should stamp timestamp + supplied fields."""
    p = make_provenance(engine="orca", method="b3lyp", basis="def2-svp", model="DFT")
    assert p.engine == "orca"
    assert p.method == "b3lyp"
    assert p.basis == "def2-svp"
    assert p.timestamp_utc


def test_output_roundtrip_preserves_provenance() -> None:
    """Output serialization/deserialization should preserve provenance."""
    seed = 7
    p = make_provenance(engine="tblite", method="gfn2-xtb", random_seed=seed)
    out = Output(provenance=p)
    back = Output.from_dict(out.to_dict())
    assert back.provenance is not None
    assert back.provenance.engine == "tblite"
    assert back.provenance.random_seed == seed
