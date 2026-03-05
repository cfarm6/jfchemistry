"""Tests for PropertiesToDisk utility."""

import json

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.utilities import PropertiesToDisk

FILE_NUM = 2


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


def _energy_properties(energy_ev: float) -> _EnergyProperties:
    return _EnergyProperties(
        system=_SystemProperties(
            total_energy=SystemProperty(name="Total Energy", value=energy_ev * ureg.eV)
        )
    )


def test_properties_to_disk_single(tmp_path) -> None:
    """Single properties object should write one JSON file."""
    maker = PropertiesToDisk(output_dir=str(tmp_path), filename="single_props.json")
    resp = maker.make.original(maker, _energy_properties(-5.0))

    files = resp.output.files["properties_json"]
    assert len(files) == 1
    p = tmp_path / "single_props.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert "system" in data


def test_properties_to_disk_list_suffixes(tmp_path) -> None:
    """List input should write indexed suffix files."""
    maker = PropertiesToDisk(output_dir=str(tmp_path), filename="props.json")
    props = [_energy_properties(-5.0), _energy_properties(-4.9)]
    resp = maker.make.original(maker, props)

    files = resp.output.files["properties_json"]
    assert len(files) == FILE_NUM
    p0 = tmp_path / "props_0.json"
    p1 = tmp_path / "props_1.json"
    assert p0.exists()
    assert p1.exists()
