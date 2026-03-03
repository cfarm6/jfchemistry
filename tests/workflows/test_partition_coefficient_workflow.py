"""Tests for fresh molecule-first partition coefficient workflow."""

from dataclasses import dataclass, field

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.partition_coefficient.workflow import (
    PartitionCoefficientReductionCalculation,
    PartitionCoefficientWorkflow,
)


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


def _single_h_molecule() -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=0,
        spin_multiplicity=1,
        charge_spin_check=False,
    )


@dataclass
class _GeneratorStub(PymatGenMaker[Molecule, Molecule]):
    """Conformer/tautomer generator stub: returns N copied structures."""

    n: int = 3

    def _operation(self, input: Molecule, **kwargs):
        structures = [input.copy() for _ in range(self.n)]
        return structures, None


@dataclass
class _PassThroughOptimizer(PymatGenMaker[Molecule, Molecule]):
    """Optimizer stub returning input structures with empty properties."""

    def _operation(self, input, **kwargs):
        if not isinstance(input, list):
            input = [input]
        return input, [Properties() for _ in input]


@dataclass
class _SinglePointStub(PymatGenMaker[Molecule, Molecule]):
    """Single-point stub assigning preset energies by index."""

    energies: list[float] = field(default_factory=lambda: [-5.0, -4.9, -4.8])

    def _operation(self, input, **kwargs):
        if not isinstance(input, list):
            input = [input]
        props = [_energy_properties(self.energies[i]) for i in range(len(input))]
        return input, props


def test_reduction_computes_logp() -> None:
    """Reducer should compute finite logP and delta-G from phase energies."""
    reducer = PartitionCoefficientReductionCalculation(temperature=298.15)
    alpha = [_energy_properties(-5.0), _energy_properties(-4.9)]
    beta = [_energy_properties(-5.2), _energy_properties(-5.1)]
    out = reducer.make.original(reducer, alpha, beta).output
    assert out.properties.system.log_partition_coefficient.value is not None
    assert out.properties.system.delta_g_transfer.value is not None


def test_workflow_make_accepts_single_molecule() -> None:
    """Workflow interface should be molecule-first and produce flow/output."""
    wf = PartitionCoefficientWorkflow(
        tautomer_generator=None,
        conformer_generator=_GeneratorStub(n=3),
        geometry_optimizer=_PassThroughOptimizer(),
        single_point=_SinglePointStub(energies=[-5.0, -4.95, -4.9]),
    )
    mol = _single_h_molecule()
    response = wf.make.original(wf, mol)
    assert response.detour is not None
    assert response.output is not None


def test_reduction_requires_total_energy() -> None:
    """Reducer should fail clearly if total energy is missing."""
    reducer = PartitionCoefficientReductionCalculation()
    with pytest.raises(ValueError, match=r"Missing system\.total_energy"):
        reducer.make.original(reducer, [Properties()], [_energy_properties(-1.0)])
