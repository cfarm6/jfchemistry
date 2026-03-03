"""Tests for conformer ensemble thermochemistry workflow."""

from dataclasses import dataclass, field

import pytest
from pymatgen.core.structure import Molecule

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.conformer_ensemble.workflow import (
    ConformerEnsembleCalculation,
    ConformerEnsembleWorkflow,
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


def _single_hydrogen_molecule(charge: int = 0, spin: int = 1) -> Molecule:
    return Molecule(
        ["H"],
        [[0.0, 0.0, 0.0]],
        charge=charge,
        spin_multiplicity=spin,
        charge_spin_check=False,
    )


@dataclass
class _DummyConformerGenerator(PymatGenMaker[Molecule, Molecule]):
    """Generate N conformers with simple translated coordinates and no properties."""

    n_conformers: int = 3

    def _operation(self, input: Molecule, **kwargs):
        conformers = []
        for i in range(self.n_conformers):
            c = input.copy()
            coords = c.cart_coords
            coords[:, 0] = coords[:, 0] + 0.1 * i
            c._coords = coords
            conformers.append(c)
        return conformers, None


@dataclass
class _SinglePointByIndex(PymatGenMaker[Molecule, Molecule]):
    """Single-point maker that assigns energies by call index."""

    energies_ev: list[float] = field(default_factory=lambda: [-10.0, -9.98, -9.95])
    _counter: int = 0

    def _operation(self, input: Molecule, **kwargs):
        idx = self._counter
        self._counter += 1
        e = self.energies_ev[idx]
        props = _energy_properties(e)
        return input, props


def test_boltzmann_weights_normalize() -> None:
    """Weights should sum to one and prefer low-energy conformers."""
    w = ConformerEnsembleCalculation._boltzmann_weights([0.0, 0.05, 0.10], 298.15)
    assert float(w.sum()) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_ensemble_energetics_bounds() -> None:
    """Weighted mean/free energy should not exceed the lowest conformer energy."""
    weighted, free, _ = ConformerEnsembleCalculation._ensemble_energetics(
        [0.0, 0.04, 0.08], 298.15
    )
    assert weighted >= 0.0
    assert free <= weighted
    assert free <= 0.0 + 1e-12


def test_missing_total_energy_raises_clear_error() -> None:
    """Reducer should raise clear error when a conformer misses total_energy."""

    class _MissingEnergyProperties(Properties):
        system: PropertyClass

    missing = _MissingEnergyProperties(system=PropertyClass())
    valid = _energy_properties(0.0)
    calc = ConformerEnsembleCalculation()

    with pytest.raises(
        ValueError,
        match=r"Missing system\.total_energy for conformer_properties\[0\]\.",
    ):
        calc.make.original(calc, [missing, valid])


def test_workflow_make_accepts_molecule_and_builds_flow() -> None:
    """Workflow should accept a Molecule and use generator + single point makers."""
    wf = ConformerEnsembleWorkflow(
        conformer_generator=_DummyConformerGenerator(n_conformers=3),
        single_point=_SinglePointByIndex(energies_ev=[-10.0, -9.98, -9.95]),
        temperature=298.15,
    )
    mol = _single_hydrogen_molecule()

    flow, output = wf._build_flow(mol)

    expected_jobs = 3  # generator + single-point evaluator + reducer
    assert len(flow.jobs) == expected_jobs
    assert output.properties is not None
    assert output.files is not None


def test_workflow_requires_generator() -> None:
    """Workflow should require a conformer generator maker."""
    wf = ConformerEnsembleWorkflow()
    with pytest.raises(ValueError, match="requires a `conformer_generator` attribute"):
        wf._build_flow(_single_hydrogen_molecule())
