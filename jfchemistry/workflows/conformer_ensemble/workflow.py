"""Conformer ensemble thermochemistry and Boltzmann averaging workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import OutputReference, Response
from pydantic import ConfigDict

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

KB_EV_PER_K = 8.617333262145e-5


class ConformerEnsembleSystemProperties(PropertyClass):
    """System properties from conformer ensemble reduction."""

    ensemble_free_energy: SystemProperty | OutputReference
    boltzmann_weighted_energy: SystemProperty | OutputReference


class ConformerEnsembleProperties(Properties):
    """Properties for conformer ensemble workflow."""

    system: ConformerEnsembleSystemProperties


class ConformerEnsembleOutput(Output):
    """Output for conformer ensemble workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    files: Optional[Any] = None
    properties: Optional[Any] = None


@dataclass
class ConformerEnsembleCalculation(PymatGenMaker):
    """Reduce conformer energies into Boltzmann ensemble properties."""

    name: str = "Conformer Ensemble Calculation"
    temperature: float = 298.15
    _properties_model: type[ConformerEnsembleProperties] = ConformerEnsembleProperties
    _output_model: type[Output] = Output

    @staticmethod
    def _extract_total_energy_ev(properties: Properties, label: str) -> float:
        _properties = Properties.model_validate(properties, extra="allow", strict=False)
        if (
            _properties.system is None
            or not hasattr(_properties.system, "total_energy")
            or (te := getattr(_properties.system, "total_energy", None)) is None
            or getattr(te, "value", None) is None
        ):
            raise ValueError(f"Missing system.total_energy for {label}.")
        energy = te.value
        if hasattr(energy, "to"):
            return float(energy.to(ureg.eV).magnitude)
        if isinstance(energy, (int, float)):
            return float(energy)
        raise ValueError(f"Could not parse system.total_energy for {label}.")

    @staticmethod
    def _boltzmann_weights(energies_ev: list[float], temperature: float) -> np.ndarray:
        if len(energies_ev) < 1:
            raise ValueError("At least one conformer energy is required")
        if temperature <= 0:
            raise ValueError("temperature must be > 0 K")
        energies = np.asarray(energies_ev, dtype=float)
        delta = energies - energies.min()
        beta = 1.0 / (KB_EV_PER_K * temperature)
        weights = np.exp(-beta * delta)
        z = weights.sum()
        if z <= 0:
            raise ValueError("Invalid Boltzmann partition function (non-positive)")
        return weights / z

    @classmethod
    def _ensemble_energetics(
        cls, energies_ev: list[float], temperature: float
    ) -> tuple[float, float, np.ndarray]:
        weights = cls._boltzmann_weights(energies_ev, temperature)
        energies = np.asarray(energies_ev, dtype=float)
        weighted_energy = float(np.dot(weights, energies))
        z = float(np.exp(-(energies - energies.min()) / (KB_EV_PER_K * temperature)).sum())
        free_energy = float(energies.min() - KB_EV_PER_K * temperature * np.log(z))
        return weighted_energy, free_energy, weights

    @jfchem_job()
    def make(self, conformer_properties: list[Properties]) -> Response[_output_model]:
        """Compute ensemble thermochemistry from conformer property list."""
        energies = [
            self._extract_total_energy_ev(props, f"conformer_properties[{i}]")
            for i, props in enumerate(conformer_properties)
        ]
        weighted_energy, free_energy, weights = self._ensemble_energetics(
            energies_ev=energies, temperature=self.temperature
        )
        return Response(
            output=ConformerEnsembleOutput(
                properties=self._properties_model(
                    system=ConformerEnsembleSystemProperties(
                        ensemble_free_energy=SystemProperty(
                            name="Ensemble Free Energy",
                            value=free_energy * ureg.eV,
                            description="Boltzmann ensemble free energy from conformer energies",
                        ),
                        boltzmann_weighted_energy=SystemProperty(
                            name="Boltzmann Weighted Energy",
                            value=weighted_energy * ureg.eV,
                            description="Boltzmann-weighted mean conformer energy",
                        ),
                    )
                ),
                files={"boltzmann_weights": weights.tolist()},
            )
        )


@dataclass
class ConformerEnsembleWorkflow(PymatGenMaker):
    """Conformer-ensemble workflow from a single Molecule input."""

    name: str = "Conformer Ensemble Workflow"
    conformer_generator: PymatGenMaker | None = None
    single_point: PymatGenMaker | None = None
    temperature: float = 298.15
    _properties_model: type[ConformerEnsembleProperties] = ConformerEnsembleProperties
    _output_model: type[ConformerEnsembleOutput] = ConformerEnsembleOutput

    def _build_flow(self, molecule: Molecule) -> tuple[Flow, ConformerEnsembleOutput]:
        if self.conformer_generator is None:
            raise ValueError(
                "ConformerEnsembleWorkflow requires a `conformer_generator` attribute."
            )

        conformer_job = self.conformer_generator.make(molecule)

        calc = ConformerEnsembleCalculation(temperature=self.temperature)

        if self.single_point is not None:
            sp_eval_job = self.single_point.make(conformer_job.output.structure)
            reducer_job = calc.make(sp_eval_job.output.properties)
            jobs = [conformer_job, sp_eval_job, reducer_job]
            files = {
                "conformers": conformer_job.output.files,
                "single_point": sp_eval_job.output.files,
                "boltzmann_weights": reducer_job.output.files["boltzmann_weights"],
            }
        else:
            reducer_job = calc.make(conformer_job.output.properties)
            jobs = [conformer_job, reducer_job]
            files = {
                "conformers": conformer_job.output.files,
                "boltzmann_weights": reducer_job.output.files["boltzmann_weights"],
            }

        flow = Flow(jobs, name=self.name)
        output = ConformerEnsembleOutput(
            structure=conformer_job.output.structure,
            properties=reducer_job.output.properties,
            files=files,
        )
        return flow, output

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Create conformer ensemble output from a Molecule input."""
        flow, output = self._build_flow(molecule)
        return Response(output=output, detour=flow)
