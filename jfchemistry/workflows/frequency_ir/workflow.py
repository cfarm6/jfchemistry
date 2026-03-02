"""Frequency/IR analysis workflow with PySCF-GPU backend preference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from jobflow.core.job import OutputReference, Response
from pydantic import ConfigDict

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass

KB_EV_PER_K = 8.617333262145e-5
EH_TO_EV = 27.211386245988
CMINV_TO_EV = 1.2398419843320026e-4


class FrequencyIRSystemProperties(PropertyClass):
    """System properties from frequency/IR analysis."""

    zpe: SystemProperty | OutputReference
    thermal_correction_energy: SystemProperty | OutputReference
    thermal_correction_enthalpy: SystemProperty | OutputReference
    thermal_correction_gibbs: SystemProperty | OutputReference
    imaginary_frequency_count: SystemProperty | OutputReference


class FrequencyIRProperties(Properties):
    """Properties for frequency/IR analysis workflow."""

    system: FrequencyIRSystemProperties


class FrequencyIROutput(Output):
    """Output payload for frequency/IR analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    files: Optional[Any] = None
    properties: Optional[FrequencyIRProperties] = None


@dataclass
class FrequencyIRAnalysisCalculation(PymatGenMaker):
    """Reduce frequencies/intensities into thermochemistry + IR spectrum."""

    name: str = "Frequency/IR Analysis Calculation"
    temperature: float = 298.15
    pressure_pa: float = 101325.0
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[Output] = Output

    @staticmethod
    def select_backend(
        prefer_gpu: bool = True,
        available_backends: Optional[list[str]] = None,
    ) -> str:
        """Select backend with PySCF-GPU preference when available."""
        if available_backends is None:
            available_backends = ["pyscf-gpu", "orca", "ase"]
        normalized = [b.lower() for b in available_backends]
        if prefer_gpu and "pyscf-gpu" in normalized:
            return "pyscf-gpu"
        if "orca" in normalized:
            return "orca"
        if "ase" in normalized:
            return "ase"
        if len(normalized) < 1:
            raise ValueError("No available backends provided")
        return normalized[0]

    @staticmethod
    def count_imaginary_frequencies(
        frequencies_cm1: list[float],
        threshold_cm1: float = -10.0,
    ) -> int:
        """Count significantly imaginary modes."""
        arr = np.asarray(frequencies_cm1, dtype=float)
        return int(np.sum(arr < threshold_cm1))

    @staticmethod
    def build_ir_spectrum(  # noqa: PLR0913
        frequencies_cm1: list[float],
        intensities_km_mol: list[float],
        sigma_cm1: float = 20.0,
        wmin_cm1: float = 0.0,
        wmax_cm1: float = 4000.0,
        npts: int = 4000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct Gaussian-broadened IR spectrum."""
        if len(frequencies_cm1) != len(intensities_km_mol):
            raise ValueError("frequencies and intensities must have the same length")
        x = np.linspace(wmin_cm1, wmax_cm1, npts)
        y = np.zeros_like(x)
        for f, inten in zip(frequencies_cm1, intensities_km_mol, strict=False):
            if f > 0.0:
                y += float(inten) * np.exp(-0.5 * ((x - float(f)) / sigma_cm1) ** 2)
        return x, y

    @staticmethod
    def thermochemistry_corrections(
        frequencies_cm1: list[float],
        temperature: float,
    ) -> tuple[float, float, float, float]:
        """Compute simple harmonic thermal corrections in eV.

        Returns:
            (zpe, thermal_energy_corr, thermal_enthalpy_corr, thermal_gibbs_corr)
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0 K")

        freqs = np.asarray([f for f in frequencies_cm1 if f > 1.0], dtype=float)
        energies_ev = freqs * CMINV_TO_EV

        zpe = float(0.5 * energies_ev.sum())

        beta = 1.0 / (KB_EV_PER_K * temperature)
        # Harmonic vibrational thermal energy correction: sum( hv/(exp(beta hv)-1) )
        vib_u = float(np.sum(energies_ev / (np.exp(beta * energies_ev) - 1.0)))

        # Vibrational entropy term T*S_vib = sum( kT * [x/(exp(x)-1) - ln(1-exp(-x))] )
        x = beta * energies_ev
        entropy_term = x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))
        t_s_vib = float(np.sum(KB_EV_PER_K * temperature * entropy_term))

        thermal_energy_corr = zpe + vib_u
        thermal_enthalpy_corr = thermal_energy_corr + KB_EV_PER_K * temperature
        thermal_gibbs_corr = thermal_enthalpy_corr - t_s_vib
        return zpe, thermal_energy_corr, thermal_enthalpy_corr, thermal_gibbs_corr

    @jfchem_job()
    def make(
        self,
        frequencies_cm1: list[float],
        intensities_km_mol: list[float],
        available_backends: Optional[list[str]] = None,
        prefer_gpu: bool = True,
    ) -> Response[_output_model]:
        """Compute frequency diagnostics, IR spectrum, and thermal corrections."""
        backend = self.select_backend(prefer_gpu=prefer_gpu, available_backends=available_backends)
        imag_count = self.count_imaginary_frequencies(frequencies_cm1)
        x, y = self.build_ir_spectrum(
            frequencies_cm1=frequencies_cm1,
            intensities_km_mol=intensities_km_mol,
            sigma_cm1=self.sigma_cm1,
        )
        zpe, e_corr, h_corr, g_corr = self.thermochemistry_corrections(
            frequencies_cm1=frequencies_cm1,
            temperature=self.temperature,
        )
        return Response(
            output=FrequencyIROutput(
                properties=self._properties_model(
                    system=FrequencyIRSystemProperties(
                        zpe=SystemProperty(name="Zero Point Energy", value=zpe * ureg.eV),
                        thermal_correction_energy=SystemProperty(
                            name="Thermal Correction to Energy", value=e_corr * ureg.eV
                        ),
                        thermal_correction_enthalpy=SystemProperty(
                            name="Thermal Correction to Enthalpy", value=h_corr * ureg.eV
                        ),
                        thermal_correction_gibbs=SystemProperty(
                            name="Thermal Correction to Gibbs", value=g_corr * ureg.eV
                        ),
                        imaginary_frequency_count=SystemProperty(
                            name="Imaginary Frequency Count",
                            value=float(imag_count) * ureg.dimensionless,
                        ),
                    )
                ),
                files={
                    "backend": backend,
                    "spectrum_wavenumber_cm1": x.tolist(),
                    "spectrum_intensity_au": y.tolist(),
                },
            )
        )


@dataclass
class FrequencyIRAnalysisWorkflow(PymatGenMaker):
    """Workflow wrapper for frequency/IR reduction."""

    name: str = "Frequency/IR Analysis Workflow"
    temperature: float = 298.15
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @jfchem_job()
    def make(
        self,
        frequencies_cm1: list[float],
        intensities_km_mol: list[float],
        available_backends: Optional[list[str]] = None,
        prefer_gpu: bool = True,
    ) -> Response[_output_model]:
        """Create frequency/IR output from vibrational data."""
        calc = FrequencyIRAnalysisCalculation(
            temperature=self.temperature,
            sigma_cm1=self.sigma_cm1,
        )
        return calc.make.original(
            calc,
            frequencies_cm1=frequencies_cm1,
            intensities_km_mol=intensities_km_mol,
            available_backends=available_backends,
            prefer_gpu=prefer_gpu,
        )
