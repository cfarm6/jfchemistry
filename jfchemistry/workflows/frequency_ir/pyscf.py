"""PySCF-specific frequency and IR analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jobflow.core.job import Response
from pyscf.hessian import thermo as pyscf_thermo

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.workflows.frequency_ir.workflow import (
    EH_TO_EV,
    FrequencyIRAnalysisCalculation,
    FrequencyIROutput,
    FrequencyIRProperties,
    FrequencyIRSystemProperties,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

    from jfchemistry.calculators.pyscfgpu import PySCFCalculator


@dataclass
class FrequencyIRPySCFCalculation(PymatGenMaker):
    """Compute frequencies/IR with PySCF and reduce thermochemistry terms."""

    name: str = "PySCF Frequency/IR Calculation"
    calculator: PySCFCalculator | None = None
    temperature: float = 298.15
    pressure_pa: float = 101325.0
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    def _run_pyscf_frequency_data(self, molecule: Molecule) -> dict[str, Any]:
        """Run PySCF Hessian analysis and return vibrational data payload."""
        if self.calculator is None:
            raise ValueError("FrequencyIRPySCFCalculation requires `calculator`.")

        mol = self.calculator._get_mol(molecule)
        mf = self.calculator._setup_mf(mol)
        mf.kernel()

        hess = mf.Hessian().kernel()
        analysis = pyscf_thermo.harmonic_analysis(mol, hess)
        frequencies_cm1 = [float(x) for x in analysis["freq_wavenumber"]]

        # PySCF harmonic analysis does not provide IR intensities directly.
        intensities_km_mol = [0.0 for _ in frequencies_cm1]

        th = pyscf_thermo.thermo(
            mf,
            analysis["freq_au"],
            temperature=self.temperature,
            pressure=self.pressure_pa,
        )
        energy_ev = float(th["E_elec"][0]) * EH_TO_EV
        zpe_ev = float(th["ZPE"][0]) * EH_TO_EV
        e_corr_ev = (float(th["E_tot"][0]) - float(th["E_elec"][0])) * EH_TO_EV
        h_corr_ev = (float(th["H_tot"][0]) - float(th["E_elec"][0])) * EH_TO_EV
        g_corr_ev = (float(th["G_tot"][0]) - float(th["E_elec"][0])) * EH_TO_EV

        return {
            "frequencies_cm1": frequencies_cm1,
            "intensities_km_mol": intensities_km_mol,
            "energy_ev": energy_ev,
            "zpe_ev": zpe_ev,
            "e_corr_ev": e_corr_ev,
            "h_corr_ev": h_corr_ev,
            "g_corr_ev": g_corr_ev,
        }

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run PySCF-specific frequency/IR calculation for one molecule."""
        data = self._run_pyscf_frequency_data(molecule)
        frequencies_cm1 = data["frequencies_cm1"]
        intensities_km_mol = data["intensities_km_mol"]

        imag_count = FrequencyIRAnalysisCalculation.count_imaginary_frequencies(frequencies_cm1)
        x, y = FrequencyIRAnalysisCalculation.build_ir_spectrum(
            frequencies_cm1=frequencies_cm1,
            intensities_km_mol=intensities_km_mol,
            sigma_cm1=self.sigma_cm1,
        )

        return Response(
            output=FrequencyIROutput(
                properties=self._properties_model(
                    system=FrequencyIRSystemProperties(
                        zpe=SystemProperty(
                            name="Zero Point Energy",
                            value=data["zpe_ev"] * ureg.eV,
                        ),
                        thermal_correction_energy=SystemProperty(
                            name="Thermal Correction to Energy",
                            value=data["e_corr_ev"] * ureg.eV,
                        ),
                        thermal_correction_enthalpy=SystemProperty(
                            name="Thermal Correction to Enthalpy",
                            value=data["h_corr_ev"] * ureg.eV,
                        ),
                        thermal_correction_gibbs=SystemProperty(
                            name="Thermal Correction to Gibbs",
                            value=data["g_corr_ev"] * ureg.eV,
                        ),
                        imaginary_frequency_count=SystemProperty(
                            name="Imaginary Frequency Count",
                            value=float(imag_count) * ureg.dimensionless,
                        ),
                    )
                ),
                files={
                    "backend": "pyscf",
                    "frequencies_cm1": frequencies_cm1,
                    "intensities_km_mol": intensities_km_mol,
                    "spectrum_wavenumber_cm1": x.tolist(),
                    "spectrum_intensity_au": y.tolist(),
                },
            )
        )


@dataclass
class FrequencyIRPySCFWorkflow(PymatGenMaker):
    """PySCF frequency/IR workflow entry-point."""

    name: str = "PySCF Frequency/IR Workflow"
    calculator: PySCFCalculator | None = None
    temperature: float = 298.15
    pressure_pa: float = 101325.0
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run PySCF-specific frequency/IR calculation."""
        calc = FrequencyIRPySCFCalculation(
            calculator=self.calculator,
            temperature=self.temperature,
            pressure_pa=self.pressure_pa,
            sigma_cm1=self.sigma_cm1,
        )
        return calc.make.original(calc, molecule)
