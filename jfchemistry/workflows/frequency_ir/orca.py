"""ORCA-specific frequency and IR analysis workflow."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jobflow.core.job import Response

from jfchemistry import SystemProperty, ureg
from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.workflows.frequency_ir.workflow import (
    FrequencyIRAnalysisCalculation,
    FrequencyIROutput,
    FrequencyIRProperties,
    FrequencyIRSystemProperties,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule


@dataclass
class FrequencyIRORCACalculation(PymatGenMaker):
    """Compute frequencies/IR with ORCA and reduce thermochemistry terms."""

    name: str = "ORCA Frequency/IR Calculation"
    calculator: ORCACalculator | None = None
    temperature: float = 298.15
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @staticmethod
    def _parse_orca_output_text(text: str) -> tuple[list[float], list[float], float | None]:
        """Parse ORCA output text for frequencies, IR intensities, and final energy.

        This parser is intentionally permissive to support minor ORCA formatting differences.
        """
        frequencies: list[float] = []
        intensities: list[float] = []

        # Frequencies: lines like "  1:   123.45 cm**-1"
        freq_re = re.compile(r"^\s*\d+\s*:\s*(-?\d+\.\d+)\s*cm\*\*-1", re.MULTILINE)
        frequencies = [float(x) for x in freq_re.findall(text)]

        # IR spectrum table lines typically include mode, freq, intensity
        # Example captures numbers with at least 3 columns; use last as intensity.
        ir_block = re.search(r"IR SPECTRUM(.*?)(?:\n\s*\n|\Z)", text, flags=re.DOTALL)
        if ir_block:
            for line in ir_block.group(1).splitlines():
                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                min_columns = 3
                if len(nums) >= min_columns:
                    try:
                        intensities.append(float(nums[-1]))
                    except ValueError:
                        continue

        if len(intensities) != len(frequencies):
            # fall back: zeros if not parsed cleanly
            intensities = [0.0 for _ in frequencies]

        # Final single-point energy in Eh
        e_match = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
        energy_ev = float(e_match[-1]) * 27.211386245988 if e_match else None

        return frequencies, intensities, energy_ev

    def _run_orca_frequency_data(self, molecule: Molecule) -> dict[str, Any]:
        """Run ORCA frequency analysis and return parsed vibrational data."""
        if self.calculator is None:
            raise ValueError("FrequencyIRORCACalculation requires `calculator`.")

        with tempfile.TemporaryDirectory(prefix="jfchem_orca_freq_") as tmpdir:
            mol_path = Path(tmpdir) / "input.xyz"
            molecule.to(str(mol_path), fmt="xyz")

            calc = ORCACalculator(**self.calculator.as_dict())
            calc.working_dir = tmpdir
            # ensure frequency job keyword
            if "FREQ" not in [k.upper() for k in calc.additional_keywords]:
                calc.additional_keywords.append("FREQ")

            sk_list = calc._set_keywords()
            opi_calc = calc._build_calculator("orca_freq")

            from opi.input.structures.structure import Structure

            opi_calc.structure = Structure.from_xyz(str(mol_path))
            calc._set_structure_charge_and_spin(
                opi_calc,
                molecule.charge,
                molecule.spin_multiplicity,
            )
            calc._configure_calculator_input(opi_calc, sk_list)
            opi_calc.write_input()
            opi_calc.run()

            out_path = Path(tmpdir) / "orca_freq.out"
            if not out_path.exists():
                # fallback to OPI output object text if file naming differs
                output = opi_calc.get_output()
                text = str(output)
            else:
                text = out_path.read_text(errors="ignore")

        frequencies_cm1, intensities_km_mol, energy_ev = self._parse_orca_output_text(text)
        return {
            "frequencies_cm1": frequencies_cm1,
            "intensities_km_mol": intensities_km_mol,
            "energy_ev": energy_ev,
        }

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run ORCA-specific frequency/IR calculation for one molecule."""
        data = self._run_orca_frequency_data(molecule)
        frequencies_cm1 = data["frequencies_cm1"]
        intensities_km_mol = data["intensities_km_mol"]

        imag_count = FrequencyIRAnalysisCalculation.count_imaginary_frequencies(frequencies_cm1)
        x, y = FrequencyIRAnalysisCalculation.build_ir_spectrum(
            frequencies_cm1=frequencies_cm1,
            intensities_km_mol=intensities_km_mol,
            sigma_cm1=self.sigma_cm1,
        )
        zpe, e_corr, h_corr, g_corr = FrequencyIRAnalysisCalculation.thermochemistry_corrections(
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
                    "backend": "orca",
                    "frequencies_cm1": frequencies_cm1,
                    "intensities_km_mol": intensities_km_mol,
                    "spectrum_wavenumber_cm1": x.tolist(),
                    "spectrum_intensity_au": y.tolist(),
                },
            )
        )


@dataclass
class FrequencyIRORCAWorkflow(PymatGenMaker):
    """ORCA frequency/IR workflow entry-point."""

    name: str = "ORCA Frequency/IR Workflow"
    calculator: ORCACalculator | None = None
    temperature: float = 298.15
    sigma_cm1: float = 20.0
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run ORCA-specific frequency/IR calculation."""
        calc = FrequencyIRORCACalculation(
            calculator=self.calculator,
            temperature=self.temperature,
            sigma_cm1=self.sigma_cm1,
        )
        return calc.make.original(calc, molecule)
