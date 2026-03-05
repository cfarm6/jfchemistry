"""ASE-specific frequency and IR analysis workflow."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase.vibrations.infrared import Infrared
from jobflow.core.job import Response

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.workflows.frequency_ir.workflow import (
    CMINV_TO_EV,
    KB_EV_PER_K,
    FrequencyIRAnalysisCalculation,
    FrequencyIROutput,
    FrequencyIRProperties,
    FrequencyIRSystemProperties,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

    from jfchemistry.calculators.ase.ase_calculator import ASECalculator


@dataclass
class FrequencyIRASECalculation(PymatGenMaker):
    """Compute frequencies/IR with ASE and reduce thermochemistry terms."""

    name: str = "ASE Frequency/IR Calculation"
    calculator: ASECalculator | None = None
    temperature: float = 298.15
    pressure_pa: float = 101325.0
    sigma_cm1: float = 20.0
    displacement: float = 0.01
    nfree: int = 2
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @staticmethod
    def _guess_geometry(atoms_count: int) -> str:
        monatomic_max = 1
        linear_count = 2
        if atoms_count <= monatomic_max:
            return "monatomic"
        if atoms_count == linear_count:
            return "linear"
        return "nonlinear"

    def _run_ase_frequency_data(self, molecule: Molecule) -> dict[str, Any]:
        """Run ASE vibrational analysis and return frequencies/intensities/energy."""
        if self.calculator is None:
            raise ValueError("FrequencyIRASECalculation requires `calculator`.")

        atoms = molecule.to_ase_atoms()
        charge = int(molecule.charge)
        spin = int(molecule.spin_multiplicity) if molecule.spin_multiplicity else 1
        self.calculator._set_calculator(atoms, charge=charge, spin_multiplicity=spin)

        with tempfile.TemporaryDirectory(prefix="jfchem_freq_") as tmpdir:
            vib_name = str(Path(tmpdir) / "vib")
            ir_name = str(Path(tmpdir) / "ir")
            frequencies: list[float]
            intensities: list[float]

            try:
                ir = Infrared(atoms, name=ir_name, delta=self.displacement, nfree=self.nfree)
                ir.run()
                frequencies = [float(x) for x in ir.get_frequencies()]
                intensities: list[int | float] = [float(x) for x in ir.intensities]
                ir.clean()
            except Exception:
                vib = Vibrations(atoms, name=vib_name, delta=self.displacement, nfree=self.nfree)
                vib.run()
                frequencies = [float(x) for x in vib.get_frequencies()]
                intensities = [0.0 for _ in frequencies]
                vib.clean()

        energy_ev = float(atoms.get_potential_energy())
        return {
            "frequencies_cm1": frequencies,
            "intensities_km_mol": intensities,
            "energy_ev": energy_ev,
            "atoms": atoms,
        }

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run ASE frequency job for one molecule and compute IR/thermo outputs."""
        data = self._run_ase_frequency_data(molecule)
        frequencies_cm1 = data["frequencies_cm1"]
        intensities_km_mol = data["intensities_km_mol"]
        atoms = data["atoms"]
        energy_ev = data["energy_ev"]

        imag_count = FrequencyIRAnalysisCalculation.count_imaginary_frequencies(frequencies_cm1)
        x, y = FrequencyIRAnalysisCalculation.build_ir_spectrum(
            frequencies_cm1=frequencies_cm1,
            intensities_km_mol=intensities_km_mol,
            sigma_cm1=self.sigma_cm1,
        )

        vib_energies_ev = [f * CMINV_TO_EV for f in frequencies_cm1 if f > 1.0]
        thermo = IdealGasThermo(
            vib_energies=vib_energies_ev,
            potentialenergy=energy_ev,
            atoms=atoms,
            geometry=self._guess_geometry(len(atoms)),
            symmetrynumber=1,
            spin=max((int(molecule.spin_multiplicity) - 1) / 2, 0),
        )
        h_total = float(thermo.get_enthalpy(self.temperature, verbose=False))
        g_total = float(
            thermo.get_gibbs_energy(
                self.temperature,
                pressure=self.pressure_pa,
                verbose=False,
            )
        )
        zpe = float(thermo.get_ZPE_correction())
        # Ideal-gas relation: H = U + k_B T (per molecule)
        e_total = h_total - KB_EV_PER_K * self.temperature
        e_corr = e_total - energy_ev
        h_corr = h_total - energy_ev
        g_corr = g_total - energy_ev

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
                    "backend": "ase",
                    "frequencies_cm1": frequencies_cm1,
                    "intensities_km_mol": intensities_km_mol,
                    "spectrum_wavenumber_cm1": x.tolist(),
                    "spectrum_intensity_au": y.tolist(),
                },
            )
        )


@dataclass
class FrequencyIRASEWorkflow(PymatGenMaker):
    """ASE frequency/IR workflow entry-point."""

    name: str = "ASE Frequency/IR Workflow"
    calculator: ASECalculator | None = None
    temperature: float = 298.15
    pressure_pa: float = 101325.0
    sigma_cm1: float = 20.0
    displacement: float = 0.01
    nfree: int = 2
    _properties_model: type[FrequencyIRProperties] = FrequencyIRProperties
    _output_model: type[FrequencyIROutput] = FrequencyIROutput

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Run ASE-specific frequency/IR calculation."""
        calc = FrequencyIRASECalculation(
            calculator=self.calculator,
            temperature=self.temperature,
            pressure_pa=self.pressure_pa,
            sigma_cm1=self.sigma_cm1,
            displacement=self.displacement,
            nfree=self.nfree,
        )
        return calc.make.original(calc, molecule)
