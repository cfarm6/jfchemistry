"""Tests for frequency/IR analysis workflow."""

import numpy as np

from jfchemistry.workflows.frequency_ir.workflow import FrequencyIRAnalysisCalculation


def test_backend_selector_prefers_pyscf_gpu() -> None:
    """Backend selector should prefer pyscf-gpu when available."""
    picked = FrequencyIRAnalysisCalculation.select_backend(
        prefer_gpu=True,
        available_backends=["orca", "pyscf-gpu", "ase"],
    )
    assert picked == "pyscf-gpu"


def test_imaginary_frequency_detection() -> None:
    """Imaginary count should exclude near-zero modes by threshold."""
    count = FrequencyIRAnalysisCalculation.count_imaginary_frequencies(
        [-50.0, -12.0, -5.0, 10.0, 1000.0],
        threshold_cm1=-10.0,
    )
    expected_imaginary = 2
    assert count == expected_imaginary


def test_gaussian_broadened_ir_spectrum_generation() -> None:
    """IR spectrum should be generated on requested grid with positive peaks."""
    npts = 2000
    x, y = FrequencyIRAnalysisCalculation.build_ir_spectrum(
        frequencies_cm1=[500.0, 1500.0],
        intensities_km_mol=[100.0, 250.0],
        sigma_cm1=15.0,
        wmin_cm1=0.0,
        wmax_cm1=3000.0,
        npts=npts,
    )
    assert len(x) == npts
    assert len(y) == npts
    assert float(np.max(y)) > 0.0


def test_thermo_corrections_consistency() -> None:
    """Thermal corrections should satisfy expected ordering relationships."""
    zpe, e_corr, h_corr, g_corr = FrequencyIRAnalysisCalculation.thermochemistry_corrections(
        frequencies_cm1=[200.0, 500.0, 1500.0, 3200.0],
        temperature=298.15,
    )
    assert zpe > 0.0
    assert e_corr >= zpe
    assert h_corr >= e_corr
    assert g_corr <= h_corr


def test_workflow_make_includes_backend_and_spectrum() -> None:
    """Reducer output should include backend selection and spectrum arrays."""
    calc = FrequencyIRAnalysisCalculation(temperature=298.15, sigma_cm1=20.0)
    response = calc.make.original(
        calc,
        frequencies_cm1=[300.0, 1200.0, 2800.0],
        intensities_km_mol=[50.0, 120.0, 90.0],
        available_backends=["pyscf-gpu", "ase"],
        prefer_gpu=True,
    )

    out = response.output
    assert out is not None
    assert out.properties is not None
    assert out.files is not None
    assert out.files["backend"] == "pyscf-gpu"
    assert len(out.files["spectrum_wavenumber_cm1"]) == len(out.files["spectrum_intensity_au"])
