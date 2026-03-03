"""Example: frequency/IR analysis workflow."""

from pathlib import Path

from jfchemistry.workflows.frequency_ir import FrequencyIRAnalysisWorkflow


def main() -> None:
    """Run IR spectrum + thermo analysis from sample vibrational data."""
    frequencies_cm1 = [120.0, 320.0, 780.0, 1210.0, 1580.0, 3020.0]
    intensities_km_mol = [15.0, 25.0, 80.0, 120.0, 95.0, 45.0]

    wf = FrequencyIRAnalysisWorkflow(temperature=298.15, sigma_cm1=18.0)
    response = wf.make.original(
        wf,
        frequencies_cm1=frequencies_cm1,
        intensities_km_mol=intensities_km_mol,
        available_backends=["pyscf-gpu", "ase"],
        prefer_gpu=True,
    )
    out = response.output

    print("Frequency/IR results")
    print("- Backend:", out.files["backend"])
    print("- Imaginary frequency count:", out.properties.system.imaginary_frequency_count.value)
    print("- ZPE (eV):", out.properties.system.zpe.value)

    x = out.files["spectrum_wavenumber_cm1"]
    y = out.files["spectrum_intensity_au"]
    lines = ["wavenumber_cm-1,intensity_au"]
    lines.extend(f"{xx:.6f},{yy:.8f}" for xx, yy in zip(x, y, strict=False))
    out_csv = Path("examples/frequency_ir_spectrum.csv")
    out_csv.write_text("\n".join(lines) + "\n")
    print(f"- Wrote spectrum CSV: {out_csv}")


if __name__ == "__main__":
    main()
