"""Example: conformer ensemble thermochemistry reduction."""

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.conformer_ensemble import ConformerEnsembleWorkflow


class _SystemProperties(PropertyClass):
    total_energy: SystemProperty


class _EnergyProperties(Properties):
    system: _SystemProperties


def energy_properties(energy_ev: float) -> _EnergyProperties:
    """Build a minimal properties object with total energy in eV."""
    return _EnergyProperties(
        system=_SystemProperties(
            total_energy=SystemProperty(name="Total Energy", value=energy_ev * ureg.eV)
        )
    )


def main() -> None:
    """Run conformer ensemble workflow on synthetic conformer energies."""
    conformers = [
        energy_properties(-10.00),
        energy_properties(-9.98),
        energy_properties(-9.95),
    ]

    wf = ConformerEnsembleWorkflow(temperature=298.15)
    response = wf.make.original(wf, conformer_properties=conformers)
    out = response.output

    print("Conformer ensemble results")
    print("- Ensemble free energy (eV):", out.properties.system.ensemble_free_energy.value)
    print(
        "- Boltzmann-weighted energy (eV):",
        out.properties.system.boltzmann_weighted_energy.value,
    )
    print("- Boltzmann weights:", out.files["boltzmann_weights"])


if __name__ == "__main__":
    main()
