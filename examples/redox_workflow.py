"""Example: redox workflow for vertical and adiabatic IP/EA."""

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.workflows.redox import RedoxPropertyWorkflow


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
    """Run redox reduction from synthetic neutral/cation/anion energies."""
    neutral_relaxed = energy_properties(-5.00)
    cation_relaxed = energy_properties(-4.20)
    anion_relaxed = energy_properties(-5.70)
    cation_on_neutral = energy_properties(-4.00)
    anion_on_neutral = energy_properties(-5.40)

    wf = RedoxPropertyWorkflow()
    response = wf.make.original(
        wf,
        neutral_relaxed=neutral_relaxed,
        cation_relaxed=cation_relaxed,
        anion_relaxed=anion_relaxed,
        cation_on_neutral_geom=cation_on_neutral,
        anion_on_neutral_geom=anion_on_neutral,
        neutral_charge=0,
        cation_charge=1,
        anion_charge=-1,
        neutral_spin=1,
        cation_spin=2,
        anion_spin=2,
    )
    out = response.output

    print("Redox results (eV)")
    print("- Vertical IP:", out.properties.system.vertical_ip.value)
    print("- Vertical EA:", out.properties.system.vertical_ea.value)
    print("- Adiabatic IP:", out.properties.system.adiabatic_ip.value)
    print("- Adiabatic EA:", out.properties.system.adiabatic_ea.value)


if __name__ == "__main__":
    main()
