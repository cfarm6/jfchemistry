"""Redox workflow for vertical and adiabatic IP/EA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from jobflow.core.job import OutputReference, Response
from pydantic import ConfigDict

from jfchemistry import SystemProperty, ureg
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass


class RedoxSystemProperties(PropertyClass):
    """System-level redox properties."""

    vertical_ip: SystemProperty | OutputReference
    vertical_ea: SystemProperty | OutputReference
    adiabatic_ip: SystemProperty | OutputReference
    adiabatic_ea: SystemProperty | OutputReference


class RedoxProperties(Properties):
    """Properties model for redox workflow."""

    system: RedoxSystemProperties


class RedoxOutput(Output):
    """Output for redox workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    files: Optional[Any] = None
    properties: Optional[RedoxProperties] = None


@dataclass
class RedoxPropertyCalculation(PymatGenMaker):
    """Reducer for vertical and adiabatic redox terms."""

    name: str = "Redox Property Calculation"
    _properties_model: type[RedoxProperties] = RedoxProperties
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
    def validate_charge_spin_states(  # noqa: PLR0913
        neutral_charge: int,
        cation_charge: int,
        anion_charge: int,
        neutral_spin: int,
        cation_spin: int,
        anion_spin: int,
    ) -> None:
        """Validate basic charge/spin consistency for redox states."""
        if cation_charge != neutral_charge + 1:
            raise ValueError("cation charge must equal neutral_charge + 1")
        if anion_charge != neutral_charge - 1:
            raise ValueError("anion charge must equal neutral_charge - 1")
        for name, spin in [
            ("neutral_spin", neutral_spin),
            ("cation_spin", cation_spin),
            ("anion_spin", anion_spin),
        ]:
            if spin < 1:
                raise ValueError(f"{name} must be >= 1")

    @classmethod
    def _compute_redox_terms(
        cls,
        neutral_relaxed: Properties,
        cation_relaxed: Properties,
        anion_relaxed: Properties,
        cation_on_neutral_geom: Properties,
        anion_on_neutral_geom: Properties,
    ) -> tuple[float, float, float, float]:
        en = cls._extract_total_energy_ev(neutral_relaxed, "neutral_relaxed")
        ec_relaxed = cls._extract_total_energy_ev(cation_relaxed, "cation_relaxed")
        ea_relaxed = cls._extract_total_energy_ev(anion_relaxed, "anion_relaxed")
        ec_vertical = cls._extract_total_energy_ev(cation_on_neutral_geom, "cation_on_neutral_geom")
        ea_vertical = cls._extract_total_energy_ev(anion_on_neutral_geom, "anion_on_neutral_geom")

        vertical_ip = ec_vertical - en
        vertical_ea = en - ea_vertical
        adiabatic_ip = ec_relaxed - en
        adiabatic_ea = en - ea_relaxed
        return vertical_ip, vertical_ea, adiabatic_ip, adiabatic_ea

    @jfchem_job()
    def make(  # noqa: PLR0913
        self,
        neutral_relaxed: Properties,
        cation_relaxed: Properties,
        anion_relaxed: Properties,
        cation_on_neutral_geom: Properties,
        anion_on_neutral_geom: Properties,
        neutral_charge: int = 0,
        cation_charge: int = 1,
        anion_charge: int = -1,
        neutral_spin: int = 1,
        cation_spin: int = 2,
        anion_spin: int = 2,
    ) -> Response[_output_model]:
        """Compute redox properties from state energies."""
        self.validate_charge_spin_states(
            neutral_charge=neutral_charge,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
            neutral_spin=neutral_spin,
            cation_spin=cation_spin,
            anion_spin=anion_spin,
        )
        v_ip, v_ea, a_ip, a_ea = self._compute_redox_terms(
            neutral_relaxed=neutral_relaxed,
            cation_relaxed=cation_relaxed,
            anion_relaxed=anion_relaxed,
            cation_on_neutral_geom=cation_on_neutral_geom,
            anion_on_neutral_geom=anion_on_neutral_geom,
        )
        return Response(
            output=RedoxOutput(
                properties=self._properties_model(
                    system=RedoxSystemProperties(
                        vertical_ip=SystemProperty(name="Vertical IP", value=v_ip * ureg.eV),
                        vertical_ea=SystemProperty(name="Vertical EA", value=v_ea * ureg.eV),
                        adiabatic_ip=SystemProperty(name="Adiabatic IP", value=a_ip * ureg.eV),
                        adiabatic_ea=SystemProperty(name="Adiabatic EA", value=a_ea * ureg.eV),
                    )
                )
            )
        )


@dataclass
class RedoxPropertyWorkflow(PymatGenMaker):
    """Workflow wrapper over redox reducer."""

    name: str = "Redox Property Workflow"
    _properties_model: type[RedoxProperties] = RedoxProperties
    _output_model: type[RedoxOutput] = RedoxOutput

    @jfchem_job()
    def make(  # noqa: PLR0913
        self,
        neutral_relaxed: Properties,
        cation_relaxed: Properties,
        anion_relaxed: Properties,
        cation_on_neutral_geom: Properties,
        anion_on_neutral_geom: Properties,
        neutral_charge: int = 0,
        cation_charge: int = 1,
        anion_charge: int = -1,
        neutral_spin: int = 1,
        cation_spin: int = 2,
        anion_spin: int = 2,
    ) -> Response[_output_model]:
        """Generate redox output from input state properties."""
        calc = RedoxPropertyCalculation()
        return calc.make.original(
            calc,
            neutral_relaxed=neutral_relaxed,
            cation_relaxed=cation_relaxed,
            anion_relaxed=anion_relaxed,
            cation_on_neutral_geom=cation_on_neutral_geom,
            anion_on_neutral_geom=anion_on_neutral_geom,
            neutral_charge=neutral_charge,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
            neutral_spin=neutral_spin,
            cation_spin=cation_spin,
            anion_spin=anion_spin,
        )
