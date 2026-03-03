"""Redox workflow for vertical and adiabatic IP/EA."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

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
    properties: Optional[Any] = None


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
    """Redox workflow from a single Molecule using optimizer + single-point makers."""

    name: str = "Redox Property Workflow"
    optimizer: PymatGenMaker | None = None
    single_point: PymatGenMaker | None = None
    neutral_charge: int | None = None
    neutral_spin_multiplicity: int | None = None
    cation_charge: int | None = None
    cation_spin_multiplicity: int | None = None
    anion_charge: int | None = None
    anion_spin_multiplicity: int | None = None
    _properties_model: type[RedoxProperties] = RedoxProperties
    _output_model: type[RedoxOutput] = RedoxOutput

    def _resolve_states(self, molecule: Molecule) -> tuple[int, int, int, int, int, int]:
        base_charge = int(molecule.charge)
        base_spin = int(molecule.spin_multiplicity) if molecule.spin_multiplicity is not None else 1
        neutral_charge = self.neutral_charge if self.neutral_charge is not None else base_charge
        neutral_spin = (
            self.neutral_spin_multiplicity
            if self.neutral_spin_multiplicity is not None
            else base_spin
        )
        cation_charge = self.cation_charge if self.cation_charge is not None else neutral_charge + 1
        anion_charge = self.anion_charge if self.anion_charge is not None else neutral_charge - 1
        cation_spin = (
            self.cation_spin_multiplicity
            if self.cation_spin_multiplicity is not None
            else max(2, neutral_spin)
        )
        anion_spin = (
            self.anion_spin_multiplicity
            if self.anion_spin_multiplicity is not None
            else max(2, neutral_spin)
        )
        RedoxPropertyCalculation.validate_charge_spin_states(
            neutral_charge=neutral_charge,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
            neutral_spin=neutral_spin,
            cation_spin=cation_spin,
            anion_spin=anion_spin,
        )
        return neutral_charge, neutral_spin, cation_charge, cation_spin, anion_charge, anion_spin

    @staticmethod
    def _set_state(molecule: Molecule, charge: int, spin: int) -> Molecule:
        state = molecule.copy()
        if hasattr(state, "_charge_spin_check"):
            object.__setattr__(state, "_charge_spin_check", False)
        state.set_charge_and_spin(charge, spin)
        return state

    def _with_state(
        self,
        maker: PymatGenMaker,
        charge: int,
        spin: int,
        relax: bool,
    ) -> PymatGenMaker:
        m = deepcopy(maker)
        if hasattr(m, "charge"):
            m.charge = charge
        if hasattr(m, "spin_multiplicity"):
            m.spin_multiplicity = spin
        if hasattr(m, "steps") and not relax:
            m.steps = 0
        return m

    def _build_flow(self, molecule: Molecule) -> tuple[Flow, RedoxOutput]:
        if self.optimizer is None:
            raise ValueError("RedoxPropertyWorkflow requires an `optimizer` attribute.")
        if self.single_point is None:
            raise ValueError("RedoxPropertyWorkflow requires a `single_point` attribute.")

        n_q, n_s, c_q, c_s, a_q, a_s = self._resolve_states(molecule)

        neutral_relaxed_job = self._with_state(self.optimizer, n_q, n_s, True).make(
            self._set_state(molecule, n_q, n_s)
        )
        cation_relaxed_job = self._with_state(self.optimizer, c_q, c_s, True).make(
            self._set_state(molecule, c_q, c_s)
        )
        anion_relaxed_job = self._with_state(self.optimizer, a_q, a_s, True).make(
            self._set_state(molecule, a_q, a_s)
        )

        cation_vertical_job = self._with_state(self.single_point, c_q, c_s, False).make(
            neutral_relaxed_job.output.structure
        )
        anion_vertical_job = self._with_state(self.single_point, a_q, a_s, False).make(
            neutral_relaxed_job.output.structure
        )

        reducer = RedoxPropertyCalculation()
        final_job = reducer.make(
            neutral_relaxed=neutral_relaxed_job.output.properties,
            cation_relaxed=cation_relaxed_job.output.properties,
            anion_relaxed=anion_relaxed_job.output.properties,
            cation_on_neutral_geom=cation_vertical_job.output.properties,
            anion_on_neutral_geom=anion_vertical_job.output.properties,
            neutral_charge=n_q,
            cation_charge=c_q,
            anion_charge=a_q,
            neutral_spin=n_s,
            cation_spin=c_s,
            anion_spin=a_s,
        )

        flow = Flow(
            [
                neutral_relaxed_job,
                cation_relaxed_job,
                anion_relaxed_job,
                cation_vertical_job,
                anion_vertical_job,
                final_job,
            ],
            name=self.name,
        )
        output = RedoxOutput(
            structure=neutral_relaxed_job.output.structure,
            properties=final_job.output.properties,
            files={
                "neutral_relaxed": neutral_relaxed_job.output.files,
                "cation_relaxed": cation_relaxed_job.output.files,
                "anion_relaxed": anion_relaxed_job.output.files,
                "cation_on_neutral_geom": cation_vertical_job.output.files,
                "anion_on_neutral_geom": anion_vertical_job.output.files,
            },
        )
        return flow, output

    @jfchem_job()
    def make(self, molecule: Molecule) -> Response[_output_model]:
        """Create redox workflow from a single pymatgen Molecule input."""
        flow, output = self._build_flow(molecule)
        return Response(output=output, detour=flow)
