"""Partition coefficient workflow (fresh molecule-first implementation).

This workflow follows a FlexiSol-like staged protocol:
1) optional tautomer generation per phase
2) conformer generation per phase
3) optional conformer filtering
4) geometry optimization per phase
5) optional post-opt filtering
6) single-point energies per phase
7) Boltzmann reduction and partition-coefficient evaluation
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import OutputReference, Response
from pint import Quantity

from jfchemistry import SystemProperty, ureg
from jfchemistry.conformers import CRESTConformers
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.modification.tautomerization import CRESTTautomerization
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.single_point import ORCASinglePointCalculator

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

    from jfchemistry.conformers.base import ConformerGeneration
    from jfchemistry.filters import EnergyFilter, PrismPrunerFilter
    from jfchemistry.modification.tautomerization.base import TautomerMaker
    from jfchemistry.optimizers.base import GeometryOptimization
    from jfchemistry.single_point.base import SinglePointCalculation

# eV/K
KB_EV_PER_K = 8.617333262145e-5
# standard-state correction term (legacy value used in prior implementation)
G_STANDARD_STATE_EV = 1.89 * 2.611447e22 / 6.02214076e23


class PartitionCoefficientSystemProperty(PropertyClass):
    """Partition coefficient properties."""

    log_partition_coefficient: SystemProperty | OutputReference
    delta_g_transfer: SystemProperty | OutputReference


class PartitionCoefficientProperties(Properties):
    """Properties for partition-coefficient workflow."""

    system: PartitionCoefficientSystemProperty


@dataclass
class PartitionCoefficientReductionCalculation(PymatGenMaker):
    """Reduce phase conformer energies to log partition coefficient."""

    name: str = "Partition Coefficient Reduction"
    temperature: float = 298.15
    alpha_phase: str = "OCTANOL"
    beta_phase: str = "WATER"
    apply_standard_state_correction: bool = True
    _properties_model: type[PartitionCoefficientProperties] = PartitionCoefficientProperties
    _output_model: type[Output] = Output

    def __post_init__(self):
        """Normalize unit-bearing temperature inputs."""
        if isinstance(self.temperature, Quantity):
            object.__setattr__(self, "temperature", to_magnitude(self.temperature, "kelvin"))
        super().__post_init__()

    @staticmethod
    def _extract_energies_ev(properties_list: list[Properties], label: str) -> np.ndarray:
        vals: list[float] = []
        for i, prop in enumerate(properties_list):
            p = Properties.model_validate(prop, extra="allow", strict=False)
            if (
                p.system is None
                or not hasattr(p.system, "total_energy")
                or (te := getattr(p.system, "total_energy", None)) is None
                or getattr(te, "value", None) is None
            ):
                raise ValueError(f"Missing system.total_energy for {label}[{i}]")
            v = te.value
            vals.append(float(v.to(ureg.eV).magnitude) if hasattr(v, "to") else float(v))
        if len(vals) < 1:
            raise ValueError(f"No energies found for phase {label}")
        return np.asarray(vals, dtype=float)

    def _boltzmann_weighted_energy(self, energies_ev: np.ndarray, phase: str) -> float:
        energies = energies_ev.copy()
        if self.apply_standard_state_correction and phase.upper() != "GAS":
            energies = energies + G_STANDARD_STATE_EV
        kT = KB_EV_PER_K * self.temperature
        factors = np.exp(-(energies - np.min(energies)) / kT)
        weights = factors / factors.sum()
        return float(np.dot(energies, weights))

    @jfchem_job()
    def make(
        self,
        alpha_properties: list[Properties],
        beta_properties: list[Properties],
    ) -> Response[_output_model]:
        """Compute logP from phase conformer energies."""
        alpha_e = self._extract_energies_ev(alpha_properties, "alpha")
        beta_e = self._extract_energies_ev(beta_properties, "beta")

        g_alpha = self._boltzmann_weighted_energy(alpha_e, self.alpha_phase)
        g_beta = self._boltzmann_weighted_energy(beta_e, self.beta_phase)

        delta_g = g_beta - g_alpha
        log_p = delta_g / (self.temperature * KB_EV_PER_K * np.log(10.0))

        return Response(
            output=Output(
                properties=self._properties_model(
                    system=PartitionCoefficientSystemProperty(
                        log_partition_coefficient=SystemProperty(
                            name=f"logP({self.beta_phase}/{self.alpha_phase})",
                            value=float(log_p) * ureg.dimensionless,
                        ),
                        delta_g_transfer=SystemProperty(
                            name=f"ΔG transfer {self.alpha_phase}→{self.beta_phase}",
                            value=float(delta_g) * ureg.eV,
                        ),
                    )
                )
            )
        )


@dataclass
class PartitionCoefficientWorkflow(PymatGenMaker):
    """Molecule-first partition-coefficient workflow.

    make(structure: Molecule) is the only user-facing workflow input.
    """

    name: str = "Partition Coefficient Workflow"
    threads: int = 1
    temperature: float = 298.15
    alpha_phase: str = "OCTANOL"
    beta_phase: str = "WATER"

    tautomer_generator: Optional[TautomerMaker] = field(
        default_factory=lambda: CRESTTautomerization(threads=1)
    )
    conformer_generator: ConformerGeneration = field(
        default_factory=lambda: CRESTConformers(threads=1)
    )
    geometry_optimizer: GeometryOptimization = field(default_factory=lambda: ORCAOptimizer(cores=1))
    single_point: SinglePointCalculation = field(
        default_factory=lambda: ORCASinglePointCalculator(cores=1)
    )

    conformer_energy_filter: Optional[EnergyFilter] = None
    conformer_structural_filter: Optional[PrismPrunerFilter] = None
    optimized_energy_filter: Optional[EnergyFilter] = None
    optimized_structural_filter: Optional[PrismPrunerFilter] = None

    _properties_model: type[Properties] = Properties
    _output_model: type[Output] = Output

    def __post_init__(self):
        """Normalize unit-bearing temperature inputs."""
        if isinstance(self.temperature, Quantity):
            object.__setattr__(self, "temperature", to_magnitude(self.temperature, "kelvin"))
        super().__post_init__()

    @staticmethod
    def _phase_to_crest_solvation(phase: str) -> tuple[str, str]:
        return ("alpb", phase.lower())

    def _configure_for_phase(self, maker: Any, phase: str) -> Any:
        m = deepcopy(maker)
        # Common solvent-related knobs used across jfchemistry makers
        if hasattr(m, "solvent"):
            m.solvent = phase.upper()
        if hasattr(m, "solvation"):
            try:
                m.solvation = self._phase_to_crest_solvation(phase)
            except Exception:
                pass
        if hasattr(m, "threads"):
            m.threads = self.threads
        if hasattr(m, "cores"):
            m.cores = self.threads
        return m

    def _build_phase_pipeline(self, structure: Molecule, phase: str) -> tuple[list[Any], Any]:
        jobs: list[Any] = []

        current_structure = structure
        if self.tautomer_generator is not None:
            tauto_maker = self._configure_for_phase(self.tautomer_generator, phase)
            tautomer_job = tauto_maker.make(current_structure)
            print("TAUTOMER JOB UUID: ", tautomer_job.uuid)
            jobs.append(tautomer_job)
            current_structure = tautomer_job.output.structure

        conf_maker = self._configure_for_phase(self.conformer_generator, phase)
        conformer_job = conf_maker.make(current_structure)
        print("CONFORMER JOB UUID: ", conformer_job.uuid)
        jobs.append(conformer_job)
        phase_structures = conformer_job.output.structure

        if self.conformer_energy_filter is not None:
            ef_job = self.conformer_energy_filter.make(
                phase_structures,
                conformer_job.output.properties,
            )
            print("ENERGY FILTER JOB UUID: ", ef_job.uuid)
            jobs.append(ef_job)
            phase_structures = ef_job.output.structure

        if self.conformer_structural_filter is not None:
            sf_job = self.conformer_structural_filter.make(phase_structures)
            print("STRUCTURAL FILTER JOB UUID: ", sf_job.uuid)
            jobs.append(sf_job)
            phase_structures = sf_job.output.structure

        opt_maker = self._configure_for_phase(self.geometry_optimizer, phase)
        opt_job = opt_maker.make(phase_structures)
        print("OPTIMIZATION JOB UUID: ", opt_job.uuid)
        jobs.append(opt_job)
        optimized_structures = opt_job.output.structure
        optimized_properties = opt_job.output.properties

        if self.optimized_energy_filter is not None:
            oef_job = self.optimized_energy_filter.make(optimized_structures, optimized_properties)
            print("OPTIMIZED ENERGY FILTER JOB UUID: ", oef_job.uuid)
            jobs.append(oef_job)
            optimized_structures = oef_job.output.structure
            optimized_properties = oef_job.output.properties

        if self.optimized_structural_filter is not None:
            osf_job = self.optimized_structural_filter.make(optimized_structures)
            print("OPTIMIZED STRUCTURAL FILTER JOB UUID: ", osf_job.uuid)
            jobs.append(osf_job)
            optimized_structures = osf_job.output.structure

        sp_maker = self._configure_for_phase(self.single_point, phase)
        sp_job = sp_maker.make(optimized_structures)
        print("SINGLE-POINT JOB UUID: ", sp_job.uuid)
        jobs.append(sp_job)

        return jobs, sp_job.output.properties

    @jfchem_job()
    def make(self, structure: Molecule) -> Response[_output_model]:
        """Build partition workflow from a provided 3D molecule."""
        alpha_jobs, alpha_props = self._build_phase_pipeline(structure, self.alpha_phase)
        beta_jobs, beta_props = self._build_phase_pipeline(structure, self.beta_phase)

        reducer = PartitionCoefficientReductionCalculation(
            temperature=self.temperature,
            alpha_phase=self.alpha_phase,
            beta_phase=self.beta_phase,
        )
        reduce_job = reducer.make(alpha_props, beta_props)

        flow = Flow([*alpha_jobs, *beta_jobs, reduce_job], name=self.name)
        output = Output(
            structure=structure,
            files={
                "alpha_phase": self.alpha_phase,
                "beta_phase": self.beta_phase,
                "reduction_output": reduce_job.output,
            },
        )
        return Response(output=output, detour=flow)
