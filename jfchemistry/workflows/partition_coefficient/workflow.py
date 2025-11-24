"""Partition coefficient calculation workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from jobflow.core.flow import Flow
from jobflow.core.job import OutputReference, Response

from jfchemistry import SingleStructureMaker, SystemProperty
from jfchemistry.conformers import CRESTConformers
from jfchemistry.conformers.base import ConformerGeneration
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.filters.energy import EnergyFilter
from jfchemistry.filters.structural.base import StructuralFilter
from jfchemistry.filters.structural.prism_filter import PrismPrunerFilter
from jfchemistry.modification.tautomers import CRESTTautomers
from jfchemistry.modification.tautomers.base import TautomerMaker
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.optimizers.base import GeometryOptimization
from jfchemistry.single_point import ORCASinglePointCalculator
from jfchemistry.single_point.base import SinglePointEnergyCalculator

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

    from jfchemistry.conformers.crest import SolvationType

    from .solvent_list import PartitionCoefficientSolventType

# CONSTANTS
R = 8.31446261815324 * 6.241509e18 / 6.02214076e23  # eV/K
kb = 8.617333262e-5  # eV/K
G = 1.89 * 2.611447e22 / 6.02214076e23  # eV

# Alpha Phase
_DEFAULT_ALPHA_TAUTOMER_GENERATOR = object()
AlphaTautomerGeneratorArg = Optional[TautomerMaker] | object
_DEFAULT_ALPHA_TAUTOMER_ENERGY = object()
AlphaTautomerEnergyArg = Optional[SinglePointEnergyCalculator] | object
_DEFAULT_ALPHA_CONFORMER_GENERATOR = object()
AlphaConformerGeneratorArg = Optional[ConformerGeneration] | object
_DEFAULT_ALPHA_CONFORMER_ENERGY = object()
AlphaConformerEnergyArg = Optional[SinglePointEnergyCalculator] | object
_DEFAULT_ALPHA_GEOMETRY_OPTIMIZER = object()
AlphaGeometryOptimizerArg = Optional[GeometryOptimization] | object
_DEFAULT_ALPHA_SINGLE_POINT_ENERGY = object()
AlphaSinglePointEnergyArg = SinglePointEnergyCalculator | object

# Beta Phase
_DEFAULT_BETA_TAUTOMER_GENERATOR = object()
BetaTautomerGeneratorArg = Optional[TautomerMaker] | object
_DEFAULT_BETA_TAUTOMER_ENERGY = object()
BetaTautomerEnergyArg = Optional[SinglePointEnergyCalculator] | object
_DEFAULT_BETA_CONFORMER_GENERATOR = object()
BetaConformerGeneratorArg = Optional[ConformerGeneration] | object
_DEFAULT_BETA_CONFORMER_ENERGY = object()
BetaConformerEnergyArg = Optional[SinglePointEnergyCalculator] | object
_DEFAULT_BETA_GEOMETRY_OPTIMIZER = object()
BetaGeometryOptimizerArg = Optional[GeometryOptimization] | object
_DEFAULT_BETA_SINGLE_POINT_ENERGY = object()
BetaSinglePointEnergyArg = SinglePointEnergyCalculator | object

# Filters
_DEFAULT_TAUTOMER_ENERGY_FILTER = object()
TautomerEnergyFilterArg = Optional[EnergyFilter] | object
_DEFAULT_TAUTOMER_STRUCTURAL_FILTER = object()
TautomerStructuralFilterArg = Optional[StructuralFilter] | object
_DEFAULT_CONFORMER_ENERGY_FILTER = object()
ConformerEnergyFilterArg = Optional[EnergyFilter] | object
_DEFAULT_CONFORMER_STRUCTURAL_FILTER = object()
ConformerStructuralFilterArg = Optional[StructuralFilter] | object
_DEFAULT_GEOMETRY_OPTIMIZER_ENERGY_FILTER = object()
GeometryOptimizerEnergyFilterArg = Optional[EnergyFilter] | object
_DEFAULT_GEOMETRY_OPTIMIZER_STRUCTURAL_FILTER = object()
GeometryOptimizerStructuralFilterArg = Optional[StructuralFilter] | object


@dataclass
class PhaseComponents:
    """Container for phase-specific workflow components."""

    tautomer_generator: Optional[TautomerMaker | object]
    tautomer_energy: Optional[SinglePointEnergyCalculator | object]
    conformer_generator: Optional[ConformerGeneration | object]
    conformer_energy: Optional[SinglePointEnergyCalculator | object]
    geometry_optimizer: Optional[GeometryOptimization | object]
    single_point_energy: Optional[SinglePointEnergyCalculator | object]


@dataclass
class FilterSet:
    """Container for workflow filters."""

    tautomer_energy: Optional[EnergyFilter | object]
    tautomer_structural: Optional[StructuralFilter | object]
    conformer_energy: Optional[EnergyFilter | object]
    conformer_structural: Optional[StructuralFilter | object]
    geometry_optimizer_energy: Optional[EnergyFilter | object]
    geometry_optimizer_structural: Optional[StructuralFilter | object]


class PartitionCoefficientSystemProperty(PropertyClass):
    """Partition coefficient."""

    partition_coefficient: SystemProperty | OutputReference


class PartitionCoefficientProperties(Properties):
    """Properties for the partition coefficient calculation."""

    system: PartitionCoefficientSystemProperty


@dataclass
class PartitionCoefficientCalculation(SingleStructureMaker):
    """Perform a partition coefficient calculation."""

    name: str = "Partition Coefficient Calculation"
    temperature: float = field(default=298.15, metadata={"description": "The temperature [K]."})
    alpha_phase: PartitionCoefficientSolventType = field(
        default="octanol", metadata={"description": "The alpha phase."}
    )
    beta_phase: PartitionCoefficientSolventType = field(
        default="water", metadata={"description": "The beta phase."}
    )
    _properties_model: type[PartitionCoefficientProperties] = PartitionCoefficientProperties
    _output_model: type[Output] = Output

    @jfchem_job()
    def make(
        self,
        alpha_properties: list[Properties] | Properties,
        beta_properties: list[Properties] | Properties,
    ) -> Response[_output_model]:
        """Make the partition coefficient calculation."""
        if not isinstance(alpha_properties, list):
            alpha_properties = [alpha_properties]
        if not isinstance(beta_properties, list):
            beta_properties = [beta_properties]
        _alpha_properties = [
            Properties.model_validate(property, extra="allow", strict=False)
            for property in alpha_properties
        ]
        _beta_properties = [
            Properties.model_validate(property, extra="allow", strict=False)
            for property in beta_properties
        ]
        alpha_energy = np.array(
            [property.system.total_energy.value * 27.2114 for property in _alpha_properties]
        )
        beta_energy = np.array(
            [property.system.total_energy.value * 27.2114 for property in _beta_properties]
        )
        if self.alpha_phase != "gas":
            alpha_energy += G
        if self.beta_phase != "gas":
            beta_energy += G
        # Boltzmann weighted average of the single point energy for each phase
        kT = kb * self.temperature
        alpha_boltz_factors = np.exp(-(alpha_energy - np.min(alpha_energy)) / kT)
        alpha_weights = alpha_boltz_factors / np.sum(alpha_boltz_factors)
        alpha_weighted_energy = np.sum(alpha_energy * alpha_weights)

        beta_boltz_factors = np.exp(-(beta_energy - np.min(beta_energy)) / kT)
        beta_weights = beta_boltz_factors / np.sum(beta_boltz_factors)
        beta_weighted_energy = np.sum(beta_energy * beta_weights)

        partition_coefficient = (beta_weighted_energy - alpha_weighted_energy) / (
            self.temperature * R
        )
        print(f"Partition Coefficient: {partition_coefficient}")
        return Response(
            output=Output(
                properties=self._properties_model(
                    system=PartitionCoefficientSystemProperty(
                        partition_coefficient=SystemProperty(
                            name=f"Partition Coefficient: {self.alpha_phase} - {self.beta_phase}",
                            value=partition_coefficient,
                            units="",
                        ),
                    )
                )
            )
        )


@dataclass
class PartitionCoefficientWorkflow(SingleStructureMaker):
    """Perform a partition coefficient calculation.

    For each phase, alpha and beta, the following steps are performed:
    1. OPTIONAL: Tautomer Generation
    2. OPTIONAL: Energy Pre-screening
    3. OPTIONAL: Structural Filtering
    4. OPTIONAL: Conformer Generation
    5. OPTIONAL: Energy Pre-screening
    6. OPTIONAL: Structural Filtering
    7. REQUIRED: Geometry Optimization
    8. OPTIONAL: Energy Screening
    9. OPTIONAL: Structural Filtering
    8. REQUIRED: Single Point Energy Calculation

    The Boltzmann weighted average of the single point energy for each phase is used to calculate the partition coefficient.

    The workflow defaults to the settings used for the generation of the FlexiSol dataset with the following exceptions:
    - Prism-Pruner is used instead of CENSO
    - CREST is used instead of GOAT

    """  # noqa: E501

    name: str = "Partition Coefficient Workflow"
    threads: int = 1
    alpha_phase: str = "octanol"
    beta_phase: str = "water"
    temperature: float = 298.15
    crest_executable: str = "crest"
    _properties_model: type[Properties] = Properties
    _output_model: type[Output] = Output

    @jfchem_job()
    def make(  # noqa: PLR0913
        self,
        structure: Molecule,
        alpha_tautomer_generator: AlphaTautomerGeneratorArg = _DEFAULT_ALPHA_TAUTOMER_GENERATOR,
        alpha_tautomer_energy: AlphaTautomerEnergyArg = _DEFAULT_ALPHA_TAUTOMER_ENERGY,
        alpha_conformer_generator: AlphaConformerGeneratorArg = _DEFAULT_ALPHA_CONFORMER_GENERATOR,
        alpha_conformer_energy: AlphaConformerEnergyArg = _DEFAULT_ALPHA_CONFORMER_ENERGY,
        alpha_geometry_optimizer: AlphaGeometryOptimizerArg = _DEFAULT_ALPHA_GEOMETRY_OPTIMIZER,
        alpha_single_point_energy: AlphaSinglePointEnergyArg = _DEFAULT_ALPHA_SINGLE_POINT_ENERGY,
        beta_tautomer_generator: BetaTautomerGeneratorArg = _DEFAULT_BETA_TAUTOMER_GENERATOR,
        beta_tautomer_energy: BetaTautomerEnergyArg = _DEFAULT_BETA_TAUTOMER_ENERGY,
        beta_conformer_generator: BetaConformerGeneratorArg = _DEFAULT_BETA_CONFORMER_GENERATOR,
        beta_conformer_energy: BetaConformerEnergyArg = _DEFAULT_BETA_CONFORMER_ENERGY,
        beta_geometry_optimizer: BetaGeometryOptimizerArg = _DEFAULT_BETA_GEOMETRY_OPTIMIZER,
        beta_single_point_energy: BetaSinglePointEnergyArg = _DEFAULT_BETA_SINGLE_POINT_ENERGY,
        tautomer_energy_filter: TautomerEnergyFilterArg = (_DEFAULT_TAUTOMER_ENERGY_FILTER),
        tautomer_structural_filter: TautomerStructuralFilterArg = (
            _DEFAULT_TAUTOMER_STRUCTURAL_FILTER
        ),
        conformer_energy_filter: ConformerEnergyFilterArg = (_DEFAULT_CONFORMER_ENERGY_FILTER),
        conformer_structural_filter: ConformerStructuralFilterArg = (
            _DEFAULT_CONFORMER_STRUCTURAL_FILTER
        ),
        geometry_optimizer_energy_filter: GeometryOptimizerEnergyFilterArg = (
            _DEFAULT_GEOMETRY_OPTIMIZER_ENERGY_FILTER
        ),
        geometry_optimizer_structural_filter: GeometryOptimizerStructuralFilterArg = (
            _DEFAULT_GEOMETRY_OPTIMIZER_STRUCTURAL_FILTER
        ),
    ) -> Response[_output_model]:
        """Make the partition coefficient workflow."""
        alpha_components = self._resolve_alpha_defaults(
            alpha_tautomer_generator=alpha_tautomer_generator,
            alpha_tautomer_energy=alpha_tautomer_energy,
            alpha_conformer_generator=alpha_conformer_generator,
            alpha_conformer_energy=alpha_conformer_energy,
            alpha_geometry_optimizer=alpha_geometry_optimizer,
            alpha_single_point_energy=alpha_single_point_energy,
        )
        beta_components = self._resolve_beta_defaults(
            beta_tautomer_generator=beta_tautomer_generator,
            beta_tautomer_energy=beta_tautomer_energy,
            beta_conformer_generator=beta_conformer_generator,
            beta_conformer_energy=beta_conformer_energy,
            beta_geometry_optimizer=beta_geometry_optimizer,
            beta_single_point_energy=beta_single_point_energy,
        )
        filters = self._resolve_filters(
            tautomer_energy_filter=tautomer_energy_filter,
            tautomer_structural_filter=tautomer_structural_filter,
            conformer_energy_filter=conformer_energy_filter,
            conformer_structural_filter=conformer_structural_filter,
            geometry_optimizer_energy_filter=geometry_optimizer_energy_filter,
            geometry_optimizer_structural_filter=geometry_optimizer_structural_filter,
        )
        alpha_jobs, final_alpha_job = self._build_phase_workflow(
            structure=structure,
            components=alpha_components,
            filters=filters,
        )
        beta_jobs, final_beta_job = self._build_phase_workflow(
            structure=structure,
            components=beta_components,
            filters=filters,
        )
        partition_coefficient_calculation = PartitionCoefficientCalculation(
            temperature=self.temperature,
            alpha_phase=self.alpha_phase,
            beta_phase=self.beta_phase,
        )
        if final_alpha_job is None or final_beta_job is None:
            raise ValueError("No final jobs found")
        final_job = partition_coefficient_calculation.make(
            final_alpha_job.output.properties,  # type: ignore
            final_beta_job.output.properties,  # type: ignore
        )
        flow = Flow([*alpha_jobs, *beta_jobs, final_job], name="Partition Coefficient Workflow")
        partition_coefficient = final_job.output.properties.system.partition_coefficient
        properties = self._properties_model(
            system=PartitionCoefficientSystemProperty(
                partition_coefficient=partition_coefficient,
            )
        )
        output = Output(
            properties=properties,
            structure=structure,
            files=[
                final_alpha_job.output.files if final_alpha_job.output is not None else None,  # type: ignore
                final_beta_job.output.files if final_beta_job.output is not None else None,  # type: ignore
            ],
        )
        return Response(output=output, detour=flow)

    @staticmethod
    def _alpb_solvation(solvent: str) -> SolvationType:
        return cast("SolvationType", ("alpb", solvent))

    def _resolve_alpha_defaults(  # noqa: PLR0913
        self,
        alpha_tautomer_generator: AlphaTautomerGeneratorArg,
        alpha_tautomer_energy: AlphaTautomerEnergyArg,
        alpha_conformer_generator: AlphaConformerGeneratorArg,
        alpha_conformer_energy: AlphaConformerEnergyArg,
        alpha_geometry_optimizer: AlphaGeometryOptimizerArg,
        alpha_single_point_energy: AlphaSinglePointEnergyArg,
    ) -> PhaseComponents:
        if alpha_tautomer_generator is _DEFAULT_ALPHA_TAUTOMER_GENERATOR:
            alpha_tautomer_generator = CRESTTautomers(
                threads=self.threads,
                executable=self.crest_executable,
                solvation=self._alpb_solvation(self.alpha_phase),
                name=f"Tautomer Generation: GFN2-xTB with ALPB:{self.alpha_phase}",
            )
        if alpha_tautomer_energy is _DEFAULT_ALPHA_TAUTOMER_ENERGY:
            alpha_tautomer_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="PBE",
                basis_set="DEF2_SV_P",
                ecp="DEF2ECP",
                dispersion_correction="D4",
                solvation_model="CPCM",
                solvent=self.alpha_phase.upper(),
                name=f"Tautomer Pre-screening: PBE/DEF2-SV(P)/CPCM:{self.alpha_phase}",
            )
        if alpha_conformer_generator is _DEFAULT_ALPHA_CONFORMER_GENERATOR:
            alpha_conformer_generator = CRESTConformers(
                threads=self.threads,
                executable=self.crest_executable,
                solvation=self._alpb_solvation(self.alpha_phase),
                name=f"Conformer Generation: GFN2-xTB with ALPB:{self.alpha_phase}",
            )
        if alpha_conformer_energy is _DEFAULT_ALPHA_CONFORMER_ENERGY:
            alpha_conformer_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="R2SCAN_3C",
                ecp="DEF2ECP",
                solvation_model="CPCM",
                solvent=self.alpha_phase.upper(),
                name=f"Conformer Pre-screening: R2SCAN-3C/CPCM:{self.alpha_phase}",
            )
        if alpha_geometry_optimizer is _DEFAULT_ALPHA_GEOMETRY_OPTIMIZER:
            alpha_geometry_optimizer = ORCAOptimizer(
                cores=self.threads,
                xc_functional="R2SCAN_3C",
                ecp="DEF2ECP",
                solvation_model="CPCM",
                solvent=self.alpha_phase.upper(),
                name=f"Geometry Optimization: R2SCAN-3C/CPCM:{self.alpha_phase}",
            )
        if alpha_single_point_energy is _DEFAULT_ALPHA_SINGLE_POINT_ENERGY:
            alpha_single_point_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="WB97M_V",
                basis_set="DEF2_TZVPPD",
                ecp="DEF2ECP",
                solvation_model="SMD",
                solvent=self.alpha_phase.upper(),
                name=f"Single Point Energy Calculation: WB97M-V/DEF2-TZVPPD/SMD:{self.alpha_phase}",
            )
        return PhaseComponents(
            tautomer_generator=alpha_tautomer_generator,
            tautomer_energy=alpha_tautomer_energy,
            conformer_generator=alpha_conformer_generator,
            conformer_energy=alpha_conformer_energy,
            geometry_optimizer=alpha_geometry_optimizer,
            single_point_energy=alpha_single_point_energy,
        )

    def _resolve_beta_defaults(  # noqa: PLR0913
        self,
        beta_tautomer_generator: BetaTautomerGeneratorArg,
        beta_tautomer_energy: BetaTautomerEnergyArg,
        beta_conformer_generator: BetaConformerGeneratorArg,
        beta_conformer_energy: BetaConformerEnergyArg,
        beta_geometry_optimizer: BetaGeometryOptimizerArg,
        beta_single_point_energy: BetaSinglePointEnergyArg,
    ) -> PhaseComponents:
        if beta_tautomer_generator is _DEFAULT_BETA_TAUTOMER_GENERATOR:
            beta_tautomer_generator = CRESTTautomers(
                threads=self.threads,
                executable=self.crest_executable,
                solvation=self._alpb_solvation(self.beta_phase),
                name=f"Tautomer Generation: GFN2-xTB with ALPB:{self.beta_phase}",
            )
        if beta_tautomer_energy is _DEFAULT_BETA_TAUTOMER_ENERGY:
            beta_tautomer_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="PBE",
                basis_set="DEF2_SV_P",
                ecp="DEF2ECP",
                dispersion_correction="D4",
                solvation_model="CPCM",
                solvent=self.beta_phase.upper(),
                name=f"Tautomer Pre-screening: PBE/DEF2-SV(P)/CPCM:{self.beta_phase}",
            )
        if beta_conformer_generator is _DEFAULT_BETA_CONFORMER_GENERATOR:
            beta_conformer_generator = CRESTConformers(
                threads=self.threads,
                executable=self.crest_executable,
                solvation=self._alpb_solvation(self.beta_phase),
                name=f"Conformer Generation: GFN2-xTB with ALPB:{self.beta_phase}",
            )
        if beta_conformer_energy is _DEFAULT_BETA_CONFORMER_ENERGY:
            beta_conformer_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="R2SCAN_3C",
                ecp="DEF2ECP",
                solvation_model="CPCM",
                solvent=self.beta_phase.upper(),
                name=f"Conformer Pre-screening: R2SCAN-3C/CPCM:{self.beta_phase}",
            )
        if beta_geometry_optimizer is _DEFAULT_BETA_GEOMETRY_OPTIMIZER:
            beta_geometry_optimizer = ORCAOptimizer(
                cores=self.threads,
                xc_functional="R2SCAN_3C",
                ecp="DEF2ECP",
                solvation_model="CPCM",
                solvent=self.beta_phase.upper(),
                name=f"Geometry Optimization: R2SCAN-3C/CPCM:{self.beta_phase}",
            )
        if beta_single_point_energy is _DEFAULT_BETA_SINGLE_POINT_ENERGY:
            beta_single_point_energy = ORCASinglePointCalculator(
                cores=self.threads,
                xc_functional="WB97M_V",
                basis_set="DEF2_TZVPPD",
                ecp="DEF2ECP",
                solvation_model="SMD",
                solvent=self.beta_phase.upper(),
                name=f"Single Point Energy Calculation: WB97M-V/DEF2-TZVPPD/SMD:{self.beta_phase}",
            )
        return PhaseComponents(
            tautomer_generator=beta_tautomer_generator,
            tautomer_energy=beta_tautomer_energy,
            conformer_generator=beta_conformer_generator,
            conformer_energy=beta_conformer_energy,
            geometry_optimizer=beta_geometry_optimizer,
            single_point_energy=beta_single_point_energy,
        )

    def _resolve_filters(  # noqa: PLR0913
        self,
        tautomer_energy_filter: TautomerEnergyFilterArg,
        tautomer_structural_filter: TautomerStructuralFilterArg,
        conformer_energy_filter: ConformerEnergyFilterArg,
        conformer_structural_filter: ConformerStructuralFilterArg,
        geometry_optimizer_energy_filter: GeometryOptimizerEnergyFilterArg,
        geometry_optimizer_structural_filter: GeometryOptimizerStructuralFilterArg,
    ) -> FilterSet:
        if tautomer_energy_filter is _DEFAULT_TAUTOMER_ENERGY_FILTER:
            tautomer_energy_filter = None
        if tautomer_structural_filter is _DEFAULT_TAUTOMER_STRUCTURAL_FILTER:
            tautomer_structural_filter = PrismPrunerFilter(
                energy_threshold=12,
            )
        if conformer_energy_filter is _DEFAULT_CONFORMER_ENERGY_FILTER:
            conformer_energy_filter = None
        if conformer_structural_filter is _DEFAULT_CONFORMER_STRUCTURAL_FILTER:
            conformer_structural_filter = PrismPrunerFilter(
                energy_threshold=6,
            )
        if geometry_optimizer_energy_filter is _DEFAULT_GEOMETRY_OPTIMIZER_ENERGY_FILTER:
            geometry_optimizer_energy_filter = None
        if geometry_optimizer_structural_filter is _DEFAULT_GEOMETRY_OPTIMIZER_STRUCTURAL_FILTER:
            geometry_optimizer_structural_filter = PrismPrunerFilter(
                energy_threshold=4,
            )
            tautomer_structural_filter = geometry_optimizer_structural_filter
        return FilterSet(
            tautomer_energy=tautomer_energy_filter,
            tautomer_structural=tautomer_structural_filter,
            conformer_energy=conformer_energy_filter,
            conformer_structural=conformer_structural_filter,
            geometry_optimizer_energy=geometry_optimizer_energy_filter,
            geometry_optimizer_structural=geometry_optimizer_structural_filter,
        )

    def _build_phase_workflow(
        self,
        *,
        structure: Molecule,
        components: PhaseComponents,
        filters: FilterSet,
    ) -> tuple[list, Optional[object]]:
        jobs: list = []
        properties = None
        final_job = None
        structure_state = structure

        def queue_job(condition, make_job, set_name=None, assign_final=False):
            nonlocal structure_state, properties, final_job
            if not condition:
                return None
            if set_name is not None:
                set_name()
            job = make_job()
            jobs.append(job)
            structure_state = job.output.structure
            properties = getattr(job.output, "properties", properties)
            if assign_final:
                final_job = job
            return job

        queue_job(
            isinstance(components.tautomer_generator, TautomerMaker),
            lambda: cast("TautomerMaker", components.tautomer_generator).make(structure_state),
        )
        queue_job(
            isinstance(components.tautomer_energy, SinglePointEnergyCalculator)
            and components.tautomer_generator is not None,
            lambda: cast("SinglePointEnergyCalculator", components.tautomer_energy).make(
                structure_state
            ),
        )
        queue_job(
            isinstance(filters.tautomer_energy, EnergyFilter)
            and components.tautomer_generator is not None,
            lambda: cast("EnergyFilter", filters.tautomer_energy).make(structure_state, properties),
            set_name=lambda: setattr(
                cast("EnergyFilter", filters.tautomer_energy),
                "name",
                "Tautomer Energy Filter",
            ),
        )
        queue_job(
            isinstance(filters.tautomer_structural, StructuralFilter)
            and components.tautomer_generator is not None,
            lambda: cast("StructuralFilter", filters.tautomer_structural).make(
                structure_state, properties
            ),
            set_name=lambda: setattr(
                cast("StructuralFilter", filters.tautomer_structural),
                "name",
                "Tautomer Structural Filter",
            ),
        )
        queue_job(
            isinstance(components.conformer_generator, ConformerGeneration),
            lambda: cast("ConformerGeneration", components.conformer_generator).make(
                structure_state
            ),
        )
        queue_job(
            isinstance(components.conformer_energy, SinglePointEnergyCalculator)
            and components.conformer_generator is not None,
            lambda: cast("SinglePointEnergyCalculator", components.conformer_energy).make(
                structure_state
            ),
        )
        queue_job(
            isinstance(filters.conformer_structural, StructuralFilter)
            and components.conformer_generator is not None,
            lambda: cast("StructuralFilter", filters.conformer_structural).make(
                structure_state, properties
            ),
            set_name=lambda: setattr(
                cast("StructuralFilter", filters.conformer_structural),
                "name",
                "Conformer Structural Filter",
            ),
        )
        queue_job(
            isinstance(filters.conformer_energy, EnergyFilter)
            and components.conformer_generator is not None,
            lambda: cast("EnergyFilter", filters.conformer_energy).make(
                structure_state, properties
            ),
            set_name=lambda: setattr(
                cast("EnergyFilter", filters.conformer_energy),
                "name",
                "Conformer Energy Filter",
            ),
        )
        queue_job(
            isinstance(components.geometry_optimizer, GeometryOptimization),
            lambda: cast("GeometryOptimization", components.geometry_optimizer).make(
                structure_state
            ),
        )
        queue_job(
            isinstance(filters.geometry_optimizer_energy, EnergyFilter)
            and components.geometry_optimizer is not None,
            lambda: cast("EnergyFilter", filters.geometry_optimizer_energy).make(
                structure_state, properties
            ),
            set_name=lambda: setattr(
                cast("EnergyFilter", filters.geometry_optimizer_energy),
                "name",
                "Geometry Optimizer Energy Filter",
            ),
        )
        queue_job(
            isinstance(filters.geometry_optimizer_structural, StructuralFilter)
            and components.geometry_optimizer is not None,
            lambda: cast("StructuralFilter", filters.geometry_optimizer_structural).make(
                structure_state, properties
            ),
            set_name=lambda: setattr(
                cast("StructuralFilter", filters.geometry_optimizer_structural),
                "name",
                "Geometry Optimizer Structural Filter",
            ),
        )
        queue_job(
            isinstance(components.single_point_energy, SinglePointEnergyCalculator)
            and components.geometry_optimizer is not None,
            lambda: cast("SinglePointEnergyCalculator", components.single_point_energy).make(
                structure_state
            ),
            assign_final=True,
        )
        return jobs, final_job
