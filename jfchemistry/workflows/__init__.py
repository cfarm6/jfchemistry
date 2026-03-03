"""Workflows for the jfchemistry package."""

__all__ = [
    "ConformerEnsembleWorkflow",
    "FreeEnergyDifferenceWorkflow",
    "FrequencyIRASEWorkflow",
    "FrequencyIRAnalysisWorkflow",
    "NelsonsFourPointMethod",
    "PartitionCoefficientWorkflow",
    "RedoxPropertyWorkflow",
    "ReorganizationEnergyWorkflow",
]


def __getattr__(name: str):  # noqa: PLR0911
    """Lazily import workflows to avoid importing optional heavy dependencies on import."""
    if name == "PartitionCoefficientWorkflow":
        from .partition_coefficient import PartitionCoefficientWorkflow

        return PartitionCoefficientWorkflow
    if name == "ConformerEnsembleWorkflow":
        from .conformer_ensemble import ConformerEnsembleWorkflow

        return ConformerEnsembleWorkflow
    if name == "FrequencyIRAnalysisWorkflow":
        from .frequency_ir import FrequencyIRAnalysisWorkflow

        return FrequencyIRAnalysisWorkflow
    if name == "FrequencyIRASEWorkflow":
        from .frequency_ir import FrequencyIRASEWorkflow

        return FrequencyIRASEWorkflow
    if name == "RedoxPropertyWorkflow":
        from .redox import RedoxPropertyWorkflow

        return RedoxPropertyWorkflow
    if name == "FreeEnergyDifferenceWorkflow":
        from .free_energy_difference import FreeEnergyDifferenceWorkflow

        return FreeEnergyDifferenceWorkflow
    if name == "ReorganizationEnergyWorkflow":
        from .reorganization_energy import ReorganizationEnergyWorkflow

        return ReorganizationEnergyWorkflow
    if name == "NelsonsFourPointMethod":
        from .nelsons_four_point_method import NelsonsFourPointMethod

        return NelsonsFourPointMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
