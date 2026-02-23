"""Workflows for the jfchemistry package."""

__all__ = [
    "FreeEnergyDifferenceWorkflow",
    "NelsonsFourPointMethod",
    "PartitionCoefficientWorkflow",
    "ReorganizationEnergyWorkflow",
]


def __getattr__(name: str):
    """Lazily import workflows to avoid importing optional heavy dependencies on import."""
    if name == "PartitionCoefficientWorkflow":
        from .partition_coefficient import PartitionCoefficientWorkflow

        return PartitionCoefficientWorkflow
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
