"""Base class for structural filters."""

from dataclasses import dataclass

from jfchemistry.filters.base import EnsembleFilter


@dataclass
class StructuralFilter(EnsembleFilter):
    """Base class for structural filters."""

    name: str = "Structural Filter"
