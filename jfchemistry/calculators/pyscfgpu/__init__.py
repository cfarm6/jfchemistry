"""PySCF GPU calculator package.

Uses lazy import to avoid hard import-time failures in environments with
mismatched CuPy binary stacks.
"""

__all__ = ["PySCFCalculator", "PySCFGPUCalculator"]


def __getattr__(name: str):
    """Lazily import PySCF calculator symbols."""
    if name in {"PySCFCalculator", "PySCFGPUCalculator"}:
        from .pyscfgpu_calculator import PySCFCalculator, PySCFGPUCalculator

        return {"PySCFCalculator": PySCFCalculator, "PySCFGPUCalculator": PySCFGPUCalculator}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
