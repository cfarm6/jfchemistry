"""PySCF GPU calculator package.

Uses lazy import to avoid hard import-time failures in environments with
mismatched CuPy binary stacks.
"""

__all__ = ["PySCFGPUCalculator"]


def __getattr__(name: str):
    """Lazily import PySCF-GPU calculator symbols."""
    if name == "PySCFGPUCalculator":
        from .pyscfgpu_calculator import PySCFGPUCalculator

        return PySCFGPUCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
