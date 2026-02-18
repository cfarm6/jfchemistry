"""Helpers for normalizing float or Quantity inputs to magnitudes in a canonical unit.

Makers and calculators in jfchemistry accept unit-bearing parameters as either a
plain float (interpreted in the documented default unit) or a pint Quantity
(e.g. from ``jfchemistry.ureg`` or ``jfchemistry.Q_``). This module provides
``to_magnitude`` to normalize such inputs to a float magnitude for internal use
and serialization. See each maker/calculator class docstring for a "Units"
section listing which parameters require units and in what default unit.
"""

from __future__ import annotations

from typing import cast

from pint import Quantity

from jfchemistry import ureg


def to_magnitude(value: float | Quantity, default_unit: str) -> float:
    """Normalize a float or Quantity to a magnitude in the given unit.

    Accepts plain float (interpreted as being in default_unit) or a pint Quantity.
    Returns a float magnitude in default_unit for internal use and serialization.

    Args:
        value: Either a float in default_unit or a Quantity with compatible dimensions.
        default_unit: Unit string (e.g. "eV", "kcal_per_mol", "angstrom", "fs", "kelvin").

    Returns:
        Magnitude as float in default_unit.
    """
    if isinstance(value, Quantity):
        return cast("float", value.to(ureg(default_unit)).magnitude)
    return float(value)
