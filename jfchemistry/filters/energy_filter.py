"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from pint import Quantity
from pymatgen.core.structure import Molecule, Structure

from jfchemistry import Q_, ureg
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.filters.base import Filter


@dataclass
class EnergyFilter[InputType: Structure | Molecule, OutputType: Structure | Molecule](
    Filter, PymatGenMaker[InputType, OutputType]
):
    """Base class for energy filters.

    Units:
        Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
        or ``jfchemistry.Q_``):

        - threshold: [kcal/mol]

    Attributes:
        name: The name of the energy filter.
        threshold: The threshold for the energy filter [kcal/mol]. Accepts float
            in [kcal/mol] or pint Quantity; stored as magnitude in [kcal/mol].
    """

    name: str = "Energy Filter"
    threshold: float | Quantity = field(
        default=0.0,
        metadata={
            "description": "The threshold for the energy filter [kcal/mol]. "
            "Accepts float in [kcal/mol] or pint Quantity.",
            "unit": "kcal/mol",
        },
    )

    def __post_init__(self):
        """Normalize threshold to magnitude and set _ensemble."""
        if isinstance(self.threshold, Quantity):
            object.__setattr__(self, "threshold", to_magnitude(self.threshold, "kcal_per_mol"))
        super().__post_init__()
        self._ensemble = True

    def _operation(self, input: InputType, **kwargs) -> tuple[OutputType, Properties]:
        """Perform the energy filter operation on an ensemble."""
        assert "properties" in kwargs, "Properties are required for the energy filter."
        properties = kwargs["properties"]
        parsed_properties = [
            Properties.model_validate(property, extra="allow", strict=False)
            for property in properties
            if property is not None
        ]
        energies = np.array(
            [
                te.value.magnitude
                for prop in parsed_properties
                if prop.system is not None
                and (te := getattr(prop.system, "total_energy", None)) is not None
                and te.value is not None
            ]
        ) * next(
            (
                te.value.units
                for prop in parsed_properties
                if prop.system is not None
                and (te := getattr(prop.system, "total_energy", None)) is not None
                and te.value is not None
            ),
            ureg.dimensionless,
        )
        threshold = Q_(self.threshold, "kcal_per_mol")
        minimum_energy = np.min(energies)
        delta_energy = energies - minimum_energy
        delta_energy = delta_energy.to("kcal_per_mol")
        filtered_indices = np.where(delta_energy <= threshold)
        filtered_ensemble = np.array(input)[filtered_indices].tolist()
        filtered_ensemble = [
            type(input[idx]).from_sites(atoms) for idx, atoms in enumerate(filtered_ensemble)
        ]
        filtered_properties = np.array(parsed_properties)[filtered_indices]
        return cast("OutputType", filtered_ensemble), cast(
            "Properties", filtered_properties.tolist()
        )
