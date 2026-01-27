"""Base class for energy filters."""

from dataclasses import dataclass, field

import numpy as np
from pymatgen.core.structure import Molecule, Structure

from jfchemistry import Q_, ureg
from jfchemistry.core.makers import EnsembleMaker
from jfchemistry.core.properties import Properties
from jfchemistry.filters.base import PropertyEnsembleCollection


@dataclass
class EnergyFilter[T: Structure | Molecule](EnsembleMaker[T]):
    """Base class for energy filters.

    Attributes:
        name: The name of the energy filter.
        threshold: The threshold for the energy filter [kcal/mol].
    """

    name: str = "Energy Filter"
    threshold: float = field(
        default=0.0, metadata={"description": "The threshold for the energy filter [kcal/mol]."}
    )
    _ensemble_type = T | list[T]
    _ensemble_collection_type = _ensemble_type | list[_ensemble_type]

    def _operation(
        self,
        ensemble: _ensemble_collection_type,  # type: ignore[assignment]
        properties: PropertyEnsembleCollection,
    ) -> tuple[_ensemble_collection_type, PropertyEnsembleCollection]:  # type: ignore[misc]
        """Perform the energy filter operation on an ensemble."""
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
        filtered_ensemble = np.array(ensemble)[filtered_indices].tolist()
        filtered_ensemble = [
            type(ensemble[idx]).from_sites(atoms) for idx, atoms in enumerate(filtered_ensemble)
        ]
        filtered_properties = np.array(parsed_properties)[filtered_indices]
        print("Removed", len(ensemble) - len(filtered_ensemble), "structures")
        print("Kept", len(filtered_ensemble), "structures")
        return filtered_ensemble, filtered_properties.tolist()
