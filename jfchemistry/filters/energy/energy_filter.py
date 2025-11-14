"""Base class for energy filters."""

from dataclasses import dataclass, field

import numpy as np

from jfchemistry.base_jobs import Properties
from jfchemistry.filters.base import Ensemble, EnsembleFilter, PropertyEnsemble

EH_TO_KCAL = 627.5096080305927


@dataclass
class EnergyFilter(EnsembleFilter):
    """Base class for energy filters."""

    name: str = "Energy Filter"
    threshold: float = field(
        default=0.0, metadata={"description": "The threshold for the energy filter [kcal/mol]."}
    )

    def operation(
        self, ensemble: Ensemble, properties: PropertyEnsemble
    ) -> tuple[Ensemble, PropertyEnsemble]:
        """Perform the energy filter operation on an ensemble."""
        parsed_properties = [
            Properties.model_validate(property, extra="allow", strict=False)
            for property in properties
        ]
        energies = np.array([property.system.total_energy.value for property in parsed_properties])
        minimum_energy = np.min(energies)
        delta_energy = energies - minimum_energy
        filtered_indices = np.where(delta_energy <= self.threshold / EH_TO_KCAL)[0]

        filtered_ensemble = [ensemble[index] for index in filtered_indices]
        filtered_properties = [parsed_properties[index] for index in filtered_indices]
        print("Removed", len(ensemble) - len(filtered_ensemble), "structures")
        print("Kept", len(filtered_ensemble), "structures")
        return filtered_ensemble, filtered_properties
