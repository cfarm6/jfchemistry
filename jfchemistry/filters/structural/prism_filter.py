"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from jfchemistry.core.properties import Properties, PropertyClass
from jfchemistry.filters.base import Ensemble, PropertyEnsemble
from jfchemistry.filters.structural.base import StructuralFilter

EH_TO_KCAL = 627.5096080305927

type PruningOptions = Literal["MOI", "RMSD", "RMSD_RC"]


@dataclass
class PrismPrunerFilter(StructuralFilter):
    """PrismPrunerFilter."""

    name: str = "PrismPruner Structural Filter"
    structural_threshold: float = field(
        default=0.0, metadata={"description": "The threshold for the structural filter."}
    )
    energy_threshold: float = field(
        default=0.0, metadata={"description": "The threshold for the energy filter [kcal/mol]."}
    )
    methods: list[type[PruningOptions]] = field(
        default_factory=lambda: ["MOI", "RMSD"],
        metadata={"description": "The method for the prism pruner."},
    )
    _method_dict: ClassVar[dict[str, str]] = {
        "MOI": "prune_by_moment_of_inertia",
        "RMSD": "prune_by_rmsd",
        "RMSD_RC": "prune_by_rmsd_rot_corr",
    }

    def operation(
        self, ensemble: Ensemble, properties: PropertyEnsemble
    ) -> tuple[Ensemble, PropertyEnsemble]:
        """Perform the energy filter operation on an ensemble."""
        from prism_pruner.pruner import prune

        if properties is not None:
            parsed_properties = [
                Properties.model_validate(property, extra="allow", strict=False)
                for property in properties
            ]
            energies = np.array(
                [
                    property.system.total_energy.value
                    for property in parsed_properties
                    if property.system is PropertyClass
                ]
            )
        else:
            energies = None
        coords = np.array([molecule.cart_coords for molecule in ensemble])
        atoms = np.array([s.name for s in ensemble[0].species])

        _, mask = prune(
            coords,
            atoms,
            energies=energies,
            max_dE=self.energy_threshold / EH_TO_KCAL,
            debugfunction=None,
            logfunction=None,
            moi_pruning="MOI" in self.methods,
            rmsd_pruning="RMSD" in self.methods,
            rot_corr_rmsd_pruning="RMSD_RC" in self.methods,
        )

        filtered_ensemble = [item for item, keep in zip(ensemble, mask, strict=False) if keep]
        if properties is not None:
            filtered_properties = [
                item for item, keep in zip(properties, mask, strict=False) if keep
            ]
            return filtered_ensemble, filtered_properties
        else:
            return filtered_ensemble, None
