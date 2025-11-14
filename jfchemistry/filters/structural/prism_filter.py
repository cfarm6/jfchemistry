"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from jfchemistry.base_jobs import Properties
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
        self, ensemble: Ensemble, properties: PropertyEnsemble | None
    ) -> tuple[Ensemble, PropertyEnsemble | None]:
        """Perform the energy filter operation on an ensemble."""
        from prism_pruner.pruner import prune

        if properties is not None:
            parsed_properties = [
                Properties.model_validate(property, extra="allow", strict=False)
                for property in properties
            ]
            energies = np.array(
                [property.system.total_energy.value for property in parsed_properties]
            )
        else:
            energies = None
        coords = np.array([molecule.cart_coords for molecule in ensemble])
        atoms = np.array([s.name for s in ensemble[0].species])

        kw_args = {
            "energies": energies,
            "max_dE": self.energy_threshold / EH_TO_KCAL,
            "debug_function": None,
            "logfunction": None,
        }
        if "RMSD_RC" in self.methods:
            kw_args["rot_corr_rmsd_pruning"] = True
        if "RMSD" in self.methods:
            kw_args["rmsd_pruning"] = True
        if "MOI" in self.methods:
            kw_args["moi_pruning"] = True

        _, mask = prune(*[coords, atoms], **kw_args)

        filtered_ensemble = [item for item, keep in zip(ensemble, mask, strict=False) if keep]
        if properties is not None:
            filtered_properties = [
                item for item, keep in zip(properties, mask, strict=False) if keep
            ]
            return filtered_ensemble, filtered_properties
        else:
            return filtered_ensemble, None
