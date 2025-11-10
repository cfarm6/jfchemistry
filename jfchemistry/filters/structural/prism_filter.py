"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
from prism_pruner.conformer_ensemble import ConformerEnsemble

from jfchemistry.base_jobs import Properties
from jfchemistry.filters.base import Ensemble, PropertyEnsemble
from jfchemistry.filters.structural.base import StructuralFilter

EH_TO_KCAL = 627.5096080305927


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
    method: Literal["MOI", "RMSD", "RMSD_RC"] = field(
        default="MOI", metadata={"description": "The method for the prism pruner."}
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
        import prism_pruner.pruner

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
        conformer_ensemble = ConformerEnsemble(coords, atoms)
        prune_method = getattr(prism_pruner.pruner, self._method_dict[self.method])
        args = [coords, atoms]
        kw_args = {
            "energies": energies,
            "max_dE": self.energy_threshold / EH_TO_KCAL,
        }
        if self.method == "RMSD_RC":
            from prism_pruner.graph_manipulations import graphize

            graph = graphize(conformer_ensemble.atoms, conformer_ensemble.coords[0])
            args.append(graph)
        if "RMSD" in self.method:
            kw_args["max_rmsd"] = self.structural_threshold
        elif self.method == "MOI":
            kw_args["max_deviation"] = self.structural_threshold
        _, mask = prune_method(*args, **kw_args)

        filtered_ensemble = [item for item, keep in zip(ensemble, mask, strict=False) if keep]
        if properties is not None:
            filtered_properties = [
                item for item, keep in zip(properties, mask, strict=False) if keep
            ]
            return filtered_ensemble, filtered_properties
        else:
            return filtered_ensemble, None
