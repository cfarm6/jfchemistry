"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from jfchemistry import Q_, ureg
from jfchemistry.core.properties import Properties
from jfchemistry.filters.structural.base import StructuralFilter

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
    _ensemble_type = T | list[T]
    _ensemble_collection_type = _ensemble_type | list[_ensemble_type]

    def _operation(
        self,
        ensemble: _ensemble_collection_type,
        properties: PropertyEnsembleCollection,
    ) -> tuple[_ensemble_collection_type, PropertyEnsembleCollection]:
        """Perform the energy filter operation on an ensemble."""
        from prism_pruner.pruner import prune

        if properties is not None:
            parsed_properties = [
                Properties.model_validate(property, extra="allow", strict=False)
                for property in properties
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
        else:
            energies = None
        print(energies)
        coords = np.array([molecule.cart_coords for molecule in ensemble])
        atoms = np.array([s.name for s in ensemble[0].species])
        threshold_hartree = Q_(self.energy_threshold, "kcal_per_mol").to("hartree")
        _, mask = prune(
            coords,
            atoms,
            energies=energies.to("hartree").magnitude,
            max_dE=threshold_hartree.magnitude,
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
