"""Base class for energy filters."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, cast

import numpy as np
from prism_pruner.pruner import prune
from pymatgen.core.structure import Molecule

from jfchemistry import Q_, ureg
from jfchemistry.core.input_types import RecursiveMoleculeList
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.filters.base import Filter

type PruningOptions = Literal["MOI", "RMSD", "RMSD_RC"]


@dataclass
class PrismPrunerFilter[InputType: Molecule, OutputType: RecursiveMoleculeList](
    Filter,
    PymatGenMaker[InputType, OutputType],
):
    """PrismPrunerFilter."""

    name: str = "PrismPruner Structural Filter"
    structural_threshold: float = field(
        default=0.0, metadata={"description": "The threshold for the structural filter."}
    )
    energy_threshold: Optional[float] = field(
        default=None, metadata={"description": "The threshold for the energy filter [kcal/mol]."}
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

    def __post_init__(self):
        """Ensure _ensemble is set to True for ensemble processing."""
        super().__post_init__()
        self._ensemble = True

    def _operation(
        self, input: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties] | None]:
        """Perform the energy filter operation on an ensemble."""
        properties = kwargs.get("properties", None)
        energies = None
        if self.energy_threshold is not None:
            assert "properties" in kwargs, "Properties are required for the prism pruner."
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
        coords = np.array([molecule.cart_coords for molecule in input])
        atoms = np.array([s.name for s in input[0].species])
        threshold_hartree = (
            Q_(self.energy_threshold, "kcal_per_mol").to("hartree").magnitude
            if self.energy_threshold is not None
            else 0.0
        )
        _, mask = prune(
            coords,
            atoms,
            energies=energies.to("hartree").magnitude if energies is not None else None,
            max_dE=threshold_hartree,
            debugfunction=None,
            logfunction=None,
            moi_pruning="MOI" in self.methods,
            rmsd_pruning="RMSD" in self.methods,
            rot_corr_rmsd_pruning="RMSD_RC" in self.methods,
        )

        filtered_ensemble = [item for item, keep in zip(input, mask, strict=False) if keep]
        if properties is not None:
            filtered_properties = [
                item for item, keep in zip(properties, mask, strict=False) if keep
            ]
            return filtered_ensemble, filtered_properties
        else:
            return cast("OutputType", filtered_ensemble), []
