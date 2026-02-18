"""PrismPruner-based structural and energy filtering."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, cast

import numpy as np
from pint import Quantity
from prism_pruner.pruner import prune
from pymatgen.core.structure import Molecule

from jfchemistry import Q_, ureg
from jfchemistry.core.input_types import RecursiveMoleculeList
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties
from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.filters.base import Filter

type PruningOptions = Literal["MOI", "RMSD", "RMSD_RC"]


@dataclass
class PrismPrunerFilter[InputType: Molecule, OutputType: RecursiveMoleculeList](
    Filter,
    PymatGenMaker[InputType, OutputType],
):
    """PrismPruner-based structural and energy filter for molecular ensembles.

    Units:
        Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
        or ``jfchemistry.Q_``):

        - energy_threshold: [kcal/mol]

    Attributes:
        name: The name of the filter (default: "PrismPruner Structural Filter").
        structural_threshold: The threshold for the structural filter (default: 0.0).
            Accepts float or pint Quantity.
        energy_threshold: The threshold for the energy filter [kcal/mol] (default: None).
            Accepts float in [kcal/mol] or pint Quantity.
        methods: Pruning methods to apply: "MOI", "RMSD", "RMSD_RC" (default: ["MOI", "RMSD"]).
    """

    name: str = "PrismPruner Structural Filter"
    structural_threshold: float | Quantity = field(
        default=0.0,
        metadata={
            "description": "The threshold for the structural filter. "
            "Accepts float or pint Quantity.",
            "unit": "dimensionless",
        },
    )
    energy_threshold: Optional[float | Quantity] = field(
        default=None,
        metadata={
            "description": "The threshold for the energy filter [kcal/mol]. "
            "Accepts float in [kcal/mol] or pint Quantity.",
            "unit": "kcal/mol",
        },
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
        """Normalize unit-bearing attributes and set _ensemble."""
        if isinstance(self.structural_threshold, Quantity):
            object.__setattr__(
                self,
                "structural_threshold",
                to_magnitude(self.structural_threshold, "dimensionless"),
            )
        if self.energy_threshold is not None and isinstance(self.energy_threshold, Quantity):
            object.__setattr__(
                self, "energy_threshold", to_magnitude(self.energy_threshold, "kcal_per_mol")
            )
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
