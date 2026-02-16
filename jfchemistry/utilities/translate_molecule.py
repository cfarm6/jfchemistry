"""Translate a pymatgen Molecule: user vector or center-of-mass/geometry to origin."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from pymatgen.core.structure import Molecule

from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties

# Translation vector (x, y, z)
TranslationVector = tuple[float, float, float] | list[float]

# Type alias matching JFChemMaker._operation return for override compatibility
_OpResult = tuple[Molecule | list[Molecule], Properties | list[Properties]]


@dataclass
class TranslateMolecule(PymatGenMaker[Molecule, Molecule]):
    """Translate a Molecule by a vector or move its center to the origin.

    Only supports pymatgen Molecule (not Structure). Modes:
    - "vector": apply a user-provided translation (dx, dy, dz).
    - "center_of_mass": move center of mass to the origin (mass-weighted).
    - "center_of_geometry": move centroid (center of geometry) to the origin.

    Set mode and translation as instance attributes; then call make(molecule) or make([mol1, ...]).
    """

    name: str = "TranslateMolecule"

    mode: Literal["vector", "center_of_mass", "center_of_geometry"] = "center_of_mass"
    translation: TranslationVector | None = None

    def _operation(self, input: Molecule, **kwargs: object) -> _OpResult:
        """Translate a molecule using this instance's mode and translation."""
        if input is None or not isinstance(input, Molecule):
            raise TypeError(
                "TranslateMolecule only supports Molecule inputs; "
                f"got {type(input).__name__ if input is not None else 'None'}"
            )
        atoms = input.to_ase_atoms()
        pos = np.asarray(atoms.get_positions(), dtype=float)

        if self.mode == "vector":
            if self.translation is None:
                raise ValueError("mode 'vector' requires translation (dx, dy, dz)")
            vec = np.asarray(self.translation, dtype=float)
            if vec.shape != (3,):
                raise ValueError("translation must be a length-3 vector")
            pos_new = pos + vec
        elif self.mode == "center_of_mass":
            masses = np.asarray(atoms.get_masses())
            com = np.average(pos, axis=0, weights=masses)
            pos_new = pos - com
        elif self.mode == "center_of_geometry":
            centroid = np.mean(pos, axis=0)
            pos_new = pos - centroid
        else:
            raise ValueError(
                f"mode must be 'vector', 'center_of_mass', or 'center_of_geometry'; "
                f"got {self.mode!r}"
            )

        atoms.set_positions(pos_new)
        return cast("_OpResult", (Molecule.from_ase_atoms(atoms), None))
