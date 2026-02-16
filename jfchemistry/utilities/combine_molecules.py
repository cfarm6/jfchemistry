"""Combine pymatgen Molecule objects into one."""

from dataclasses import dataclass, field
from typing import cast

from pymatgen.core.structure import Molecule

from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties

# Type alias matching JFChemMaker._operation return for override compatibility
_OpResult = tuple[Molecule | list[Molecule], Properties | list[Properties]]


@dataclass
class CombineMolecules(
    PymatGenMaker[Molecule | list[Molecule], Molecule],
):
    """Combine one or more Molecule objects into a single Molecule.

    Only supports pymatgen Molecule inputs (not periodic Structure).
    Molecules are appended in the order provided, in the same coordinate frame.
    """

    name: str = "CombineMolecules"
    _ensemble: bool = field(default=True)  # treat list as single input to combine

    def _operation(
        self,
        input: Molecule | list[Molecule],
        **kwargs: object,
    ) -> _OpResult:
        """Combine molecules into one."""
        if input is None:
            raise TypeError("molecules cannot be None")
        if isinstance(input, Molecule):
            items = [input]
        else:
            if len(input) == 0:
                raise ValueError("molecules list cannot be empty")
            for i, mol in enumerate(input):
                if mol is None:
                    raise TypeError(f"molecules[{i}] cannot be None")
                if not isinstance(mol, Molecule):
                    raise TypeError(
                        f"CombineMolecules only supports Molecule inputs; "
                        f"got molecules[{i}] as {type(mol).__name__}"
                    )
            items = input

        atoms = items[0].to_ase_atoms()
        for mol in items[1:]:
            atoms = atoms + mol.to_ase_atoms()
        combined = Molecule.from_ase_atoms(atoms)
        return cast("_OpResult", (combined, None))
