"""Save structure or molecule(s) to disk with optional list suffixing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.input_types import RecursiveMoleculeList, RecursiveStructureList
from jfchemistry.core.makers import PymatGenMaker

FormatLiteral = Literal[
    "xyz", "cif", "mol", "mol2", "pdb", "poscar", "cssr", "json", "yaml", "toml"
]
VALID_FORMATS: tuple[FormatLiteral, ...] = (
    "xyz",
    "cif",
    "mol",
    "mol2",
    "pdb",
    "poscar",
    "cssr",
    "json",
    "yaml",
    "toml",
)


def _infer_fmt(path: Path) -> str:
    """Infer pymatgen format from file extension."""
    ext = path.suffix.lstrip(".").lower()
    fmt_map = {
        "xyz": "xyz",
        "cif": "cif",
        "mol": "mol",
        "mol2": "mol2",
        "pdb": "pdb",
        "poscar": "poscar",
        "cssr": "cssr",
    }
    return fmt_map.get(ext, "xyz")


def _paths_for_list(base_path: Path, n: int, suffix_fmt: str = "_{i}") -> list[Path]:
    """Generate paths for list of structures: base_0.ext, base_1.ext, ..."""
    stem = base_path.stem
    ext = base_path.suffix
    return [base_path.parent / f"{stem}{suffix_fmt.format(i=i)}{ext}" for i in range(n)]


@dataclass
class SaveToDisk[
    InputType: RecursiveStructureList | RecursiveMoleculeList,
    OutputType: RecursiveStructureList | RecursiveMoleculeList,
](PymatGenMaker[InputType, OutputType]):
    """Save a structure or molecule to disk; appends suffixes for lists.

    Accepts a single Structure/Molecule or a list. When given a list, each
    item is written to a file with an appended suffix (e.g. output_0.xyz,
    output_1.xyz). Format is inferred from the filename extension if not
    provided.

    Attributes:
        name: Maker name for jobflow.
        suffix_fmt: Format string for list indices, e.g. "_{i}" -> output_0.xyz.
            Must contain exactly one "{i}" placeholder.
        filename: Base filename for saving.
        fmt: Format to save as.
    """

    name: str = "SaveToDisk"
    suffix_fmt: str = "_{i}"
    filename: str | None = None
    fmt: FormatLiteral | None = None
    _ensemble: bool = True

    @staticmethod
    def _write_one(
        obj: Structure | Molecule,
        path: Path,
        fmt: str | None = None,
    ) -> str:
        """Write a single structure or molecule to path. Returns absolute path as string."""
        path = path.resolve()
        fmt = fmt or _infer_fmt(path)
        if fmt not in VALID_FORMATS:
            raise ValueError(f"Invalid format: {fmt}")
        obj.to(filename=str(path), fmt=cast("Any", fmt))
        return str(path)

    def _operation(self, input: InputType, **kwargs) -> tuple[OutputType | list[OutputType], None]:
        """Save structure(s) or molecule(s) to disk.

        Args:
            input: Single Structure/Molecule or list of them.
            **kwargs: Additional kwargs to pass to the operation.

        Returns:
            Response with output.structure (unchanged input) and output.files
            (list of written absolute paths).

        Raises:
            TypeError: If input is not a Structure/Molecule or list of them.
            ValueError: If filename is empty, list is empty, or suffix_fmt invalid.
        """
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if "{i}" not in self.suffix_fmt:
            raise ValueError(
                "suffix_fmt must contain exactly one '{i}' placeholder for list "
                f"indices; got {self.suffix_fmt!r}"
            )
        single = not isinstance(input, list)
        if single:
            if not isinstance(input, (Structure, Molecule)):
                raise TypeError(
                    f"input must be a Structure, Molecule, or list of them; "
                    f"got {type(input).__name__}"
                )
            items = [input]
        else:
            if len(input) == 0:
                raise ValueError("input list cannot be empty")
            for i, obj in enumerate(input):
                if not isinstance(obj, (Structure, Molecule)):
                    raise TypeError(
                        f"input[{i}] must be a Structure or Molecule; got {type(obj).__name__}"
                    )
            items = input

        base = Path(self.filename).resolve()

        if single:
            paths = [base]
        else:
            paths = _paths_for_list(base, len(items), self.suffix_fmt)

        written: list[str] = []
        for obj, path in zip(items, paths, strict=True):
            written.append(self._write_one(obj, path, fmt=self.fmt))

        return (cast("OutputType", input), None)
