"""Packmol Structure Packing.

This module provides integration with Packmol for packing molecules into
simulation boxes using box packing or fixed position packing.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.packing.base import StructurePacking


@dataclass
class PackmolPacking(StructurePacking):
    """Pack molecules using Packmol.

    This class provides an interface to Packmol for packing molecules into
    simulation boxes. It supports two packing modes:
    - Box packing: Pack multiple copies of a molecule into a defined box
    - Fixed packing: Place molecules at fixed positions

    Attributes:
        name: Name of the packing job (default: "Packmol Packing").
        packing_mode: Packing strategy to use:
            - "box": Pack molecules into a box (requires box_dimensions and num_molecules)
            - "fixed": Place molecules at fixed positions (requires fixed_positions)
        box_dimensions: Box size in Angstroms as (x, y, z) tuple. Required for box mode.
        num_molecules: Number of molecule copies to pack. Required for box mode.
        fixed_positions: List of (x, y, z) positions for fixed packing. Required for fixed mode.
        tolerance: Minimum distance between molecules in Angstroms (default: 2.0).
        packmol_path: Path to packmol executable (default: "packmol").
        filetype: Input/output file format (default: "xyz").

    Examples:
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> from jfchemistry.packing import PackmolPacking # doctest: +SKIP
        >>> water = Molecule.from_ase_atoms(molecule("H2O")) # doctest: +SKIP
        >>> # Box packing: pack 100 water molecules into a 20x20x20 Angstrom box
        >>> packer = PackmolPacking( # doctest: +SKIP
        ...     packing_mode="box", # doctest: +SKIP
        ...     box_dimensions=(20.0, 20.0, 20.0), # doctest: +SKIP
        ...     num_molecules=100, # doctest: +SKIP
        ...     tolerance=2.0 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = packer.make(water) # doctest: +SKIP
        >>> packed_structure = job.output["structure"] # doctest: +SKIP
        >>>
        >>> # Fixed packing: place water at specific position
        >>> packer_fixed = PackmolPacking( # doctest: +SKIP
        ...     packing_mode="fixed", # doctest: +SKIP
        ...     fixed_positions=[(10.0, 10.0, 10.0)], # doctest: +SKIP
        ...     tolerance=2.0 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>> job = packer_fixed.make(water) # doctest: +SKIP
    """

    name: str = "Packmol Packing"
    packing_mode: Literal["box", "fixed"] = field(
        default="box",
        metadata={"description": "the packing strategy to use"},
    )
    box_dimensions: Optional[tuple[float, float, float]] = field(
        default=None,
        metadata={"description": "box size in Angstroms as (x, y, z) tuple"},
    )
    num_molecules: Optional[int] = field(
        default=None,
        metadata={"description": "number of molecule copies to pack"},
    )
    fixed_positions: Optional[list[tuple[float, float, float]]] = field(
        default=None,
        metadata={"description": "list of (x, y, z) positions for fixed packing"},
    )
    tolerance: float = field(
        default=2.0,
        metadata={"description": "minimum distance between molecules in Angstroms"},
    )
    packmol_path: str = field(
        default="packmol",
        metadata={"description": "path to packmol executable"},
    )
    filetype: str = field(
        default="xyz",
        metadata={"description": "input/output file format"},
    )

    def _write_packmol_input(
        self, input_mol_file: str, output_file: str, structure: Molecule
    ) -> str:
        """Generate packmol input file.

        Args:
            input_mol_file: Path to input molecule file.
            output_file: Path to output packed structure file.
            structure: Input molecule structure.

        Returns:
            Path to the generated packmol input file.

        Raises:
            ValueError: If required parameters are missing for the selected mode.
        """
        if self.packing_mode == "box":
            if self.box_dimensions is None:
                raise ValueError("box_dimensions is required for box packing mode")
            if self.num_molecules is None:
                raise ValueError("num_molecules is required for box packing mode")
        elif self.packing_mode == "fixed":
            if self.fixed_positions is None:
                raise ValueError("fixed_positions is required for fixed packing mode")

        input_file = "packmol_input.inp"
        # Convert to absolute paths for packmol
        abs_input_mol_file = os.path.abspath(input_mol_file)
        abs_output_file = os.path.abspath(output_file)

        with open(input_file, "w") as f:
            f.write(f"tolerance {self.tolerance}\n")
            f.write(f"filetype {self.filetype}\n")
            f.write(f"output {abs_output_file}\n")
            f.write("\n")

            if self.packing_mode == "box":
                if self.box_dimensions is None:
                    raise ValueError("box_dimensions is required for box packing mode")
                f.write(f"structure {abs_input_mol_file}\n")
                f.write(f"  number {self.num_molecules}\n")
                f.write(
                    f"  inside box 0. 0. 0. {self.box_dimensions[0]} "
                    f"{self.box_dimensions[1]} {self.box_dimensions[2]}\n"
                )
                f.write("end structure\n")
            elif self.packing_mode == "fixed":
                if self.fixed_positions is None:
                    raise ValueError("fixed_positions is required for fixed packing mode")
                if len(self.fixed_positions) == 0:
                    raise ValueError("fixed_positions list cannot be empty")
                for i, pos in enumerate(self.fixed_positions):
                    f.write(f"structure {abs_input_mol_file}\n")
                    f.write("  number 1\n")
                    f.write(f"  fixed {pos[0]} {pos[1]} {pos[2]} 0. 0. 0.\n")
                    f.write("end structure\n")
                    if i < len(self.fixed_positions) - 1:
                        f.write("\n")

        return input_file

    def _run_packmol(self, input_file: str) -> None:
        """Execute packmol subprocess.

        Args:
            input_file: Path to packmol input file.

        Raises:
            RuntimeError: If packmol execution fails.
        """
        try:
            result = subprocess.run(
                [self.packmol_path, "-i", input_file],
                capture_output=True,
                text=True,
                check=False,
            )
            # Packmol returns non-zero exit codes even on success in some cases
            # Check if output file was created or if there's an actual error
            if result.returncode != 0:
                # Packmol writes errors to stdout
                error_output = result.stdout if result.stdout else result.stderr
                if "STOP" in error_output or "ERROR" in error_output.upper():
                    error_msg = (
                        f"Packmol execution failed (exit code {result.returncode}):\n{error_output}"
                    )
                    raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise RuntimeError(
                f"Packmol executable not found at '{self.packmol_path}'. "
                "Please ensure packmol is installed and in your PATH."
            ) from None

    def _read_packed_structure(self, output_file: str) -> Structure:
        """Read packmol output and convert to Structure.

        Args:
            output_file: Path to packmol output file.

        Returns:
            Pymatgen Structure from the packed output.

        Raises:
            FileNotFoundError: If output file doesn't exist.
        """
        if not os.path.exists(output_file):
            raise FileNotFoundError(
                f"Packmol output file not found: {output_file}. "
                "Packmol may have failed to generate the packed structure."
            )

        # Read the packed structure
        if self.filetype == "xyz":
            mol = Molecule.from_file(output_file)
            # Convert to Structure for periodic systems (box mode)
            # For fixed mode, we can return as Molecule or Structure
            if self.packing_mode == "box" and self.box_dimensions is not None:
                # Create a Structure with the box as the lattice
                from pymatgen.core import Lattice

                lattice = Lattice.cubic(self.box_dimensions[0])
                structure = Structure(
                    lattice=lattice,
                    species=mol.species,
                    coords=mol.cart_coords,
                    coords_are_cartesian=True,
                )
                return structure
            else:
                # For fixed packing, return as Structure with no lattice
                from pymatgen.core import Lattice

                # Create a large enough lattice to contain all molecules
                max_coords = mol.cart_coords.max(axis=0)
                min_coords = mol.cart_coords.min(axis=0)
                box_size = max_coords - min_coords + 10.0  # Add padding
                lattice = Lattice.cubic(max(box_size))
                structure = Structure(
                    lattice=lattice,
                    species=mol.species,
                    coords=mol.cart_coords,
                    coords_are_cartesian=True,
                )
                return structure
        else:
            # For other file types, try to read as Structure
            return Structure.from_file(output_file)

    def operation(self, structure: Molecule) -> tuple[Structure, Optional[dict[str, Any]]]:
        """Pack a structure using Packmol.

        Args:
            structure: The molecular structure to pack.

        Returns:
            A tuple containing the packed structure and a dictionary of properties.

        Raises:
            ValueError: If required parameters are missing.
            RuntimeError: If packmol execution fails.
        """
        # Validate mode-specific parameters
        # Validation will be done in _write_packmol_input

        # Write input molecule file
        input_mol_file = f"input_molecule.{self.filetype}"
        structure.to(filename=input_mol_file, fmt=self.filetype)

        # Generate packmol output filename
        output_file = f"packed_structure.{self.filetype}"

        # Write packmol input file
        packmol_input = self._write_packmol_input(input_mol_file, output_file, structure)

        # Run packmol
        self._run_packmol(packmol_input)

        # Read packed structure (use absolute path)
        abs_output_file = os.path.abspath(output_file)
        packed_structure = self._read_packed_structure(abs_output_file)

        # Prepare properties
        properties = self.get_properties(packed_structure)

        return packed_structure, properties

    def get_properties(self, structure: Structure) -> dict[str, Any]:
        """Get the properties of the packed structure.

        Args:
            structure: The packed structure.

        Returns:
            A dictionary containing packing metadata and structure properties.
        """
        properties: dict[str, Any] = {
            "packing_mode": self.packing_mode,
            "tolerance": self.tolerance,
            "num_atoms": len(structure),
        }

        if self.packing_mode == "box" and self.box_dimensions is not None:
            properties["box_dimensions"] = self.box_dimensions
            properties["num_molecules"] = self.num_molecules
            if all(d > 0 for d in self.box_dimensions):
                volume = self.box_dimensions[0] * self.box_dimensions[1] * self.box_dimensions[2]
                properties["density"] = len(structure) / volume
            else:
                properties["density"] = None
        elif self.packing_mode == "fixed" and self.fixed_positions is not None:
            properties["fixed_positions"] = self.fixed_positions
            properties["num_molecules"] = len(self.fixed_positions)

        return properties
