"""Packmol Structure Packing.

This module provides integration with Packmol for packing molecules into
simulation boxes using box packing or fixed position packing.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, cast

from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.makers.single_maker import SingleJFChemistryMaker
from jfchemistry.core.properties import Properties
from jfchemistry.packing.base import StructurePacking


class PackmolProperties(Properties):
    packing_mode: Literal["box", "fixed"]
    tolerance: float
    num_atoms: int
    num_molecules: int
    box_dimensions: tuple[float, float, float]
    density: float
    target_density: float
    fixed_positions: list[tuple[float, float, float]]


@dataclass
class PackmolPacking[InputType: Molecule | Structure, OutputType: Molecule | Structure](
    SingleJFChemistryMaker[InputType, OutputType], StructurePacking
):
    """Pack molecules using Packmol.

    This class provides an interface to Packmol for packing molecules into
    simulation boxes. It supports two packing modes:
    - Box packing: Pack multiple copies of a molecule into a defined box
    - Fixed packing: Place molecules at fixed positions

    Attributes:
        name: Name of the packing job (default: "Packmol Packing").
        packing_mode: Packing strategy to use:
            - "box": Pack molecules into a box (requires num_molecules and either
              box_dimensions or density)
            - "fixed": Place molecules at fixed positions (requires fixed_positions)
        box_dimensions: Box size in Angstroms as (x, y, z) tuple. Required for box mode
            if density is not specified.
        density: Target density in g/cm^3. If specified, box dimensions are automatically
            calculated for a cubic box. Cannot be used together with box_dimensions.
        num_molecules: Number of molecule copies to pack. Required for box mode.
        fixed_positions: List of (x, y, z) positions for fixed packing. Required for fixed mode.
        tolerance: Minimum distance between molecules in Angstroms (default: 2.0).
        packmol_path: Path to packmol executable (default: "packmol").
        filetype: Input/output file format (default: "xyz").
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
    density: Optional[float] = field(
        default=None,
        metadata={"description": "target density in g/cm^3 (alternative to box_dimensions)"},
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
    _filetype: str = "xyz"

    def _calculate_box_dimensions_from_density(
        self, structure: Molecule
    ) -> tuple[float, float, float]:
        """Calculate box dimensions from target density.

        Args:
            structure: Input molecule structure.

        Returns:
            Box dimensions as (x, y, z) tuple in Angstroms for a cubic box.

        Raises:
            ValueError: If num_molecules or density is not set.
        """
        if self.num_molecules is None:
            raise ValueError("num_molecules is required when using density")
        if self.density is None:
            raise ValueError("density is required for density-based box calculation")

        # Avogadro's number
        AVOGADRO = 6.02214076e23  # mol^-1

        # Calculate molecular weight in g/mol
        molecular_weight = structure.composition.weight.real  # g/mol

        # Calculate total mass in grams
        total_mass = (self.num_molecules * molecular_weight) / AVOGADRO  # g

        # Calculate volume in cm^3
        volume_cm3 = total_mass / self.density  # cm^3

        # Convert to Angstrom^3 (1 cm^3 = 1e24 Angstrom^3)
        volume_ang3 = volume_cm3 * 1e24  # Angstrom^3

        # Calculate cubic box side length
        side_length = volume_ang3 ** (1.0 / 3.0)  # Angstrom

        self.box_dimensions = (side_length, side_length, side_length)
        return self.box_dimensions

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
            # At least one of box_dimensions or density must be set
            if self.box_dimensions is None:
                raise ValueError(
                    "Either box_dimensions or density must be specified for box packing mode"
                )
        elif self.packing_mode == "fixed":
            if self.fixed_positions is None:
                raise ValueError("fixed_positions is required for fixed packing mode")

        input_file = "packmol_input.inp"
        # Convert to absolute paths for packmol
        abs_input_mol_file = os.path.abspath(input_mol_file)
        abs_output_file = os.path.abspath(output_file)

        with open(input_file, "w") as f:
            f.write(f"tolerance {self.tolerance}\n")
            f.write(f"filetype {self._filetype}\n")
            f.write(f"output {abs_output_file}\n")
            f.write("\n")

            if self.packing_mode == "box":
                if self.box_dimensions is None:
                    raise ValueError("box_dimensions is required for box packing mode")
                if self.num_molecules is None:
                    raise ValueError("num_molecules is required for box packing mode")
                f.write(f"structure {abs_input_mol_file}\n")
                f.write(f"  number {self.num_molecules}\n")
                f.write(
                    f"  inside box 0. 0. 0. {self.box_dimensions[0]} \
                        {self.box_dimensions[1]} {self.box_dimensions[2]}\n"
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
                capture_output=False,
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

    def _read_packed_structure(
        self, output_file: str, box_dimensions: Optional[tuple[float, float, float]] = None
    ) -> OutputType:
        """Read packmol output and convert to Structure.

        Args:
            output_file: Path to packmol output file.
            box_dimensions: Box dimensions to use for creating the lattice.
                If None, will use self.box_dimensions or calculate from density.

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
        if self._filetype == "xyz":
            mol = Molecule.from_file(output_file)
            # Convert to Structure for periodic systems (box mode)
            # For fixed mode, we can return as Molecule or Structure
            if self.packing_mode == "box":
                # Use provided box_dimensions or fall back to self.box_dimensions
                if box_dimensions is None:
                    box_dimensions = self.box_dimensions

                if box_dimensions is not None:
                    # Create a Structure with the box as the lattice
                    from pymatgen.core import Lattice

                    # Create orthorhombic lattice with the box dimensions
                    lattice = Lattice.orthorhombic(
                        box_dimensions[0], box_dimensions[1], box_dimensions[2]
                    )
                    structure = Structure(
                        lattice=lattice,
                        species=mol.species,
                        coords=mol.cart_coords,
                        coords_are_cartesian=True,
                    )
                    return cast("OutputType", structure)
                else:
                    # Fallback: create a large enough lattice
                    from pymatgen.core import Lattice

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
                    return cast("OutputType", structure)
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
                return cast("OutputType", structure)
        else:
            # For other file types, try to read as Structure
            return cast("OutputType", Structure.from_file(output_file))

    def _operation(
        self, structure: InputType, **kwargs
    ) -> tuple[OutputType | list[OutputType], Properties | list[Properties]]:
        """Pack a structure using Packmol.

        Args:
            structure: The molecular structure to pack.

        Returns:
            A tuple containing the packed structure and properties.

        Raises:
            ValueError: If required parameters are missing.
            RuntimeError: If packmol execution fails.
        """
        # Validate mode-specific parameters
        # Validation will be done in _write_packmol_input

        # Write input molecule file
        input_mol_file = f"input_molecule.{self._filetype}"
        # assert isinstance(structure, Structure), "structure must be a molecule for packing"
        structure.to(filename=input_mol_file, fmt=self._filetype)

        # Generate packmol output filename
        output_file = f"packed_structure.{self._filetype}"

        # Get box dimensions (either specified or calculated from density)
        if self.packing_mode == "box":
            if self.density is not None:
                box_dims = self._calculate_box_dimensions_from_density(structure)
            else:
                box_dims = self.box_dimensions
        else:
            box_dims = None

        # Write packmol input file (this may calculate box_dimensions from density)
        packmol_input = self._write_packmol_input(input_mol_file, output_file, structure)

        # Run packmol
        self._run_packmol(packmol_input)

        # Read packed structure (use absolute path)
        abs_output_file = os.path.abspath(output_file)
        packed_structure = self._read_packed_structure(abs_output_file, box_dims)

        # Convert Structure to Molecule if needed (remove lattice for non-periodic representation)
        # For packed structures, we typically want to keep them as Structures,
        # but the base class signature requires Molecule. We'll convert to Molecule.
        # if isinstance(packed_structure, Structure):
        #     spin_multiplicity: int | None = None
        #     sp = getattr(packed_structure, "spin_multiplicity", None)
        #     if sp is not None:
        #         spin_multiplicity = int(sp)
        #     packed_molecule = Molecule(
        #         species=packed_structure.species,
        #         coords=packed_structure.cart_coords,
        #         charge=packed_structure.charge,
        #         spin_multiplicity=spin_multiplicity,
        #     )
        # else:
        #     packed_molecule = packed_structure

        # Prepare properties and convert to Properties object
        properties = self._get_properties(packed_structure)

        return packed_structure, properties

    def _get_properties(self, structure: Structure) -> Properties:
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

        if self.packing_mode == "box":
            properties["num_molecules"] = self.num_molecules

            # Get box dimensions from structure lattice or use specified/calculated values
            if hasattr(structure, "lattice") and structure.lattice is not None:
                # Extract box dimensions from lattice
                box_dims = (
                    structure.lattice.a,
                    structure.lattice.b,
                    structure.lattice.c,
                )
                properties["box_dimensions"] = box_dims
            elif self.box_dimensions is not None:
                box_dims = self.box_dimensions
                properties["box_dimensions"] = box_dims
            else:
                box_dims = None

            # Calculate density from packed structure
            if box_dims is not None and all(d > 0 for d in box_dims):
                volume = box_dims[0] * box_dims[1] * box_dims[2]
                # Calculate molecular weight from structure composition
                density = structure.density
                volume_cm3 = volume / 1e24  # Convert Angstrom^3 to cm^3
                properties["density"] = density if volume_cm3 > 0 else None
            else:
                properties["density"] = None

            # Include target density if it was specified
            if self.density is not None:
                properties["target_density"] = self.density  # g/cm^3
        elif self.packing_mode == "fixed" and self.fixed_positions is not None:
            properties["fixed_positions"] = self.fixed_positions
            properties["num_molecules"] = len(self.fixed_positions)

        return Properties(**properties)
