"""Polymer Generation Nodes."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Union

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import BaseModel, ConfigDict, Field, create_model
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import Polymer

from .chain_generator import finite_chain_generator, infinite_chain_generator


class PolymerInfiniteChainOutput(Output):
    """Polymer Infinite Chain Output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Structure
    files: Optional[Any] = None
    properties: Optional[Any] = None


class PolymerFiniteChainOutput(Output):
    """Polymer Infinite Chain Output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Molecule
    files: Optional[Any] = None
    properties: Optional[Any] = None


@dataclass
class GenerateInfinitePolymerChain(Maker):
    """Construct an Infinite Polymer Chain Model.

    Portions of the workflow are adapted from the PSP library: https://github.com/Ramprasad-Group/PSP
    """

    name: str = "Infinite Polymer Chain"
    num_conformers: int = field(
        default=100,
        metadata={"description": "Number of conformers to generate for the monomer."},
    )
    chain_length: int = field(default=2, metadata={"description": "Length of the polymer chain."})
    interchain_distance: float = field(
        default=12.0,
        metadata={"description": "Distance between the chains in Angstroms."},
    )
    rotation_angles: list[float] | float = field(
        default=180.0,
        metadata={
            "description": "Dihedral angle of the polymer chain\
            in degrees measured across the connection points."
        },
    )
    _dimerized: bool = False
    _output_model: type[PolymerInfiniteChainOutput] = PolymerInfiniteChainOutput
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                # Construct annotation dynamically to avoid type checker errors with variable types
                annotation = Union[
                    properties_model,  # type: ignore[valid-type]
                    list[properties_model],  # type: ignore[valid-type]
                    OutputReference,
                    list[OutputReference],
                ]

            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
                    *f_dict["metadata"],  # type: ignore
                    Field(**f_dict["attributes"]),
                ],  # type: ignore
                None,
            )

        self._output_model = create_model(
            f"{self._output_model.__name__}",
            __base__=self._output_model,
            **fields,
        )

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        # Convert single float to list if needed
        if isinstance(self.rotation_angles, (int, float)):
            if self.chain_length is None:
                raise ValueError("chain_length must be set when rotation_angles is a single value")
            self.rotation_angles = [float(self.rotation_angles)] * self.chain_length

        # Validate rotation_angles length matches chain_length
        if self.chain_length is not None and len(self.rotation_angles) != self.chain_length:
            raise ValueError(
                f"rotation_angles length ({len(self.rotation_angles)}) must equal "
                f"chain_length ({self.chain_length})"
            )

        self.make_output_model(
            self._properties_model,
        )

    @jfchem_job()
    def make(self, polymer: Polymer) -> Response[_output_model]:
        """Make a polymer infinite chain.

        Steps:
        1. Fetch the dummy atom index and bond type from the monomer smiles string
        1.1 Create a dimer if the connection points are connected to the same atom
        2. Replace the dummy atom with the appropriate atom type
        3. Convert the smiles string back to an RDKit molecule
        4. Add hydrogens to the monomer
        5. Embed the monomer
        6. Optimize an Ensemble of Conformers with UFF
        7. Use the lowest energy conformer to generate the polymer structure.
        9. Orient the monomer to have connection-point-1 and connection-point-2 on the z-axis
        10. Translate the monomer to have connection-point-1 at the origin
        11. Rotate the monomer about the z-axis by the rotation angle
        12. Remove the connection-point-2 and any extra atoms
        13. Calculate the bounding box of the monomer and set the periodic boundary conditions\
             accordingly.
        14. Build a polymer chain by repeating the monomer\
        chain_length times and rotating it by the rotation angle each time.
        15. Return the polymer structure.
        """
        if isinstance(self.rotation_angles, (int, float)):
            self.rotation_angles = [float(self.rotation_angles)] * self.chain_length

        structure = infinite_chain_generator(
            polymer.monomer,
            self.chain_length,
            self.rotation_angles,
            self.num_conformers,
            self.interchain_distance,
        )

        file = structure.to_file(fmt="cif")

        return Response(output=self._output_model(structure=structure, files={"chain.cif": file}))


@dataclass
class GenerateFinitePolymerChain(Maker):
    """Construct a Finite Polymer Chain Model.

    Portions of the workflow are adapted from the PSP library: https://github.com/Ramprasad-Group/PSP
    """

    name: str = "GenerateFinite Polymer Chain"
    num_conformers: int = field(
        default=100,
        metadata={"description": "Number of conformers to generate for the monomer."},
    )
    chain_length: int = field(default=2, metadata={"description": "Length of the polymer chain."})
    rotation_angles: list[float] | float = field(
        default=180.0,
        metadata={
            "description": "Dihedral angle of the polymer chain\
            in degrees measured across the connection points."
        },
    )
    head_angle: float = field(
        default=180.0, metadata={"description": "Dihedral angle of the head group"}
    )
    tail_angle: float = field(
        default=180.0, metadata={"description": "Dihedral angle of the tail group"}
    )
    _dimerized: bool = False
    _output_model: type[PolymerFiniteChainOutput] = PolymerFiniteChainOutput
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                # Construct annotation dynamically to avoid type checker errors with variable types
                annotation = Union[
                    properties_model,  # type: ignore[valid-type]
                    list[properties_model],  # type: ignore[valid-type]
                    OutputReference,
                    list[OutputReference],
                ]

            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
                    *f_dict["metadata"],  # type: ignore
                    Field(**f_dict["attributes"]),
                ],  # type: ignore
                None,
            )

        self._output_model = create_model(
            f"{self._output_model.__name__}",
            __base__=self._output_model,
            **fields,
        )

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        # Convert single float to list if needed
        if isinstance(self.rotation_angles, float):
            self.rotation_angles = [float(self.rotation_angles)] * self.chain_length

        # Validate rotation_angles length matches chain_length
        if isinstance(self.rotation_angles, list):
            if len(self.rotation_angles) != self.chain_length:
                raise ValueError(
                    f"rotation_angles length ({len(self.rotation_angles)}) must equal "
                    f"chain_length ({self.chain_length})"
                )

        self.make_output_model(
            self._properties_model,
        )

    @jfchem_job()
    def make(self, polymer: Polymer) -> Response[_output_model]:
        """Make a polymer finite chain.

        Steps:
        1. Build the chain with the infinite chain builder
        2. Add end caps and convert to a molecular structure
        """
        if isinstance(self.rotation_angles, (int, float)):
            self.rotation_angles = [float(self.rotation_angles)] * self.chain_length

        chain = finite_chain_generator(
            polymer,
            self.chain_length,
            self.rotation_angles,
            self.head_angle,
            self.tail_angle,
            self.num_conformers,
        )

        file = chain.to_file(fmt="cif")

        return Response(output=self._output_model(structure=chain, files={"chain.xyz": file}))
