"""Polymer Generation Nodes."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import BaseModel, ConfigDict, Field, create_model
from pymatgen.core.structure import (
    Structure,
)

from jfchemistry.base_classes import Polymer, SystemProperty
from jfchemistry.base_jobs import Output, Properties, jfchem_job

from .chain_generator import chain_generator


class PolymerInfiniteChainOutput(Output):
    """Polymer Infinite Chain Output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Structure
    files: Optional[Any] = None
    properties: Optional[Any] = None


class PolymerInfiniteChainSystemProperties(BaseModel):
    """Polymer Infinite Chain System Properties."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    total_energy: SystemProperty


class PolymerInfiniteChainProperties(Properties):
    """Polymer Infinite Chain Properties."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    system: PolymerInfiniteChainSystemProperties


@dataclass
class PolymerInfiniteChain(Maker):
    """Construct an Infinite Polymer Chain Model.

    Portions of the workflow are adapted from the PSP library: https://github.com/Ramprasad-Group/PSP
    """

    name: str = "Infinite Polymer Chain"
    num_conformers: Optional[int] = field(
        default=100,
        metadata={"description": "Number of conformers to generate for the monomer."},
    )
    dihedral_angle_cutoff: Optional[float] = field(
        default=8.0,
        metadata={"description": "Cutoff for the dihedral angle in degrees."},
    )
    chain_length: Optional[int] = field(
        default=1, metadata={"description": "Length of the polymer chain."}
    )
    inter_chain_distance: Optional[float] = field(
        default=12.0,
        metadata={"description": "Distance between the chains in Angstroms."},
    )
    rotation_angle: float = field(
        default=180.0,
        metadata={
            "description": "Dihedral angle of the polymer chain\
            in degrees measured across the connection points."
        },
    )
    _dimerized: bool = False
    _output_model: type[PolymerInfiniteChainOutput] = PolymerInfiniteChainOutput
    _properties_model: type[PolymerInfiniteChainProperties] = PolymerInfiniteChainProperties

    def make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()  # type: ignore
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    properties_model
                    | list[type[properties_model]]
                    | OutputReference
                    | list[OutputReference]
                )  # type: ignore

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
        self.make_output_model(
            self._properties_model,
        )

    @jfchem_job()
    def make(self, polymer: Polymer) -> Response[type[PolymerInfiniteChainOutput]]:
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
        structure = chain_generator(
            polymer.monomer,
            self.chain_length,
            self.rotation_angle,
            self.num_conformers,
            self.dihedral_angle_cutoff,
            self.inter_chain_distance,
        )
        file = structure.to_file(fmt="cif")
        structure.to("final.cif")
        return Response(output=self._output_model(structure=structure, files={"chain.cif": file}))
