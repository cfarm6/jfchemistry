"""Finite Polymer Chain Generator."""

from dataclasses import dataclass, field
from typing import Annotated, Union

from jobflow.core.job import OutputReference, Response
from jobflow.core.maker import Maker
from pydantic import BaseModel, Field, create_model

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import PolymerFiniteChainOutput
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import Polymer
from jfchemistry.polymers.generator import make_finite_chain


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
    dihedral_cutoff: float = field(
        default=10, metadata={"description": "Dihedral angle cutoff for the polymer chain."}
    )
    monomer_dihedral: float = field(
        default=180.0, metadata={"description": "Dihedral angle of the monomer."}
    )
    _output_model: type[PolymerFiniteChainOutput] = PolymerFiniteChainOutput
    _properties_model: type[Properties] = Properties

    def _make_output_model(self, properties_model: type[BaseModel]):
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

        self._make_output_model(
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

        chain = make_finite_chain(
            polymer,
            self.chain_length,
            self.monomer_dihedral,
            self.dihedral_cutoff,
            self.num_conformers,
            self.rotation_angles,
        )
        file = chain.to(fmt="xyz")
        chain.to("generated_chain.xyz")
        return Response(output=self._output_model(structure=chain, files={"chain.xyz": file}))
