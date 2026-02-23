"""Finite Polymer Chain Generator."""

from dataclasses import dataclass, field
from typing import Annotated, Optional, Union

from jobflow.core.job import OutputReference, Response
from jobflow.core.maker import Maker
from pint import Quantity
from pydantic import BaseModel, Field, create_model

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import PolymerFiniteChainOutput
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import Polymer
from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.polymers.generator import make_finite_chain


@dataclass
class GenerateFinitePolymerChain(Maker):
    """Construct a Finite Polymer Chain Model.

    Portions of the workflow are adapted from the PSP library: https://github.com/Ramprasad-Group/PSP

    Units:
        Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
        or ``jfchemistry.Q_``). For dihedral_angles, a list may mix floats and
        Quantities:

        - dihedral_angles: [degrees]
        - dihedral_cutoff: [degrees]
        - monomer_dihedral: [degrees]

    Args:
        num_conformers: Number of conformers to generate for the monomer.
        chain_length: Length of the polymer chain.
        dihedral_angles: Dihedral angle(s) of the polymer chain [degrees]. Accepts float, pint\
             Quantity, or list. \
        Measured across the connection points.
        dihedral_cutoff: Dihedral angle cutoff for the polymer chain [degrees]. Accepts float in \
            [degrees] or pint Quantity.
        monomer_dihedral: Dihedral angle of the monomer [degrees]. Accepts float in [degrees] or \
            pint Quantity.

    Returns:
        A Finite Polymer Chain Model.
    """

    name: str = "GenerateFinite Polymer Chain"
    num_conformers: int = field(
        default=100,
        metadata={"description": "Number of conformers to generate for the monomer."},
    )
    chain_length: Optional[int] = field(
        default=None, metadata={"description": "Length of the polymer chain."}
    )
    dihedral_angles: Optional[list[float | Quantity] | float | Quantity] = field(
        default=None,
        metadata={
            "description": "Dihedral angle(s) of the polymer chain [degrees]. Accepts float in \
                [degrees], pint Quantity, or list.",
            "unit": "degrees",
        },
    )
    dihedral_cutoff: float | Quantity = field(
        default=10,
        metadata={
            "description": "Dihedral angle cutoff for the polymer chain [degrees]. Accepts float in\
                 [degrees] or pint Quantity.",
            "unit": "degrees",
        },
    )
    monomer_dihedral: float | Quantity = field(
        default=180.0,
        metadata={
            "description": "Dihedral angle of the monomer [degrees]. Accepts float in [degrees] or\
                 pint Quantity.",
            "unit": "degrees",
        },
    )
    _output_model: type[PolymerFiniteChainOutput] = PolymerFiniteChainOutput
    _properties_model: type[Properties] = Properties

    def _make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()
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
                    annotation | None,
                    *f_dict["metadata"],
                    Field(**f_dict["attributes"]),
                ],
                None,
            )

        self._output_model = create_model(
            f"{self._output_model.__name__}",
            __base__=self._output_model,
            **fields,
        )

    def __post_init__(self):
        """Post-initialization hook to make the output model."""
        # Normalize unit-bearing attributes
        if isinstance(self.dihedral_cutoff, Quantity):
            object.__setattr__(
                self, "dihedral_cutoff", to_magnitude(self.dihedral_cutoff, "degree")
            )
        if isinstance(self.monomer_dihedral, Quantity):
            object.__setattr__(
                self, "monomer_dihedral", to_magnitude(self.monomer_dihedral, "degree")
            )
        if self.dihedral_angles is not None:
            if isinstance(self.dihedral_angles, (list, tuple)):
                object.__setattr__(
                    self,
                    "dihedral_angles",
                    [
                        to_magnitude(x, "degree") if isinstance(x, Quantity) else float(x)
                        for x in self.dihedral_angles
                    ],
                )
            elif isinstance(self.dihedral_angles, Quantity):
                object.__setattr__(
                    self, "dihedral_angles", to_magnitude(self.dihedral_angles, "degree")
                )
        # Convert single float to list if needed
        if isinstance(self.dihedral_angles, float) and isinstance(self.chain_length, int):
            assert self.chain_length > 0, "Chain length must be greater than 0"
            self.dihedral_angles = [self.dihedral_angles] * self.chain_length

        # Validate dihedral_angles length matches chain_length
        elif isinstance(self.dihedral_angles, list):
            assert self.chain_length is None, (
                "Either chain_length or dihedral_angles must be provided"
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
        chain = make_finite_chain(
            polymer,
            dihedrals=self.dihedral_angles,  # type: ignore
            dihedral_cutoff=to_magnitude(self.dihedral_cutoff, "degree"),
            number_conformers=self.num_conformers,
            monomer_dihedral=to_magnitude(self.monomer_dihedral, "degree"),
        )
        file = chain.to(fmt="xyz")
        return Response(output=self._output_model(structure=chain, files={"chain.xyz": file}))
