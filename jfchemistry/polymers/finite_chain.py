"""Finite Polymer Chain Generator."""

from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, Union

from jobflow.core.job import OutputReference, Response
from jobflow.core.maker import Maker
from pint import Quantity
from pydantic import BaseModel, Field, create_model

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import PolymerFiniteChainOutput
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import Polymer
from jfchemistry.core.unit_utils import to_magnitude
from jfchemistry.polymers.generator import (
    generate_alternating_sequence,
    generate_block_sequence,
    generate_periodic_sequence,
    generate_weighted_random_sequence,
    make_finite_chain,
    make_finite_copolymer_chain,
)


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


@dataclass
class GenerateFiniteCopolymerChain(Maker):
    """Construct a finite co-polymer chain from a sequence of polymer blocks.

    This node is analogous to ``GenerateFinitePolymerChain`` but supports
    co-polymers by selecting monomer units from multiple polymer templates
    according to an explicit sequence.
    """

    name: str = "GenerateFinite Copolymer Chain"
    num_conformers: int = field(
        default=100,
        metadata={"description": "Number of conformers to generate for each monomer type."},
    )
    sequence_mode: Literal["explicit", "weighted_random", "alternating", "periodic", "block"] = (
        field(
            default="explicit",
            metadata={"description": "Mode used to define repeating-unit sequence."},
        )
    )
    sequence: list[int] = field(
        default_factory=list,
        metadata={"description": "Ordered list of polymer indices (0-based) defining the chain."},
    )
    chain_length: Optional[int] = field(
        default=None,
        metadata={"description": "Target chain length for generated sequence modes."},
    )
    unit_weights: Optional[list[float]] = field(
        default=None,
        metadata={"description": "Relative weights for each monomer type in random sequence mode."},
    )
    random_seed: Optional[int] = field(
        default=None,
        metadata={"description": "Optional random seed for reproducible sequence generation."},
    )
    alternating_units: Optional[list[int]] = field(
        default=None,
        metadata={"description": "Unit IDs used for alternating mode."},
    )
    periodic_motif: Optional[list[int]] = field(
        default=None,
        metadata={"description": "Periodic motif used for periodic mode."},
    )
    block_units: Optional[list[int]] = field(
        default=None,
        metadata={"description": "Unit IDs for each block in block mode."},
    )
    block_lengths: Optional[list[int]] = field(
        default=None,
        metadata={"description": "Block lengths for block mode."},
    )
    dihedral_angles: Optional[list[float | Quantity] | float | Quantity] = field(
        default=None,
        metadata={
            "description": "Inter-monomer dihedral angle(s) [degrees]. "
            "Accepts float, pint Quantity, or list.",
            "unit": "degrees",
        },
    )
    dihedral_cutoff: float | Quantity = field(
        default=10,
        metadata={"description": "Conformer dihedral cutoff [degrees].", "unit": "degrees"},
    )
    monomer_dihedral: float | Quantity = field(
        default=180.0,
        metadata={"description": "Monomer internal dihedral [degrees].", "unit": "degrees"},
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
        """Normalize units and validate settings."""
        if isinstance(self.dihedral_cutoff, Quantity):
            object.__setattr__(
                self, "dihedral_cutoff", to_magnitude(self.dihedral_cutoff, "degree")
            )
        if isinstance(self.monomer_dihedral, Quantity):
            object.__setattr__(
                self, "monomer_dihedral", to_magnitude(self.monomer_dihedral, "degree")
            )

        if self.dihedral_angles is None:
            object.__setattr__(self, "dihedral_angles", [180.0] * max(len(self.sequence) - 1, 0))
        elif isinstance(self.dihedral_angles, (list, tuple)):
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
                self,
                "dihedral_angles",
                to_magnitude(self.dihedral_angles, "degree"),
            )

        if isinstance(self.dihedral_angles, float):
            object.__setattr__(
                self,
                "dihedral_angles",
                [self.dihedral_angles] * max(len(self.sequence) - 1, 0),
            )

        mode = self.sequence_mode

        if mode == "explicit":
            if len(self.sequence) == 0:
                raise ValueError("explicit mode requires sequence")
            if any(
                x is not None
                for x in [
                    self.unit_weights,
                    self.chain_length,
                    self.alternating_units,
                    self.periodic_motif,
                    self.block_units,
                    self.block_lengths,
                ]
            ):
                raise ValueError("explicit mode is mutually exclusive with generator options")

        elif mode == "weighted_random":
            if len(self.sequence) > 0:
                raise ValueError("sequence and weighted-random options are mutually exclusive")
            if self.chain_length is None or self.unit_weights is None:
                raise ValueError("weighted_random mode requires both chain_length and unit_weights")
            object.__setattr__(
                self,
                "sequence",
                generate_weighted_random_sequence(
                    chain_length=self.chain_length,
                    unit_weights=self.unit_weights,
                    seed=self.random_seed,
                ),
            )

        elif mode == "alternating":
            if len(self.sequence) > 0:
                raise ValueError("sequence and alternating options are mutually exclusive")
            if self.chain_length is None or self.alternating_units is None:
                raise ValueError("alternating mode requires chain_length and alternating_units")
            object.__setattr__(
                self,
                "sequence",
                generate_alternating_sequence(self.chain_length, self.alternating_units),
            )

        elif mode == "periodic":
            if len(self.sequence) > 0:
                raise ValueError("sequence and periodic options are mutually exclusive")
            if self.chain_length is None or self.periodic_motif is None:
                raise ValueError("periodic mode requires chain_length and periodic_motif")
            object.__setattr__(
                self,
                "sequence",
                generate_periodic_sequence(self.chain_length, self.periodic_motif),
            )

        elif mode == "block":
            if len(self.sequence) > 0:
                raise ValueError("sequence and block options are mutually exclusive")
            if self.block_units is None or self.block_lengths is None:
                raise ValueError("block mode requires block_units and block_lengths")
            object.__setattr__(
                self,
                "sequence",
                generate_block_sequence(
                    block_units=self.block_units,
                    block_lengths=self.block_lengths,
                    chain_length=self.chain_length,
                ),
            )

        if isinstance(self.dihedral_angles, list) and len(self.dihedral_angles) == 0:
            object.__setattr__(
                self,
                "dihedral_angles",
                [180.0] * max(len(self.sequence) - 1, 0),
            )

        assert len(self.sequence) > 0, "sequence must contain at least one polymer index"
        assert isinstance(self.dihedral_angles, list)
        assert len(self.dihedral_angles) == len(self.sequence) - 1, (
            "dihedral_angles length must equal len(sequence)-1"
        )

        self._make_output_model(self._properties_model)

    @jfchem_job()
    def make(self, polymers: list[Polymer]) -> Response[_output_model]:
        """Generate a finite co-polymer chain from polymer templates and sequence."""
        if max(self.sequence) >= len(polymers):
            raise ValueError("sequence includes polymer index out of range")

        chain = make_finite_copolymer_chain(
            polymers=polymers,
            sequence=self.sequence,
            dihedrals=self.dihedral_angles,  # type: ignore[arg-type]
            dihedral_cutoff=to_magnitude(self.dihedral_cutoff, "degree"),
            number_conformers=self.num_conformers,
            monomer_dihedral=to_magnitude(self.monomer_dihedral, "degree"),
        )
        file = chain.to(fmt="xyz")
        return Response(output=self._output_model(structure=chain, files={"chain.xyz": file}))
