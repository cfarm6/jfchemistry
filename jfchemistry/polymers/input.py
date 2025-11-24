"""Polymer Input Ndoes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from pydantic import BaseModel, ConfigDict, Field, create_model
from rdkit.Chem import rdmolfiles, rdmolops

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties
from jfchemistry.core.structures import Polymer, RDMolMolecule

if TYPE_CHECKING:
    from pydantic.fields import _FieldInfoAsDict


class PolymerInputOutput(Output):
    """Polymer Input Output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Polymer
    files: None = None
    properties: None = None


@dataclass
class PolymerInput(Maker):
    """Polymer Input."""

    name: str = "Polymer Input"
    _output_model: type[PolymerInputOutput] = PolymerInputOutput
    _properties_model: type[Properties] = Properties

    def make_output_model(self, properties_model: type[BaseModel]):
        """Make a properties model for the job."""
        fields = {}
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict: "_FieldInfoAsDict" = f_info.asdict()
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = properties_model | list[properties_model]  # type: ignore

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
        self.make_output_model(self._properties_model)

    @jfchem_job()
    def make(
        self, monomer: str, head: str | None = None, tail: str | None = None
    ) -> Response[_output_model]:
        """Make a polymer."""
        monomer_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer))
        head_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(head)) if head else None
        tail_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(tail)) if tail else None

        polymer = Polymer(
            head=RDMolMolecule(head_mol) if head_mol else None,
            monomer=RDMolMolecule(monomer_mol),
            tail=RDMolMolecule(tail_mol) if tail_mol else None,
        )
        print(polymer)
        return Response(output=self._output_model(structure=polymer))
