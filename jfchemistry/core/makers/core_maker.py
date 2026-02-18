"""Core Maker."""

import importlib
from dataclasses import dataclass, field
from typing import Annotated, Type

from jobflow.core.maker import Maker
from jobflow.core.reference import OutputReference
from pydantic import Field, create_model

from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


@dataclass
class CoreMaker[InputType, OutputType](Maker):
    """Base class for all makers."""

    name: str = "Single Structure Calculator Maker"
    _output_model: Type[Output] = Output
    _properties_model: Type[Properties] = Properties
    _ensemble: bool = field(
        default=False,
        metadata={
            "description": "Whether the maker expects a list of inputs (ensemble) \
                or a single input."
        },
    )

    def __post_init__(self):
        """Make a properties model for the job."""
        fields = {}
        if isinstance(self._output_model, dict):
            module = self._output_model["@module"]
            class_name = self._output_model["@callable"]
            self._output_model = getattr(importlib.import_module(module), class_name)
        for f_name, f_info in self._output_model.model_fields.items():
            f_dict = f_info.asdict()
            annotation = f_dict["annotation"]
            if f_name == "properties":
                annotation = (
                    self._properties_model
                    | list[self._properties_model]  # type: ignore
                    | OutputReference
                    | list[OutputReference]
                )
            elif f_name == "structure":
                annotation = OutputType | list[OutputType] | OutputReference | list[OutputReference]
            fields[f_name] = (
                Annotated[
                    annotation | None,  # type: ignore
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
        self._properties_model = self._properties_model
