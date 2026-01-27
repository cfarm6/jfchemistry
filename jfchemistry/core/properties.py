"""Properties of the structure."""

import base64
import pickle
from typing import Any, Optional

from pint import Quantity
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

type NestedList[T] = T | list["NestedList"]
type NestedQuantityList = Quantity | float | list["NestedQuantityList"]


class PickledQuantity:
    """Wrapper class for pickled Quantity objects to enable MontyDecoder deserialization.

    This class provides a from_dict method that MontyDecoder can use to deserialize
    pickled Quantity objects that were stored as base64-encoded pickle data.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Quantity:
        """Reconstruct a Quantity from a pickled dictionary representation.

        Args:
            d: Dictionary containing pickled Quantity data with 'data' key containing
                base64-encoded pickle data.

        Returns:
            Quantity object reconstructed from the pickled data.
        """
        data = d["data"]

        if isinstance(data, str):
            try:
                decoded = base64.b64decode(data.encode("utf-8"))
            except (ValueError, Exception) as exc:
                raise ValueError("Invalid base64 data for PickledQuantity") from exc
        else:
            decoded = data

        quantity = pickle.loads(decoded)
        return quantity


def _quantity_to_dict(qty: Any) -> Any:
    """Convert a Pint Quantity to a serializable dictionary using pickle.

    Quantity objects are pickled and base64 encoded since Pint's parser cannot
    reliably handle array/list magnitudes when parsing from strings.
    """
    if isinstance(qty, Quantity):
        # Pickle the Quantity object and encode as base64 for reliable serialization
        # This handles both scalar and array/list magnitudes correctly
        data = pickle.dumps(qty)
        encoded = base64.b64encode(data).decode("utf-8")
        return {
            "data": encoded,
            "@class": "PickledQuantity",
            "@module": "jfchemistry.core.properties",
        }
    elif isinstance(qty, (list, tuple)):
        return [_quantity_to_dict(item) for item in qty]
    else:
        # Handle numpy arrays and other array-like objects
        try:
            import numpy as np

            if isinstance(qty, np.ndarray):
                return _quantity_to_dict(qty.tolist())  # type: ignore[attr-defined]
        except ImportError:
            pass
        return qty


class Property(BaseModel):
    """A calculated property."""

    name: str
    value: NestedQuantityList
    description: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("value")
    def serialize_quantity_fields(self, value: Any, _info: Any) -> Any:
        """Serialize Quantity fields to dictionaries."""
        if value is None:
            return None
        return _quantity_to_dict(value)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create a Property from a dictionary."""
        return cls.model_validate(d, extra="ignore", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Property to a dictionary for MSON serialization."""
        # model_dump() will automatically apply field serializers for Quantity fields
        d = self.model_dump()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d


class AtomicProperty(Property):
    """An atomic property."""

    value: NestedQuantityList


class BondProperty(Property):
    """A bond property."""

    value: NestedQuantityList
    atoms1: list[int]
    atoms2: list[int]


class OrbitalProperty(Property):
    """An orbital property."""

    value: NestedQuantityList


class SystemProperty(Property):
    """A system property."""

    value: NestedQuantityList


class PropertyClass(BaseModel):
    """Class for property classes."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def convert_extra_fields(cls, data: Any) -> Property:
        """Convert extra fields to Property objects."""
        if not isinstance(data, dict):
            return data

        # Get known field names from the model
        known_fields = cls.model_fields.keys()

        # Convert only the extra fields
        for key, value in data.items():
            if key not in known_fields and isinstance(value, dict):
                data[key] = Property(**value)

        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert the PropertyClass to a dictionary for MSON serialization."""
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }
        for key, value in self.model_dump().items():
            if isinstance(value, Property):
                d[key] = value.to_dict()
            else:
                d[key] = _quantity_to_dict(value)
        return d


class Properties(BaseModel):
    """Properties of the structure."""

    atomic: Optional[PropertyClass] = None
    bond: Optional[PropertyClass] = None
    system: Optional[PropertyClass] = None
    orbital: Optional[PropertyClass] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create a Property from a dictionary."""
        return cls.model_validate(d, extra="ignore", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Properties to a dictionary for MSON serialization."""
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }
        if self.atomic is not None:
            d["atomic"] = self.atomic.to_dict()
        if self.bond is not None:
            d["bond"] = self.bond.to_dict()
        if self.system is not None:
            d["system"] = self.system.to_dict()
        if self.orbital is not None:
            d["orbital"] = self.orbital.to_dict()
        return d
