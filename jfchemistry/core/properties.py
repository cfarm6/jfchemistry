"""Properties of the structure."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator

type NestedFloatList = list[float] | list["NestedFloatList"] | float


class Property(BaseModel):
    """A calculated property."""

    name: str
    value: NestedFloatList
    units: str
    uncertainty: Optional[NestedFloatList] = None
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create a Property from a dictionary."""
        return cls.model_validate(d, extra="ignore", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Property to a dictionary."""
        return self.model_dump(mode="json")


class AtomicProperty(Property):
    """An atomic property."""

    value: NestedFloatList


class BondProperty(Property):
    """A bond property."""

    value: NestedFloatList
    atoms1: list[int]
    atoms2: list[int]


class OrbitalProperty(Property):
    """An orbital property."""

    value: NestedFloatList


class SystemProperty(Property):
    """A system property."""

    value: NestedFloatList


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
        """Convert the Property to a dictionary."""
        return self.model_dump(mode="json")
