"""Base Job Classes for single molecules."""

from typing import Any, Optional

from monty.json import MontyDecoder
from pydantic import BaseModel, ConfigDict


class Output(BaseModel):
    """Output of the job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    structure: Optional[Any] = None
    properties: Optional[Any] = None
    files: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Create an Output from a dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = MontyDecoder().process_decoded(value)
        return cls.model_validate(d, extra="allow", strict=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Output to a dictionary."""
        return self.model_dump(mode="json")
