"""Write properties objects to JSON files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobflow.core.job import Response

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties


@dataclass
class PropertiesToDisk(PymatGenMaker):
    """Persist one or more properties objects to JSON on disk.

    If a list is provided, indexed suffixes are appended to the base filename:
    e.g. ``properties.json`` -> ``properties_0.json``, ``properties_1.json``.
    """

    name: str = "Properties To Disk"
    output_dir: str = "."
    filename: str = "properties.json"
    _output_model: type[Output] = Output

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        return filename if filename.lower().endswith(".json") else f"{filename}.json"

    @jfchem_job()
    def make(self, properties: Properties | list[Properties]) -> Response[_output_model]:
        """Write properties object(s) to JSON file(s)."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = Path(self._normalize_filename(self.filename))
        written: list[str] = []

        if isinstance(properties, list):
            stem = base.stem
            suffix = base.suffix
            for idx, prop in enumerate(properties):
                p = Properties.model_validate(prop, extra="allow", strict=False)
                out_path = out_dir / f"{stem}_{idx}{suffix}"
                out_path.write_text(p.model_dump_json(indent=2))
                written.append(str(out_path))
        else:
            p = Properties.model_validate(properties, extra="allow", strict=False)
            out_path = out_dir / base
            out_path.write_text(p.model_dump_json(indent=2))
            written.append(str(out_path))

        return Response(output=Output(files={"properties_json": written}))
