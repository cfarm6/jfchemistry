# Contributor guide: styling and makers

This document defines **enforced styling** and **requirements for adding or changing makers**. For general contribution workflow (forking, PRs, tests), see [docs/contributing.md](docs/contributing.md).

---

## Styling (enforced)

### Pre-commit

Pre-commit runs on commit and push. Fix issues before pushing.

```bash
# Install hooks (once)
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

### Commands

| Task        | Command           | Purpose                          |
|------------|-------------------|----------------------------------|
| Format     | `pixi run fmt`    | Ruff format (line length 100)    |
| Lint       | `pixi run lint`   | Ruff check (with auto-fix)       |
| Type check | `pixi run types`  | `ty` static type checking       |
| Tests      | `pixi run test`   | Pytest (including doctests)      |

Require the **development** environment (e.g. `pixi install -e development` or `-e dev`).

### Code style rules

- **Line length**: 100 characters (Ruff).
- **Docstrings**: Google style; convention set in `[tool.ruff.lint.pydocstyle]` in `pyproject.toml`.
- **Imports**: Sorted and tidied (Ruff: isort, TID).
- **Type hints**: Required for public APIs; checked with `ty`.
- **Doctests**: Must pass; run with `pixi run test`.

Before opening a PR, run:

```bash
pixi run fmt
pixi run lint
pixi run types
pixi run test
```

(or use pre-commit so these run automatically).

---

## Adding or modifying a maker

Makers are jobflow `Maker` subclasses that define reusable, serializable jobs. All makers in this project must follow the conventions below so they behave consistently and document and serialize correctly.

### 1. Class shape

- **Dataclass**: Use `@dataclass` (and `field()` for any attribute that needs defaults or metadata).
- **Inheritance**: Inherit from the appropriate base (e.g. `CoreMaker`, `JFChemMaker`/`PymatGenMaker`, or a domain base like `Filter`, `GeometryOptimization`) and use the generic parameters for input/output types where applicable.
- **Name**: Set a clear `name: str` default (e.g. `name: str = "My Maker"`).

### 2. Field metadata (required for public fields)

Every **public** constructor argument that is a `field()` must have **metadata** with at least:

- **`"description"`**: Short, user-facing description of the parameter (used for docs/serialization).
- **`"unit"`**: For any parameter that has a physical unit, add `"unit": "<unit_string>"` (e.g. `"kcal/mol"`, `"Å"`, `"K"`, `"fs"`, `"eV/Å"`, `"degrees"`, `"atm"`, `"fs^-1"`, `"g/cm³"`, `"°"`). Use the same unit string as in the docstring (bracket notation). For dimensionless quantities use `"dimensionless"`.

Internal/private fields (e.g. `_output_model`, `_properties_model`, `_input_dict`, `_commands`) should still have a `"description"` if they use `field()`.

Example:

```python
from dataclasses import dataclass, field

@dataclass
class MyMaker(PymatGenMaker[InputType, OutputType]):
    threshold: float | Quantity = field(
        default=0.0,
        metadata={
            "description": "Energy threshold [kcal/mol]. Accepts float or pint Quantity.",
            "unit": "kcal/mol",
        },
    )
```

### 3. Unit-bearing parameters

- **Types**: Use `float | Quantity` (or `Optional[float | Quantity]`) for parameters that have units.
- **Normalization**: In `__post_init__`, convert any `Quantity` to a float magnitude in the canonical unit using `to_magnitude(value, "<pint_unit>")` from `jfchemistry.core.unit_utils`. Use pint-style unit strings (e.g. `"kcal_per_mol"`, `"angstrom"`, `"kelvin"`, `"eV/angstrom"`).
- **Immutability**: Use `object.__setattr__(self, "attr_name", value)` when setting attributes on the dataclass instance inside `__post_init__` (avoids frozen dataclass issues).
- **Order**: Call `super().__post_init__()` after your own normalization so the base can build output/properties models from the normalized values.

Example:

```python
from jfchemistry.core.unit_utils import to_magnitude
from pint import Quantity

def __post_init__(self):
    if isinstance(self.threshold, Quantity):
        object.__setattr__(
            self, "threshold", to_magnitude(self.threshold, "kcal_per_mol")
        )
    super().__post_init__()
```

### 4. Docstring

- **Summary**: One-line summary of what the maker does.
- **Units section**: If the maker has any unit-bearing parameters, add a **Units** section as a bullet list. First line: “Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg`` or ``jfchemistry.Q_``):”. Then one line per parameter: “- param_name: [unit]”.
- **Attributes**: List public attributes with their default and a short description (including the default unit where relevant).
- **Examples**: Optional but encouraged; use `# doctest: +SKIP` for slow or environment-dependent examples.

Example:

```python
"""One-line summary.

Units:
    Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
    or ``jfchemistry.Q_``):

    - threshold: [kcal/mol]

Attributes:
    name: The name of the maker (default: "My Maker").
    threshold: The energy threshold [kcal/mol] (default: 0.0). Accepts float
        or pint Quantity; stored as magnitude in [kcal/mol].
"""
```

Use **square brackets** for units in docstrings (e.g. `[kcal/mol]`, `[eV/Å]`, `[K]`, `[fs]`, `[Å]`).

### 5. Output and properties models

- Set **`_output_model`**: Type of the job output (e.g. a Pydantic model or `Output` subclass).
- Set **`_properties_model`**: Type of the properties model used in the job (e.g. `Properties` or a domain-specific subclass).
- **Ensemble makers**: If the maker operates on a list of inputs (one job per item), set `_ensemble: bool = True` (or ensure the base does). Single-input makers use `_ensemble: bool = False`.

### 6. Job implementation

- **Decorator**: Use `@jfchem_job()` on the `make` method so the job uses `_output_model` and the standard files/properties schema.
- **Logic**: Implement the core work in `_operation(self, input, **kwargs)`. It should return the appropriate output type and optional properties, and only rely on normalized (magnitude) values for unit-bearing attributes.

### Checklist for a new or updated maker

- [ ] Class is a `@dataclass` with correct base class and generics.
- [ ] Every public `field()` has `metadata` with `"description"`.
- [ ] Every unit-bearing parameter has `metadata["unit"]` and is normalized in `__post_init__` via `to_magnitude`.
- [ ] Docstring has a **Units** section (bullet list) if there are any unit-bearing parameters.
- [ ] Docstring uses bracket notation for units (e.g. `[kcal/mol]`).
- [ ] `_output_model` and `_properties_model` are set; `_ensemble` is correct.
- [ ] `make` is decorated with `@jfchem_job()` and `_operation` implements the job logic.
- [ ] `pixi run fmt`, `pixi run lint`, `pixi run types`, and `pixi run test` all pass.

---

## References

- **Unit normalization**: `jfchemistry.core.unit_utils.to_magnitude`
- **Job decorator**: `jfchemistry.core.jfchem_job.jfchem_job`
- **Base makers**: `jfchemistry.core.makers` (`CoreMaker`, `JFChemMaker`/`PymatGenMaker`)
- **Ruff**: `pyproject.toml` → `[tool.ruff]`, `[tool.ruff.lint]`
- **Pre-commit**: `.pre-commit-config.yaml`
