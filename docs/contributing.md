---
title: Contributing Guide
summary: How to contribute to JFChemistry
---

# Contributing to JFChemistry

Thank you for your interest in contributing to JFChemistry! This guide will help you get started.

## Ways to Contribute

-   🐛 Report bugs and issues
-   💡 Suggest new features or enhancements
-   📝 Improve documentation
-   🧪 Add examples and tutorials
-   🔧 Fix bugs or implement features
-   🧪 Add new calculators or methods

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/jfchemistry.git
cd jfchemistry
```

### 2. Set Up Development Environment

```bash
# Install development environment with pixi
pixi install -e dev

# Or install all optional features
pixi install -e aimnet2 -e orb -e dev -e docs
```

### 3. Create a Branch

```bash
# Create a branch for your changes
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

## Development Workflow

### Code Style

JFChemistry uses [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting:

```bash
# Format code
pixi run -e dev fmt

# Run linter
pixi run -e dev lint

# Type checking
pixi run -e dev types
```

**All code must**:

-   Follow Google-style docstrings
-   Include type hints
-   Pass ruff formatting and linting
-   Have line length ≤ 100 characters

### Example Docstring

```python
def optimize_structure(
    self,
    structure: Molecule,
    fmax: float = 0.01
) -> tuple[Molecule, dict[str, Any]]:
    """Optimize a molecular structure.

    Performs geometry optimization using the configured method
    until forces fall below the threshold.

    Args:
        structure: Pymatgen Molecule to optimize.
        fmax: Maximum force threshold in eV/Å.

    Returns:
        Tuple containing:
            - Optimized Molecule structure
            - Dictionary of calculated properties

    Raises:
        CalculationError: If optimization fails to converge.

    Examples:
        >>> from jfchemistry.optimizers import TBLiteOptimizer
        >>> optimizer = TBLiteOptimizer(method="GFN2-xTB")
        >>> opt_mol, props = optimizer.optimize_structure(molecule)
    """
```

### Testing

Write tests for new features:

```bash
# Run all tests
pixi run -e dev test

# Run specific test file
pixi run -e dev pytest tests/test_your_feature.py

# Run with coverage
pixi run -e dev pytest --cov=jfchemistry
```

Tests should:

-   Cover main functionality
-   Include edge cases
-   Use doctests for simple examples
-   Mock external dependencies when appropriate

### Documentation

Update documentation for changes:

```bash
# Build documentation locally
pixi run -e docs mkdocs serve

# View at http://127.0.0.1:8000
```

Documentation updates should include:

-   Updated docstrings in code
-   Examples in relevant `.md` files
-   Updates to User Guide if needed
-   FAQ entries for common questions

## Adding New Features

### Adding a New Calculator

1. Create a new file in `jfchemistry/calculators/`:

```python
# jfchemistry/calculators/my_calculator.py
from jfchemistry.calculators import ASECalculator

class MyCalculator(ASECalculator):
    """Calculator using My Method.

    Attributes:
        parameter1: Description of parameter1.
        parameter2: Description of parameter2.
    """

    def __init__(self, parameter1: str = "default"):
        """Initialize My Calculator.

        Args:
            parameter1: Description and purpose.
        """
        self.parameter1 = parameter1

    def set_calculator(self, atoms, charge=0, spin_multiplicity=1):
        """Set up the ASE calculator."""
        # Implementation
        pass

    def get_properties(self, atoms):
        """Extract properties from calculation."""
        # Implementation
        pass
```

2. Add to `jfchemistry/calculators/__init__.py`:

```python
from .my_calculator import MyCalculator

__all__ = [..., "MyCalculator"]
```

3. Add tests in `tests/test_calculators.py`

4. Add documentation example

### Adding a New Optimizer

Similar process in `jfchemistry/optimizers/`:

```python
from jfchemistry.optimizers.base import GeometryOptimization

class MyOptimizer(GeometryOptimization):
    """Optimizer using My Method."""

    def optimize_structure(self, structure):
        """Optimize the structure."""
        # Implementation
        pass
```

### Adding Examples

Add examples to `examples/` directory:

```python
# examples/my_example.py
"""
Example: Doing Something Cool

This example demonstrates how to...
"""

from jfchemistry.inputs import Smiles
# ... rest of example

if __name__ == "__main__":
    # Runnable code
    pass
```

## Pull Request Process

### 1. Before Submitting

-   [ ] Code passes all tests
-   [ ] Code is formatted with ruff
-   [ ] Docstrings are complete and accurate
-   [ ] Documentation is updated
-   [ ] CHANGELOG is updated (if applicable)
-   [ ] Commits are clear and descriptive

### 2. Submit Pull Request

1. Push your branch to your fork:

```bash
git push origin feature/your-feature-name
```

2. Open a Pull Request on GitHub

3. Fill out the PR template:

```markdown
## Description

Brief description of changes

## Type of Change

-   [ ] Bug fix
-   [ ] New feature
-   [ ] Documentation update
-   [ ] Performance improvement

## Testing

How was this tested?

## Checklist

-   [ ] Code follows style guidelines
-   [ ] Tests pass
-   [ ] Documentation updated
-   [ ] Self-reviewed code
```

### 3. Code Review

-   Address reviewer feedback
-   Keep discussion focused and professional
-   Be open to suggestions

### 4. Merging

Once approved:

-   Maintainer will merge your PR
-   Your changes will be included in the next release

## Commit Message Guidelines

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add AimNet2 calculator for neural network potentials"
git commit -m "Fix conformer generation memory leak"
git commit -m "Update documentation for TBLite optimizer"

# Bad
git commit -m "fix stuff"
git commit -m "wip"
git commit -m "asdf"
```

Format:

```
<type>: <subject>

[optional body]

[optional footer]
```

Types:

-   `feat`: New feature
-   `fix`: Bug fix
-   `docs`: Documentation changes
-   `style`: Code style (formatting, etc.)
-   `refactor`: Code refactoring
-   `test`: Adding/updating tests
-   `chore`: Maintenance tasks

## Project Structure

```
jfchemistry/
├── jfchemistry/           # Source code
│   ├── calculators/       # Calculator implementations
│   ├── conformers/        # Conformer generation
│   ├── generation/        # 3D structure generation
│   ├── inputs/            # Input parsers
│   ├── modification/      # Structure modification
│   ├── optimizers/        # Geometry optimizers
│   └── jfchemistry.py     # Core classes
├── tests/                 # Test files
├── docs/                  # Documentation source
├── examples/              # Example scripts
└── pyproject.toml         # Project configuration
```

## Best Practices

### 1. Keep Changes Focused

-   One feature/fix per PR
-   Keep PRs reasonably sized
-   Split large changes into multiple PRs

### 2. Write Good Tests

```python
def test_optimizer_convergence():
    """Test that optimizer converges for simple molecule."""
    # Arrange
    molecule = create_test_molecule()
    optimizer = TBLiteOptimizer(fmax=0.01)

    # Act
    job = optimizer.make(molecule)
    result = job.output

    # Assert
    assert result["structure"] is not None
    assert result["properties"]["converged"] is True
```

### 3. Document Public APIs

Every public class, method, and function needs:

-   Concise summary line
-   Detailed description
-   Args documentation
-   Returns documentation
-   Examples (when helpful)

### 4. Handle Errors Gracefully

```python
try:
    result = calculation()
except CalculationError as e:
    logger.error(f"Calculation failed: {e}")
    raise
```

## Questions?

-   Open an issue for questions
-   Join discussions on GitHub
-   Check existing issues and PRs

## Code of Conduct

Be respectful and constructive. We're all here to advance computational chemistry!

Thank you for contributing! 🎉
