"""Base class for structure generation.

This module provides the base Maker class for molecular dynamics workflows
in jfchemistry.
"""

from typing import Optional

from pymatgen.core.structure import SiteCollection

from jfchemistry.core.outputs import Output


class MolecularDynamicsOutput(Output):
    """Output for a molecular dynamics simulations."""

    trajectory: Optional[list[SiteCollection] | list[list[SiteCollection]]]


class MolecularDynamics:
    """Base Maker for running molecular dynamics simulations.

    This class serves as the base interface for all molecular dynamics
    implementations in jfchemistry. Subclasses should implement the
    run_simulation and get_properties methods.

    Attributes:
        name: The name of the molecular dynamics job.
    """

    name: str = "Molecular Dynamics"
