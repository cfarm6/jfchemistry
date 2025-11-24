"""Base class for structure generation.

This module provides the base Maker class for molecular dynamics workflows
in jfchemistry.
"""

from typing import Any, Optional

from jobflow import Maker
from pymatgen.core.structure import SiteCollection

from jfchemistry.core.outputs import Output


class MolecularDynamicsOutput(Output):
    """Output for a molecular dynamics simulations."""

    trajectory: Optional[list[SiteCollection] | list[list[SiteCollection]]]


class MolecularDynamics(Maker):
    """Base Maker for running molecular dynamics simulations.

    This class serves as the base interface for all molecular dynamics
    implementations in jfchemistry. Subclasses should implement the
    run_simulation and get_properties methods.

    Attributes:
        name: The name of the molecular dynamics job.
    """

    name: str = "Molecular Dynamics"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Run a molecular dynamics simulation.

        This method should run a molecular dynamics simulation on the given structure.

        Args:
            structure: The molecular structure to optimize.

        Returns:
            A tuple containing the optimized molecular structure and a dictionary
            of properties from the optimization.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_properties(self, structure: SiteCollection):
        """Get the properties of the structure.

        Args:
            structure: The molecular structure to extract properties from.

        Returns:
            A dictionary containing the properties of the structure.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
