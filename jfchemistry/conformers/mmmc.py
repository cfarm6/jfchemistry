"""CREST-based conformer generation using metadynamics.

This module provides integration with CREST (Conformer-Rotamer Ensemble
Sampling Tool) for comprehensive conformational searching using metadynamics
and GFN-xTB methods.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch_sim as ts
from ase import optimize
from multiple_minimum_monte_carlo.batch_calculation import TorchSimCalculation
from multiple_minimum_monte_carlo.calculation import ASEOptimization
from multiple_minimum_monte_carlo.conformer import Conformer
from multiple_minimum_monte_carlo.conformer_ensemble import ConformerEnsemble
from pymatgen.core.structure import Molecule

from jfchemistry.conformers.base import ConformerGeneration
from jfchemistry.core.outputs import Output
from jfchemistry.core.properties import Properties, PropertyClass, SystemProperty
from jfchemistry.optimizers.ase import ASEOptimizer
from jfchemistry.optimizers.torchsim import TorchSimOptimizer

EV_TO_KCAL = 23.0605


class MMMCSystemProperties(PropertyClass):
    """System properties of the MMMC conformer generation."""

    total_energy: SystemProperty


class MMMCProperties(Properties):
    """Properties of the MMMC conformer generation."""

    system: MMMCSystemProperties


class MMMCOutput(Output):
    """Output of the MMMC conformer generation."""

    structure: list[Molecule]


@dataclass
class MMMCConformers(ConformerGeneration):
    """Generate conformers with the multiple minimum monte carlo method."""

    name: str = "Multiple Minimum Monte Carlo Conformer Generation"
    optimizer: ASEOptimizer | TorchSimOptimizer = field(
        default_factory=lambda: ASEOptimizer,
        metadata={"description": "the calculator to use for the calculation"},
    )
    energy_window: float = field(
        default=10.0,
        metadata={"description": "the energy window for the conformer ensemble [kcal/mol]"},
    )
    max_bonds_rotate: int = field(
        default=3,
        metadata={"description": "Maximum number of rotatable bonds to rotate in each step"},
    )
    max_attempts: int = field(
        default=1000,
        metadata={
            "description": "Maximum number of times to try and rotate dihedrals per iteration"
        },
    )
    angle_step: float = field(
        default=30.0, metadata={"description": "Step size for bond rotation [degrees]"}
    )
    rmsd_threshold: float = field(
        default=0.3, metadata={"description": "RMSD threshold for conformer selection [Angstrom]"}
    )
    initial_optimization: bool = field(
        default=True,
        metadata={"description": "Whether to perform initial optimization of the structure"},
    )
    random_walk: bool = field(
        default=False, metadata={"description": "If True, use random walk for bond rotations"}
    )
    reduce_angle: bool = field(
        default=False,
        metadata={"description": " If True, reduce angle step size during the search"},
    )
    reduce_angle_every: int = field(
        default=50,
        metadata={
            "description": "Number of iterations to reduce angle step size.\
                Only applicable if reduce_angle is True"
        },
    )
    reduce_angle_by: int = field(
        default=2,
        metadata={
            "description": "Factor to reduce angle step size.\
                Only applicable if reduce_angle is True"
        },
    )
    only_heavy: bool = field(
        default=False,
        metadata={"description": "If True, only heavy atoms are considered for bond rotations"},
    )
    parallel: bool = field(default=False, metadata={"description": "If True, run MMMC in parallel"})
    num_cpus: int = field(
        default=1,
        metadata={
            "description": "Number of CPUs to use for MMMC. Only applicable if parallel is True"
        },
    )

    _filename: str = "molecule.xyz"
    _optimizer: Optional[ASEOptimizer | TorchSimOptimizer] = field(default=None)
    _properties_model: type[MMMCProperties] = MMMCProperties
    _output_model: type[MMMCProperties] = MMMCProperties

    def operation(
        self, structure: Molecule
    ) -> tuple[Molecule | list[Molecule], list[MMMCProperties]]:
        """Operation of the MMMC conformer generation."""
        # Write to XYZ file
        structure.to(self._filename)
        # Create conformer
        conformer = Conformer(input_xyz=self._filename, charge=int(structure.charge))
        # Create Optimizer
        if isinstance(self.optimizer, ASEOptimizer):
            atoms = self.optimizer.calculator.set_calculator(
                structure.to_ase_atoms(),
                charge=int(structure.charge),
                spin_multiplicity=int(structure.spin_multiplicity),
            )
            calc = ASEOptimization(
                atoms.calc,
                optimizer=getattr(optimize, self.optimizer.optimizer),
                fmax=self.optimizer.fmax,
                max_cycles=self.optimizer.steps,
            )
        elif isinstance(self.optimizer, TorchSimOptimizer):
            model = self.optimizer.calculator.get_model()
            calc = TorchSimCalculation(
                model=model,
                optimizer=ts.Optimizer.fire,
                max_cycles=self.optimizer.max_steps,
                device=self.optimizer.calculator.device,
            )
        # Build conformer ensemble
        conformer_ensemble = ConformerEnsemble(
            conformer=conformer,
            calc=calc,
            energy_window=self.energy_window,
            max_bonds_rotate=self.max_bonds_rotate,
            max_attempts=self.max_attempts,
            angle_step=self.angle_step,
            rmsd_threshold=self.rmsd_threshold,
            initial_optimization=self.initial_optimization,
            random_walk=self.random_walk,
            reduce_angle=self.reduce_angle,
            reduce_angle_every=self.reduce_angle_every,
            reduce_angle_by=self.reduce_angle_by,
            only_heavy=self.only_heavy,
            parallel=self.parallel,
            num_cpus=self.num_cpus,
        )
        # RUN
        conformer_ensemble.run_monte_carlo()
        print(conformer_ensemble.final_energies)
        # Get the best conformer
        if conformer.atoms is None:
            raise ValueError("Atoms are not set")
        if conformer.atoms is None or isinstance(conformer.atoms, list):
            raise ValueError("Atoms are not set")
        atom_symbols = list(conformer.atoms.get_chemical_symbols())
        molecules = [
            Molecule(species=atom_symbols, coords=coords)
            for coords in conformer_ensemble.final_ensemble
        ]
        from ase import io

        for _, molecule in enumerate(molecules):
            io.write("conformer.xyz", molecule.to_ase_atoms(), append=True)
        # Return the properties
        properties = [
            MMMCProperties(
                system=MMMCSystemProperties(
                    total_energy=SystemProperty(
                        name="total_energy", value=energy / EV_TO_KCAL, units="eV"
                    )
                )
            )
            for energy in conformer_ensemble.final_energies
        ]

        return (molecules, properties)
