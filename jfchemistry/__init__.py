"""jfchemistry: A computational chemistry workflow package.

This package provides a comprehensive framework for computational chemistry workflows
using jobflow. It supports molecular structure generation, optimization, conformer
generation, and property calculations using various computational chemistry methods.

The package is built around two main base classes:
    - SingleMoleculeMaker: For operations on molecules without 3D geometry (RDKit molecules)
    - SingleStructureMaker: For operations on structures with 3D geometry (Pymatgen structures)

Main Features:
    - Structure generation from SMILES, PubChem CID
    - Conformer generation using RDKit and CREST
    - Geometry optimization using ASE, AimNet2, ORB models, and TBLite
    - Structure modifications (protonation, deprotonation)
    - Property calculations with various quantum chemistry methods

Examples:
    >>> from jfchemistry.inputs import Smiles
    >>> from jfchemistry.generation import RDKitGeneration
    >>> from jfchemistry.optimizers import AimNet2Optimizer
    >>>
    >>> # Create a molecular structure workflow
    >>> smiles_input = Smiles()
    >>> generator = RDKitGeneration(num_conformers=10)
    >>> optimizer = AimNet2Optimizer()
    >>>
    >>> # Build workflow
    >>> smiles_job = smiles_input.make("CCO")
    >>> gen_job = generator.make(smiles_job.output["structure"])
    >>> opt_job = optimizer.make(gen_job.output["structure"])
"""

from .jfchemistry import (
    AtomicProperty,
    BondProperty,
    Properties,
    Property,
    RDMolMolecule,
    SingleMoleculeMaker,
    SingleStructureMaker,
    SystemProperty,
)

__all__ = [
    "AtomicProperty",
    "BondProperty",
    "Properties",
    "Property",
    "RDMolMolecule",
    "SingleMoleculeMaker",
    "SingleStructureMaker",
    "SystemProperty",
]
