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
"""

from .base_classes import (
    AtomicProperty,
    BondProperty,
    Property,
    RDMolMolecule,
    SystemProperty,
)
from .base_jobs import SingleMoleculeMaker, SingleStructureMaker

__all__ = [
    "AtomicProperty",
    "BondProperty",
    "Property",
    "RDMolMolecule",
    "SingleMoleculeMaker",
    "SingleStructureMaker",
    "SystemProperty",
]
