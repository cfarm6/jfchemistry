"""jfchemistry: A computational chemistry workflow package.

This package provides a comprehensive framework for computational chemistry workflows
using jobflow. It supports molecular structure generation, optimization, conformer
generation, and property calculations using various computational chemistry methods.

The package is built around two main base classes:
    - SingleRDMoleculeMaker: For operations on molecules without 3D geometry (RDKit molecules)
    - SingleStructureMaker: For operations on structures with 3D geometry (Pymatgen structures)

Main Features:
    - Structure generation from SMILES, PubChem CID
    - Conformer generation using RDKit and CREST
    - Geometry optimization using ASE, AimNet2, ORB models, and TBLite
    - Structure modifications (protonation, deprotonation)
    - Property calculations with various quantum chemistry methods
"""

from pint import UnitRegistry, set_application_registry

# from jfchemistry.core.makers.single_rdmolecule import SingleRDMoleculeMaker
# from jfchemistry.core.makers.single_structure import SingleStructureMaker
from jfchemistry.core.properties import AtomicProperty, BondProperty, Property, SystemProperty


def setup_computational_chemistry_units():
    """Creates a UnitRegistry with computational chemistry energy units."""
    ureg = UnitRegistry(system="atomic")
    ureg.define("wavenumber = planck_constant * speed_of_light / centimeter = cm^-1 = kayser")
    ureg.define("kcal_per_mol = kilocalorie / avogadro_number")
    ureg.define("kJ_per_mol = kJ / avogadro_number")
    return ureg


ureg = setup_computational_chemistry_units()
set_application_registry(ureg)
Q_ = ureg.Quantity

__all__ = [
    "AtomicProperty",
    "BondProperty",
    "Property",
    # "RDMolMolecule",
    # "SingleRDMoleculeMaker",
    # "SingleStructureMaker",
    "SystemProperty",
    "ureg",
    "Q_",
]
