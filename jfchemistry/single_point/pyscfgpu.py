"""Base Class for ORCA DFT Calculations."""

from dataclasses import dataclass

from gpu4pyscf import dft
from pymatgen.core import Molecule
from pyscf import gto

# Import fully typed Literal definitions
from jfchemistry.calculators.pyscfgpu import PySCFGPUCalculator
from jfchemistry.calculators.pyscfgpu.pyscfgpu_calculator import PySCFProperties
from jfchemistry.core.makers.single_molecule import SingleMoleculeMaker
from jfchemistry.core.properties import Properties
from jfchemistry.single_point.base import SinglePointEnergyCalculator


@dataclass
class PySCFGPUSinglePoint(PySCFGPUCalculator, SingleMoleculeMaker, SinglePointEnergyCalculator):
    """PySCF GPU Calculator with full type support.

    This calculator wraps the PySCF GPU package to provide
    DFT calculation capabilities.

    Attributes:
        name: Name of the calculator (default: "PySCF GPU").
        cores: Number of CPU cores to use for parallel calculations (default: 1).
        basis_set: Basis set to use for the calculation (488 options available).
        xc_functional: Exchange-correlation functional to use (195 options available).
    """

    name: str = "PySCF GPU Single Point Calculator"

    _properties_model: type[PySCFProperties] = PySCFProperties
    _filename = "input.xyz"

    def operation(
        self, molecule: Molecule
    ) -> tuple[Molecule | list[Molecule], Properties | list[Properties]]:
        """Calculate the single point energy of a molecule using PySCF GPU."""
        # Write to XYZ file
        molecule.to(self._filename, fmt="xyz")
        # Make the calculator
        mol = gto.Mole()
        mol.atom = self._filename
        mol.basis = self.basis_set
        mol.build()
        mf = dft.RKS(mol)
        mf = mf.newton()
        mf.kernel()
        properties = self.get_properties(mf)
        return molecule, properties
