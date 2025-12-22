"""Base Class for    DFT Calculations."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import cupy as cp
from ase import units
from monty.json import MSONable
from pyscf.dft.libxc import XC_CODES
from pyscf.gto.basis import ALIAS, GTH_ALIAS
from pyscf.scf import hf

from jfchemistry import AtomicProperty, SystemProperty
from jfchemistry.calculators.base import WavefunctionCalculator
from jfchemistry.core.properties import Properties, PropertyClass


class PySCFSystemProperties(PropertyClass):
    """System properties of the ORCA calculation."""

    total_energy: SystemProperty


class PySCFAtomicProperties(PropertyClass):
    """Atomic properties of the ORCA calculation."""

    homo_participation_ratio: Optional[AtomicProperty] = None
    lumo_participation_ratio: Optional[AtomicProperty] = None


class PySCFProperties(Properties):
    """Properties of the PySCF calculation."""

    system: PySCFSystemProperties
    atomic: PySCFAtomicProperties


basis_sets = Literal[
    *list(ALIAS.keys()),  # type: ignore
    *list(GTH_ALIAS.keys()),  # type: ignore
]

xc_functionals = Literal[*list(XC_CODES.keys())]  # type: ignore


@dataclass
class PySCFGPUCalculator(WavefunctionCalculator, MSONable):
    """PySCF GPU DFT Calculator with full type support."""

    name: str = "PySCF GPU Calculator"
    cores: int = field(
        default=1,
        metadata={"description": "The number of CPU cores to use for parallel calculations"},
    )
    basis_set: Optional[basis_sets] = field(
        default=None,
        metadata={"description": "The basis set to use for the calculation"},
    )
    xc_functional: Optional[xc_functionals] = field(
        default=None,
        metadata={"description": "The exchange-correlation functional to use for the calculation"},
    )
    dispersion_correction: Optional[str] = field(
        default=None,
        metadata={"description": "The dispersion correction to use for the calculation"},
    )
    participation_ratio: bool = field(
        default=False,
        metadata={
            "description": "Calculate the per-atom \
                participation ratio for a series of molecular orbitals"
        },
    )
    homo_threshold: Optional[float] = field(
        default=None,
        metadata={
            "description": "The threshold energies from the HOMO orbital to be considered for the \
                participation ratio calculation in eV"
        },
    )
    lumo_threshold: Optional[float] = field(
        default=None,
        metadata={
            "description": "The threshold energies from the LUMO orbital to be considered for the \
                participation ratio calculation in eV"
        },
    )
    _properties_model: type[PySCFProperties] = PySCFProperties

    def get_properties(self, mf: hf.RHF) -> PySCFProperties:
        """Parse the properties from the output."""
        total_energy = mf.e_tot
        if self.participation_ratio:
            S = mf.get_ovlp()
            B = cp.zeros((mf.mol.natm, S.shape[0]))
            for i in range(mf.mol.nbas):
                atom_idx = mf.mol.bas_atom(i)
                B[atom_idx, i] = 1
            P_homo = []
            P_lumo = []
            if self.homo_threshold is not None:
                homo_mask = mf.mo_occ > 0.0  # type: ignore
                homo_energies = mf.mo_energy[homo_mask]  # type: ignore
                homo_energy = homo_energies[-1]
                threshold = self.homo_threshold / units.Hartree
                homo_orbitals = mf.mo_coeff[:, homo_mask]  # type: ignore
                homo_orbitals = homo_orbitals[:, homo_energies + threshold >= homo_energy]  # type: ignore
                for C_i in homo_orbitals.T:
                    for i in range(mf.mol.natm):
                        _, _, ao_start, ao_end = mf.mol.aoslice_by_atom()[i, :]
                        ao_indices = cp.arange(ao_start, ao_end)
                        B[i, ao_indices] = C_i[ao_indices]
                    P_i = B @ S @ C_i
                    P_homo.append(P_i)
            if self.lumo_threshold is not None:
                lumo_mask = mf.mo_occ == 0.0  # type: ignore
                lumo_energies = mf.mo_energy[lumo_mask]  # type: ignore
                lumo_energy = lumo_energies[0]
                threshold = self.lumo_threshold / units.Hartree
                lumo_orbitals = mf.mo_coeff[:, lumo_mask]  # type: ignore
                lumo_orbitals = lumo_orbitals[:, lumo_energies - threshold <= lumo_energy]  # type: ignore
                for C_i in lumo_orbitals.T:
                    for i in range(mf.mol.natm):
                        _, _, ao_start, ao_end = mf.mol.aoslice_by_atom()[i, :]
                        ao_indices = cp.arange(ao_start, ao_end)
                        B[i, ao_indices] = C_i[ao_indices]
                    P_i = B @ S @ C_i
                    P_lumo.append(P_i)
        system_properties = PySCFSystemProperties(
            total_energy=SystemProperty(value=total_energy, units="Ha", name="Total Energy")
        )
        atomic_properties = PySCFAtomicProperties(
            homo_participation_ratio=AtomicProperty(
                value=P_homo, units="", name="HOMO Participation Ratio"
            ),
            lumo_participation_ratio=AtomicProperty(
                value=P_lumo, units="", name="LUMO Participation Ratio"
            ),
        )
        properties = PySCFProperties(system=system_properties, atomic=atomic_properties)
        return properties
