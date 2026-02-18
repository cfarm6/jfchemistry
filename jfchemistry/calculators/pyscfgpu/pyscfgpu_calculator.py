"""PySCF GPU DFT calculator and related classes."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import cupy as cp
import numpy as np
from ase import units
from gpu4pyscf import dft
from monty.json import MSONable
from pint import Quantity
from pyscf import gto
from pyscf.dft.libxc import XC_CODES
from pyscf.gto.basis import ALIAS, GTH_ALIAS
from pyscf.scf import hf

from jfchemistry import AtomicProperty, SystemProperty, ureg
from jfchemistry.calculators.base import WavefunctionCalculator
from jfchemistry.core.properties import OrbitalProperty, Properties, PropertyClass
from jfchemistry.core.unit_utils import to_magnitude


class PySCFOrbitalProperties(PropertyClass):
    """Orbital properties of the PySCF calculation."""

    mo_coefficients: OrbitalProperty
    mo_energies: OrbitalProperty
    mo_occupations: OrbitalProperty


class PySCFSystemProperties(PropertyClass):
    """System properties of the PySCF calculation."""

    total_energy: SystemProperty


class PySCFAtomicProperties(PropertyClass):
    """Atomic properties of the PySCF calculation."""

    homo_participation_ratio: Optional[AtomicProperty] = None
    lumo_participation_ratio: Optional[AtomicProperty] = None


class PySCFProperties(Properties):
    """Properties of the PySCF calculation."""

    system: PySCFSystemProperties
    atomic: PySCFAtomicProperties
    orbital: PySCFOrbitalProperties


basis_sets = Literal[
    *list(ALIAS.keys()),  # type: ignore
    *list(GTH_ALIAS.keys()),  # type: ignore
]

xc_functionals = Literal[*list(XC_CODES.keys())]  # type: ignore


@dataclass
class PySCFGPUCalculator(WavefunctionCalculator, MSONable):
    """PySCF GPU DFT Calculator with full type support.

    Units:
        Pass a float in the listed unit or a pint Quantity (e.g. ``jfchemistry.ureg``
        or ``jfchemistry.Q_``):

        - homo_threshold: [eV]
        - lumo_threshold: [eV]

    Attributes:
        name: Name of the calculator (default: "PySCF GPU Calculator").
        cores: The number of CPU cores to use for parallel calculations (default: 1).
        basis_set: The basis set to use for the calculation (default: None).
        xc_functional: The exchange-correlation functional to use for \
            the calculation (default: None).
        dispersion_correction: The dispersion correction to use for the calculation (default: None).
        participation_ratio: Whether to calculate the per-atom participation ratio for a series of \
            molecular orbitals (default: False).
        homo_threshold: The threshold energies from the HOMO orbital to be considered for the \
            participation ratio calculation [eV] (default: None). Accepts float or pint Quantity.
        lumo_threshold: The threshold energies from the LUMO orbital to be considered for the \
            participation ratio calculation [eV] (default: None). Accepts float or pint Quantity.
    """

    name: str = "PySCF GPU Calculator"
    cores: int = field(
        default=1,
        metadata={"description": "The number of CPU cores to use for parallel calculations"},
    )
    joltqc: bool = field(
        default=False,
        metadata={"description": "Whether to use JoltQC for the calculation"},
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
    homo_threshold: Optional[float | Quantity] = field(
        default=None,
        metadata={
            "description": "The threshold energies from the HOMO orbital to be considered for the "
            "participation ratio calculation [eV]. Accepts float or pint Quantity.",
            "unit": "eV",
        },
    )
    lumo_threshold: Optional[float | Quantity] = field(
        default=None,
        metadata={
            "description": "The threshold energies from the LUMO orbital to be considered for the "
            "participation ratio calculation [eV]. Accepts float or pint Quantity.",
            "unit": "eV",
        },
    )
    _properties_model: type[PySCFProperties] = PySCFProperties

    def __post_init__(self):
        """Normalize unit-bearing attributes."""
        if self.homo_threshold is not None and isinstance(self.homo_threshold, Quantity):
            object.__setattr__(self, "homo_threshold", to_magnitude(self.homo_threshold, "eV"))
        if self.lumo_threshold is not None and isinstance(self.lumo_threshold, Quantity):
            object.__setattr__(self, "lumo_threshold", to_magnitude(self.lumo_threshold, "eV"))

    def _setup_mf(self, mol: gto.Mole) -> dft.RKS:
        mf = dft.RKS(mol, xc=self.xc_functional)
        if self.joltqc:
            try:
                import jqc.pyscf  # type: ignore  # Optional dependency, may not be available

                mf = jqc.pyscf.apply(mf)
            except ImportError as err:
                raise ImportError(
                    "JoltQC is not installed. Please install JoltQC to use JoltQC for\
                    the calculation. See https://github.com/ByteDance-Seed/JoltQC\
                    for more information."
                ) from err
            return mf
        return mf

    def _get_properties(self, mf: hf.RHF) -> PySCFProperties:
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
                homo_orbitals = homo_orbitals[:, homo_energies + threshold >= homo_energy]
                for C_i in homo_orbitals.T:
                    for i in range(mf.mol.natm):
                        _, _, ao_start, ao_end = mf.mol.aoslice_by_atom()[i, :]
                        ao_indices = cp.arange(ao_start, ao_end)
                        B[i, ao_indices] = C_i[ao_indices]
                    P_i = B @ S @ C_i
                    P_homo.append(P_i)
            if self.lumo_threshold is not None:
                lumo_mask = mf.mo_occ == 0.0
                lumo_energies = mf.mo_energy[lumo_mask]  # type: ignore
                lumo_energy = lumo_energies[0]
                threshold = self.lumo_threshold / units.Hartree
                lumo_orbitals = mf.mo_coeff[:, lumo_mask]  # type: ignore
                lumo_orbitals = lumo_orbitals[:, lumo_energies - threshold <= lumo_energy]
                for C_i in lumo_orbitals.T:
                    for i in range(mf.mol.natm):
                        _, _, ao_start, ao_end = mf.mol.aoslice_by_atom()[i, :]
                        ao_indices = cp.arange(ao_start, ao_end)
                        B[i, ao_indices] = C_i[ao_indices]
                    P_i = B @ S @ C_i
                    P_lumo.append(P_i)
        system_properties = PySCFSystemProperties(
            total_energy=SystemProperty(value=total_energy * ureg.hartree, name="Total Energy")
        )
        atomic_properties = PySCFAtomicProperties(
            homo_participation_ratio=AtomicProperty(value=P_homo, name="HOMO Participation Ratio")
            if self.participation_ratio
            else None,
            lumo_participation_ratio=AtomicProperty(value=P_lumo, name="LUMO Participation Ratio")
            if self.participation_ratio
            else None,
        )
        mo_coeff = np.asarray(mf.mo_coeff)
        mo_energy = np.asarray(mf.mo_energy)
        mo_occ = np.asarray(mf.mo_occ)
        orbital_properties = PySCFOrbitalProperties(
            mo_coefficients=OrbitalProperty(
                name="Molecular Orbital Coefficients",
                value=mo_coeff.tolist(),
                description="Coefficients of molecular orbitals in the basis set",
            ),
            mo_energies=OrbitalProperty(
                name="Orbital Energies",
                value=(mo_energy * ureg.hartree).to(ureg.eV),
                description="Orbital energies",
            ),
            mo_occupations=OrbitalProperty(
                name="Orbital Occupations",
                value=mo_occ.tolist(),
                description="Orbital occupation numbers",
            ),
        )
        properties = PySCFProperties(
            system=system_properties, atomic=atomic_properties, orbital=orbital_properties
        )
        return properties
