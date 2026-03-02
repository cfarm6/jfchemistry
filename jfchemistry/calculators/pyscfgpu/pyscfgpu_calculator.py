"""PySCF GPU DFT calculator and related classes."""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import cupy as cp
import numpy as np
from ase import units
from gpu4pyscf import dft
from monty.json import MSONable
from pint import Quantity
from pyscf import gto, lo
from pyscf.dft.libxc import XC_CODES
from pyscf.gto.basis import ALIAS, GTH_ALIAS
from pyscf.scf import hf

from jfchemistry import AtomicProperty, SystemProperty, ureg
from jfchemistry.calculators.base import WavefunctionCalculator
from jfchemistry.core.properties import OrbitalProperty, Properties, PropertyClass
from jfchemistry.core.solvation import ImplicitSolventConfig, to_pyscfgpu
from jfchemistry.core.unit_utils import to_magnitude


class PySCFOrbitalProperties(PropertyClass):
    """Orbital properties of the PySCF calculation."""

    mo_coefficients: OrbitalProperty
    mo_energies: OrbitalProperty
    mo_occupations: OrbitalProperty
    overlap_matrix: OrbitalProperty
    localized_mo_coefficients: Optional[OrbitalProperty] = None
    localized_mo_indices: Optional[OrbitalProperty] = None


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
localized_orbital_methods = Literal[
    "boys",
    "pipek_mezey",
    "edmiston_ruedenberg",
    "cholesky",
    "iao",
    "ibo",
    "nao",
    "orth_ao_lowdin",
    "orth_ao_meta_lowdin",
    "orth_ao_nao",
    "vvo",
    "livvo",
]
localized_orbital_spaces = Literal["all", "occupied", "virtual"]
pm_pop_methods = Literal["mulliken", "meta-lowdin", "lowdin", "iao", "becke"]
ibo_loc_methods = Literal["IBO", "PM"]


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
        store_localized_orbitals: Whether to compute and store localized orbital coefficients \
            from ``pyscf.lo`` (default: False).
        localized_orbital_method: The localization method from ``pyscf.lo`` to use when \
            ``store_localized_orbitals`` is True (default: "boys").
        localized_orbital_space: Which orbitals are localized: all orbitals, occupied only, or \
            virtual only (default: "occupied").
        localized_orbital_pm_pop_method: Pipek-Mezey population method (default: "meta-lowdin").
        localized_orbital_iao_minao: Reference basis for IAO/IBO-related methods (default: "minao").
        localized_orbital_iao_orthogonalize: Apply ``lo.vec_lowdin`` to IAOs before use
            (default: True).
        localized_orbital_ibo_locmethod: IBO localization algorithm selection (default: "IBO").
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
    implicit_solvent: Optional[ImplicitSolventConfig] = field(
        default=None,
        metadata={
            "description": "Unified implicit-solvent config. PySCF-GPU adapter currently "
            "supports only model='none'."
        },
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
    store_localized_orbitals: bool = field(
        default=False,
        metadata={"description": "Whether to compute and store localized orbitals"},
    )
    localized_orbital_method: localized_orbital_methods = field(
        default="boys",
        metadata={
            "description": "Localization method to use from pyscf.lo "
            "(boys, pipek_mezey, edmiston_ruedenberg, cholesky)"
        },
    )
    localized_orbital_space: localized_orbital_spaces = field(
        default="occupied",
        metadata={"description": "Which MO subspace to localize (all, occupied, virtual)"},
    )
    localized_orbital_pm_pop_method: pm_pop_methods = field(
        default="meta-lowdin",
        metadata={
            "description": "Pipek-Mezey population method "
            "(mulliken, meta-lowdin, lowdin, iao, becke)"
        },
    )
    localized_orbital_init_guess: Optional[str] = field(
        default=None,
        metadata={
            "description": "Optional initial guess keyword for iterative localizers "
            "(e.g. atomic, cholesky)"
        },
    )
    localized_orbital_conv_tol: Optional[float] = field(
        default=None,
        metadata={"description": "Optional convergence tolerance for iterative localizers"},
    )
    localized_orbital_conv_tol_grad: Optional[float] = field(
        default=None,
        metadata={
            "description": "Optional gradient convergence tolerance for iterative localizers"
        },
    )
    localized_orbital_max_cycle: Optional[int] = field(
        default=None,
        metadata={"description": "Optional macro-iteration cap for iterative localizers"},
    )
    localized_orbital_max_iters: Optional[int] = field(
        default=None,
        metadata={"description": "Optional micro-iteration cap for iterative localizers"},
    )
    localized_orbital_max_stepsize: Optional[float] = field(
        default=None,
        metadata={"description": "Optional step size for iterative localizers"},
    )
    localized_orbital_pm_exponent: int = field(
        default=2,
        metadata={"description": "Pipek-Mezey exponent (2 or 4)"},
    )
    localized_orbital_iao_minao: str = field(
        default="minao",
        metadata={"description": "Reference basis for IAO construction"},
    )
    localized_orbital_iao_lindep_threshold: float = field(
        default=1e-8,
        metadata={"description": "Linear-dependence threshold used by IAO builder"},
    )
    localized_orbital_iao_orthogonalize: bool = field(
        default=True,
        metadata={"description": "Orthogonalize IAOs with vec_lowdin before storing/using"},
    )
    localized_orbital_ibo_locmethod: ibo_loc_methods = field(
        default="IBO",
        metadata={"description": "IBO localization method: IBO or PM"},
    )
    localized_orbital_ibo_exponent: int = field(
        default=4,
        metadata={"description": "IBO localization exponent"},
    )
    localized_orbital_ibo_grad_tol: float = field(
        default=1e-8,
        metadata={"description": "IBO localization gradient tolerance"},
    )
    localized_orbital_ibo_max_iter: int = field(
        default=200,
        metadata={"description": "IBO localization iteration cap"},
    )
    localized_orbital_pre_orth_ao: Optional[str] = field(
        default="ANO",
        metadata={
            "description": "Reference AO basis for orth_ao projection; set None to skip "
            "pre-orthogonalization"
        },
    )
    _properties_model: type[PySCFProperties] = PySCFProperties

    def __post_init__(self):
        """Normalize unit-bearing attributes and validate solvent support."""
        if self.homo_threshold is not None and isinstance(self.homo_threshold, Quantity):
            object.__setattr__(self, "homo_threshold", to_magnitude(self.homo_threshold, "eV"))
        if self.lumo_threshold is not None and isinstance(self.lumo_threshold, Quantity):
            object.__setattr__(self, "lumo_threshold", to_magnitude(self.lumo_threshold, "eV"))
        if self.implicit_solvent is not None:
            to_pyscfgpu(self.implicit_solvent)

    def _setup_mf(self, mol: gto.Mole) -> dft.RKS:
        mf = dft.RKS(mol, xc=self.xc_functional)
        if self.joltqc:
            try:
                import jqc.pyscf  # ty:ignore[unresolved-import]

                mf = jqc.pyscf.apply(mf)
            except ImportError as err:
                raise ImportError(
                    "JoltQC is not installed. Please install JoltQC to use JoltQC for\
                    the calculation. See https://github.com/ByteDance-Seed/JoltQC\
                    for more information."
                ) from err
            return mf
        return mf

    def _get_localization_mask(self, mo_occ: np.ndarray) -> np.ndarray:
        """Return boolean mask selecting which MOs to localize."""
        if self.localized_orbital_space == "all":
            return np.ones_like(mo_occ, dtype=bool)
        if self.localized_orbital_space == "occupied":
            return mo_occ > 0.0
        return mo_occ == 0.0

    def _configure_localizer(self, localizer: Any) -> Any:
        """Apply optional solver settings to iterative PySCF localizers."""
        if self.localized_orbital_init_guess is not None and hasattr(localizer, "init_guess"):
            localizer.init_guess = self.localized_orbital_init_guess
        if self.localized_orbital_conv_tol is not None and hasattr(localizer, "conv_tol"):
            localizer.conv_tol = self.localized_orbital_conv_tol
        if self.localized_orbital_conv_tol_grad is not None and hasattr(localizer, "conv_tol_grad"):
            localizer.conv_tol_grad = self.localized_orbital_conv_tol_grad
        if self.localized_orbital_max_cycle is not None and hasattr(localizer, "max_cycle"):
            localizer.max_cycle = self.localized_orbital_max_cycle
        if self.localized_orbital_max_iters is not None and hasattr(localizer, "max_iters"):
            localizer.max_iters = self.localized_orbital_max_iters
        if self.localized_orbital_max_stepsize is not None and hasattr(localizer, "max_stepsize"):
            localizer.max_stepsize = self.localized_orbital_max_stepsize
        return localizer

    def _compute_iaos(
        self, mf: hf.RHF, orbocc: np.ndarray, overlap_matrix: np.ndarray
    ) -> np.ndarray:
        """Build IAOs from occupied MOs with optional Lowdin orthogonalization."""
        iaos = lo.iao.iao(
            mf.mol,
            orbocc,
            minao=self.localized_orbital_iao_minao,
            lindep_threshold=self.localized_orbital_iao_lindep_threshold,
        )
        if self.localized_orbital_iao_orthogonalize:
            iaos = lo.vec_lowdin(iaos, overlap_matrix)
        return np.asarray(iaos)

    def _build_localization_context(
        self, mf: hf.RHF, mo_coeff: np.ndarray, mo_occ: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Create masks and blocks needed by localization handlers."""
        mo_mask = self._get_localization_mask(mo_occ)
        occ_mask = mo_occ > 0.0
        vir_mask = mo_occ == 0.0
        return {
            "mo_mask": mo_mask,
            "occ_mask": occ_mask,
            "vir_mask": vir_mask,
            "mo_block": mo_coeff[:, mo_mask],
            "occ_block": mo_coeff[:, occ_mask],
            "vir_block": mo_coeff[:, vir_mask],
            "overlap": np.asarray(mf.get_ovlp()),
        }

    def _index_tuple(self, mask: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return index tuple for canonical MO mask."""
        return None, np.where(mask)[0]

    def _handle_boys(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localizer = self._configure_localizer(lo.Boys(mf.mol, ctx["mo_block"]))
        localized = np.asarray(localizer.kernel())
        return localized, np.where(ctx["mo_mask"])[0]

    def _handle_pipek_mezey(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localizer = lo.PM(mf.mol, ctx["mo_block"], mf=mf)
        localizer.pop_method = self.localized_orbital_pm_pop_method
        localizer.exponent = self.localized_orbital_pm_exponent
        localizer = self._configure_localizer(localizer)
        localized = np.asarray(localizer.kernel())
        return localized, np.where(ctx["mo_mask"])[0]

    def _handle_edmiston_ruedenberg(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localizer = self._configure_localizer(lo.ER(mf.mol, ctx["mo_block"]))
        localized = np.asarray(localizer.kernel())
        return localized, np.where(ctx["mo_mask"])[0]

    def _handle_cholesky(
        self, _mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localized = np.asarray(lo.cholesky_mos(ctx["mo_block"]))
        return localized, np.where(ctx["mo_mask"])[0]

    def _handle_iao(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not np.any(ctx["occ_mask"]):
            return None, None
        localized = self._compute_iaos(mf, ctx["occ_block"], ctx["overlap"])
        return localized, np.where(ctx["occ_mask"])[0]

    def _handle_ibo(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not np.any(ctx["occ_mask"]):
            return None, None
        iaos = self._compute_iaos(mf, ctx["occ_block"], ctx["overlap"])
        if self.localized_orbital_ibo_locmethod == "PM":
            iaos = np.asarray(
                lo.iao.iao(
                    mf.mol,
                    ctx["occ_block"],
                    minao=self.localized_orbital_iao_minao,
                    lindep_threshold=self.localized_orbital_iao_lindep_threshold,
                )
            )
        localized = lo.ibo.ibo(
            mf.mol,
            ctx["occ_block"],
            locmethod=self.localized_orbital_ibo_locmethod,
            iaos=iaos,
            exponent=self.localized_orbital_ibo_exponent,
            grad_tol=self.localized_orbital_ibo_grad_tol,
            max_iter=self.localized_orbital_ibo_max_iter,
            minao=self.localized_orbital_iao_minao,
        )
        return np.asarray(localized), np.where(ctx["occ_mask"])[0]

    def _handle_nao(
        self, mf: hf.RHF, _ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return np.asarray(lo.nao.nao(mf.mol, mf)), None

    def _handle_orth_ao_lowdin(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localized = lo.orth.orth_ao(
            mf.mol,
            method="lowdin",
            pre_orth_ao=self.localized_orbital_pre_orth_ao,
            s=ctx["overlap"],
        )
        return np.asarray(localized), None

    def _handle_orth_ao_meta_lowdin(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localized = lo.orth.orth_ao(
            mf.mol,
            method="meta_lowdin",
            pre_orth_ao=self.localized_orbital_pre_orth_ao,
            s=ctx["overlap"],
        )
        return np.asarray(localized), None

    def _handle_orth_ao_nao(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        localized = lo.orth.orth_ao(mf, method="nao", s=ctx["overlap"])
        return np.asarray(localized), None

    def _handle_vvo(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not np.any(ctx["occ_mask"]) or not np.any(ctx["vir_mask"]):
            return None, None
        localized = lo.vvo.vvo(mf.mol, ctx["occ_block"], ctx["vir_block"])
        return np.asarray(localized), np.where(ctx["vir_mask"])[0]

    def _handle_livvo(
        self, mf: hf.RHF, ctx: dict[str, np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not np.any(ctx["occ_mask"]) or not np.any(ctx["vir_mask"]):
            return None, None
        localized = lo.vvo.livvo(mf.mol, ctx["occ_block"], ctx["vir_block"])
        return np.asarray(localized), np.where(ctx["vir_mask"])[0]

    def _compute_localized_orbitals(
        self, mf: hf.RHF, mo_coeff: np.ndarray, mo_occ: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute localized orbital coefficients for a selected MO subspace."""
        if not self.store_localized_orbitals:
            return None, None

        ctx = self._build_localization_context(mf, mo_coeff, mo_occ)
        if not np.any(ctx["mo_mask"]):
            return None, None

        handlers: dict[
            str,
            Callable[
                [hf.RHF, dict[str, np.ndarray]],
                tuple[Optional[np.ndarray], Optional[np.ndarray]],
            ],
        ] = {
            "boys": self._handle_boys,
            "pipek_mezey": self._handle_pipek_mezey,
            "edmiston_ruedenberg": self._handle_edmiston_ruedenberg,
            "cholesky": self._handle_cholesky,
            "iao": self._handle_iao,
            "ibo": self._handle_ibo,
            "nao": self._handle_nao,
            "orth_ao_lowdin": self._handle_orth_ao_lowdin,
            "orth_ao_meta_lowdin": self._handle_orth_ao_meta_lowdin,
            "orth_ao_nao": self._handle_orth_ao_nao,
            "vvo": self._handle_vvo,
            "livvo": self._handle_livvo,
        }
        handler = handlers.get(self.localized_orbital_method)
        if handler is None:
            raise ValueError(
                f"Unsupported localized orbital method: {self.localized_orbital_method}"
            )
        return handler(mf, ctx)

    def _get_properties(self, mf: hf.RHF) -> PySCFProperties:
        """Parse the properties from the output."""
        total_energy = mf.e_tot
        overlap_matrix = mf.get_ovlp()
        if self.participation_ratio:
            S = overlap_matrix
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
        localized_mo_coeff, localized_mo_indices = self._compute_localized_orbitals(
            mf=mf, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
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
            overlap_matrix=OrbitalProperty(
                name="Overlap Matrix",
                value=cp.asnumpy(overlap_matrix).tolist(),
                description="Atomic orbital overlap matrix (S)",
            ),
            localized_mo_coefficients=OrbitalProperty(
                name=f"Localized Molecular Orbital Coefficients ({self.localized_orbital_method})",
                value=localized_mo_coeff.tolist(),
                description=(
                    "Localized MO coefficients for the selected orbital space "
                    f"({self.localized_orbital_space})"
                ),
            )
            if localized_mo_coeff is not None
            else None,
            localized_mo_indices=OrbitalProperty(
                name=f"Localized Orbital Indices ({self.localized_orbital_space})",
                value=localized_mo_indices.tolist(),
                description="Indices of canonical molecular orbitals used for localization",
            )
            if localized_mo_indices is not None
            else None,
        )
        properties = PySCFProperties(
            system=system_properties, atomic=atomic_properties, orbital=orbital_properties
        )
        return properties
