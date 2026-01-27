"""TBLite calculator for extended tight-binding calculations.

This module provides integration with TBLite (Tight-Binding Library) for
performing GFN-xTB semi-empirical quantum chemistry calculations.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from monty.json import MSONable

from jfchemistry import ureg
from jfchemistry.calculators.ase.ase_calculator import ASECalculator
from jfchemistry.calculators.base import SemiempiricalCalculator
from jfchemistry.core.properties import (
    AtomicProperty,
    BondProperty,
    OrbitalProperty,
    Properties,
    PropertyClass,
    SystemProperty,
)

BOND_ORDER_THRESHOLD = 0.1

TBLiteSolvationType = Literal["alpb"]
TBLiteSolventType = Literal[
    "Acetone",
    "Acetonitrile",
    "Aniline",
    "Benzaldehyde",
    "Benzene",
    "Dichloromethane",
    "Chloroform",
    "Carbon Disulfide",
    "Dioxane",
    "DMF",
    "DMSO",
    "Ethanol",
    "Ether",
    "Ethylacetate",
    "Furane",
    "Hexadecane",
    "Hexane",
    "Methanol",
    "Nitromethane",
    "Octanol",
    "Wet Octanol",
    "Phenol",
    "Toluene",
    "THF",
    "Water",
]


class TBLiteAtomicProperties(PropertyClass):
    """Atomic properties of the TBLite calculator."""

    tblite_partial_charges: AtomicProperty
    tblite_forces: Optional[AtomicProperty] = None


class TBLiteBondProperties(PropertyClass):
    """Bond properties of the TBLite calculator."""

    tblite_wiberg_bond_orders: BondProperty


class TBLiteOrbitalProperties(PropertyClass):
    """Orbital properties of the TBLite calculator."""

    tblite_orbital_energies: OrbitalProperty
    tblite_orbital_occupations: OrbitalProperty
    tblite_orbital_coefficients: OrbitalProperty


class TBLiteSystemProperties(PropertyClass):
    """System properties of the TBLite calculator."""

    total_energy: SystemProperty
    dipole_moment: SystemProperty
    quadrupole_moment: SystemProperty
    density_matrix: SystemProperty
    homo_lumo_gap: SystemProperty
    homo_energy: SystemProperty
    lumo_energy: SystemProperty


class TBLiteProperties(Properties):
    """Properties of the TBLite calculator."""

    system: Optional[TBLiteSystemProperties] = None
    atomic: Optional[TBLiteAtomicProperties] = None
    bond: Optional[TBLiteBondProperties] = None
    orbital: Optional[TBLiteOrbitalProperties] = None


@dataclass
class TBLiteCalculator(ASECalculator, SemiempiricalCalculator, MSONable):
    """TBLite calculator for GFN-xTB semi-empirical methods.

    TBLite provides implementations of the GFN (Geometrical-dependent
    Forcefield for Noncovalent interactions) extended tight-binding methods
    developed by the Grimme group. These methods offer a good balance between
    accuracy and computational efficiency for large molecular systems.

    The calculator computes extensive molecular properties including energies,
    partial charges, bond orders, molecular orbitals, dipole/quadrupole moments,
    and HOMO-LUMO gaps.

    Attributes:
        name: Name of the calculator (default: "TBLite Calculator").
        method: Semi-empirical method to use. Options:
            - "GFN2-xTB": GFN2-xTB method (default, recommended for most cases)
            - "GFN1-xTB": GFN1-xTB method
            - "IPEA1-xTB": IPEA1-xTB method
        charge: Molecular charge override. If None, uses charge from structure.
        multiplicity: Spin multiplicity override. If None, uses spin from structure.
        accuracy: Numerical accuracy parameter (default: 1.0).
        electronic_temperature: Electronic temperature in Kelvin for Fermi smearing
            (default: 300.0).
        max_iterations: Maximum SCF iterations (default: 250).
        initial_guess: Initial guess for electronic structure. Options:
            - "sad": Superposition of atomic densities (default)
            - "eeq": Electronegativity equilibration
        mixer_damping: Damping parameter for SCF mixing (default: 0.4).
        verbosity: Output verbosity level (default: 0).

    Examples:
        >>> from jfchemistry.calculators import TBLiteCalculator # doctest: +SKIP
        >>> from ase.build import molecule # doctest: +SKIP
        >>> from pymatgen.core import Molecule # doctest: +SKIP
        >>> mol = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
        >>> # Create calculator with custom settings
        >>> calc = TBLiteCalculator(  # doctest: +SKIP
        ...     method="GFN2-xTB", # doctest: +SKIP
        ...     accuracy=0.1,  # Tighter convergence # doctest: +SKIP
        ...     max_iterations=500 # doctest: +SKIP
        ... ) # doctest: +SKIP
        >>>
        >>> # Compute properties
        >>> atoms = mol.to_ase_atoms() # doctest: +SKIP
        >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
        >>> props = calc.get_properties(atoms) # doctest: +SKIP

    """

    name: str = "TBLite Calculator"
    method: Literal["GFN1-xTB", "GFN2-xTB", "IPEA1-xTB"] = field(
        default="GFN2-xTB", metadata={"description": "The method to use"}
    )
    accuracy: float = field(default=1.0, metadata={"description": "The accuracy to use"})
    electronic_temperature: float = field(
        default=300.0, metadata={"description": "The electronic temperature to use"}
    )
    max_iterations: int = field(
        default=250, metadata={"description": "The maximum number of iterations to use"}
    )
    initial_guess: Literal["sad", "eeq"] = field(
        default="sad", metadata={"description": "The initial guess to use"}
    )
    mixer_damping: float = field(default=0.4, metadata={"description": "The mixer damping to use"})
    verbosity: int = field(default=0, metadata={"description": "The verbosity to use"})
    _properties_model: type[TBLiteProperties] = TBLiteProperties
    solvation: Optional[TBLiteSolvationType] = field(
        default=None, metadata={"description": "The solvation model to use"}
    )
    solvent: Optional[TBLiteSolventType] = field(
        default=None, metadata={"description": "The solvent to use"}
    )

    def _set_calculator(self, atoms: Atoms, charge: float = 0, spin_multiplicity: int = 1) -> Atoms:
        """Set the TBLite calculator on the atoms object.

        Configures and attaches a TBLite GFN-xTB calculator to the atoms object
        with the specified method, charge, and SCF parameters.

        Args:
            atoms: ASE Atoms object to attach calculator to.
            charge: Total molecular charge (default: 0). Overridden by self.charge if set.
            spin_multiplicity: Spin multiplicity 2S+1 (default: 1). Overridden by
                self.multiplicity if set.

        Returns:
            ASE Atoms object with TBLite calculator attached.

        Raises:
            ImportError: If the 'tblite' package is not installed.

        Examples:
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from jfchemistry.calculators import TBLiteCalculator # doctest: +SKIP
            >>>
            >>> # Create calculator and set up a water molecule
            >>> calc = TBLiteCalculator(method="GFN2-xTB") # doctest: +SKIP
            >>> atoms = molecule("H2O") # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>>
            >>> props = atoms.get_properties() # doctest: +SKIP
            >>> energy = props["Global"]["Total Energy [eV]"] # doctest: +SKIP
        """
        try:
            from tblite.ase import TBLite  # doctest: +SKIP
        except ImportError as e:
            raise ImportError(
                "The 'tblite' package is required to use TBLiteCalculator but is not available. "
                "Please install it with: pip install tblite or conda install tblite-python"
            ) from e
        if self.charge is not None:
            charge = self.charge
        if self.spin_multiplicity is not None:
            spin_multiplicity = self.spin_multiplicity
        atoms.calc = TBLite(
            method=self.method,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            accuracy=self.accuracy,
            electronic_temperature=self.electronic_temperature,
            max_iterations=self.max_iterations,
            initial_guess=self.initial_guess,
            mixer_damping=self.mixer_damping,
            verbosity=self.verbosity,
            solvation=(self.solvation, self.solvent) if self.solvation and self.solvent else None,
        )

        return atoms

    def _get_properties(self, atoms: Atoms) -> TBLiteProperties:
        """Extract comprehensive properties from the TBLite calculation.

        Computes and organizes a wide range of molecular properties from the
        GFN-xTB calculation including energies, multipole moments, partial charges,
        bond orders, and molecular orbital information.

        Args:
            atoms: ASE Atoms object with TBLite calculator attached and calculation
                completed.

        Returns:
            Dictionary with four main sections:
                - "Global": System-level properties
                    - "Total Energy [eV]": Total molecular energy
                    - "Dipole Moment [D]": Dipole moment vector (x, y, z)
                    - "Quadrupole Moment [D]": Quadrupole tensor (xx, yy, zz, xy, xz, yz)
                    - "Density Matrix": Electronic density matrix
                    - "HOMO-LUMO Gap [eV]": Energy gap between frontier orbitals
                    - "HOMO Energy [eV]": Highest occupied molecular orbital energy
                    - "LUMO Energy [eV]": Lowest unoccupied molecular orbital energy
                - "Atomic": Per-atom properties
                    - "Mulliken Partial Charges [e]": Mulliken population analysis charges
                - "Bond": Pairwise bond properties
                    - "Wiberg Bond Order": List of bond orders for all atom pairs
                        Each entry: {"i": atom_index, "j": atom_index, "value": bond_order}
                - "Orbital": Molecular orbital information
                    - "Orbital Energies [eV]": Energy of each molecular orbital
                    - "Orbital Occupations": Occupation number of each orbital
                    - "Orbital Coefficients": Coefficients of molecular orbitals

        Examples:
            >>> from ase.build import molecule # doctest: +SKIP
            >>> from pymatgen.core import Molecule # doctest: +SKIP
            >>> from jfchemistry.calculators import TBLiteCalculator # doctest: +SKIP
            >>>
            >>> # Create a simple molecule (ethane) and set up calculator
            >>> mol = Molecule.from_ase_atoms(molecule("C2H6")) # doctest: +SKIP
            >>> calc = TBLiteCalculator(method="GFN2-xTB") # doctest: +SKIP
            >>> atoms = mol.to_ase_atoms() # doctest: +SKIP
            >>> atoms = calc.set_calculator(atoms, charge=0, spin_multiplicity=1) # doctest: +SKIP
            >>> props = calc.get_properties(atoms) # doctest: +SKIP
        """
        atoms.get_potential_energy()
        forces = atoms.get_forces()  # type: ignore
        props = atoms.calc._res.dict()
        energy = props["energy"] * Hartree
        dipole = props["dipole"] * Bohr
        quadrupole = props["quadrupole"] * Bohr**2
        charges = props["charges"]
        bond_orders = props["bond-orders"]
        orbital_energies = props["orbital-energies"] * Hartree
        orbital_occupations = props["orbital-occupations"]
        # Report HOMO-LUMO Gap
        homo_index = orbital_occupations.nonzero()[0][-1]
        lumo_index = homo_index + 1
        homo_energy = orbital_energies[homo_index]
        lumo_energy = orbital_energies[lumo_index]
        homo_lumo_gap = lumo_energy - homo_energy
        orbital_coefficients = props["orbital-coefficients"]
        density_matrix = props["density-matrix"]
        wbo = np.array([], dtype=float)
        atoms_i = np.array([], dtype=int)
        atoms_j = np.array([], dtype=int)
        for i in range(len(bond_orders)):
            for j in range(len(bond_orders[i])):
                if bond_orders[i][j] > BOND_ORDER_THRESHOLD:
                    wbo = np.append(wbo, bond_orders[i][j])
                    atoms_i = np.append(atoms_i, i)
                    atoms_j = np.append(atoms_j, j)
        system_properties = TBLiteSystemProperties(
            total_energy=SystemProperty(
                name="Total Energy",
                value=energy * ureg.eV,
                description="Total energy of the system",
            ),
            dipole_moment=SystemProperty(
                name="Dipole Moment",
                value=dipole * ureg.D,
                description="Dipole moment of the system",
            ),
            quadrupole_moment=SystemProperty(
                name="Quadrupole Moment",
                value=quadrupole * ureg.D * ureg.angstrom,
                description="Quadrupole moment of the system",
            ),
            density_matrix=SystemProperty(
                name="Density Matrix",
                value=density_matrix,
                description="Density matrix of the system",
            ),
            homo_lumo_gap=SystemProperty(
                name="HOMO-LUMO Gap",
                value=homo_lumo_gap * ureg.eV,
                description="HOMO-LUMO gap of the system",
            ),
            homo_energy=SystemProperty(
                name="HOMO Energy",
                value=homo_energy * ureg.eV,
                description="HOMO energy of the system",
            ),
            lumo_energy=SystemProperty(
                name="LUMO Energy",
                value=lumo_energy * ureg.eV,
                description="LUMO energy of the system",
            ),
        )
        atomic_properties = TBLiteAtomicProperties(
            tblite_partial_charges=AtomicProperty(
                name="Mulliken Partial Charges",
                value=(charges * ureg.e).tolist(),
                description="Mulliken partial charges of the system",
            ),
            tblite_forces=AtomicProperty(
                name="Forces",
                value=forces.tolist() * (ureg.eV / ureg.angstrom),
                description="Forces of the system",
            ),
        )
        bond_properties = TBLiteBondProperties(
            tblite_wiberg_bond_orders=BondProperty(
                name="Wiberg Bond Order",
                value=wbo.tolist(),
                atoms1=atoms_i.tolist(),
                atoms2=atoms_j.tolist(),
                description="Wiberg bond order of the system",
            ),
        )
        orbital_properties = TBLiteOrbitalProperties(
            tblite_orbital_energies=OrbitalProperty(
                name="Orbital Energies",
                value=orbital_energies.tolist() * ureg.eV,
                description="Orbital energies of the system",
            ),
            tblite_orbital_occupations=OrbitalProperty(
                name="Orbital Occupations",
                value=orbital_occupations.tolist(),
                description="Orbital occupations of the system",
            ),
            tblite_orbital_coefficients=OrbitalProperty(
                name="Orbital Coefficients",
                value=orbital_coefficients.tolist(),
                description="Orbital coefficients of the system",
            ),
        )
        properties = TBLiteProperties(
            system=system_properties,
            atomic=atomic_properties,
            bond=bond_properties,
            orbital=orbital_properties,
        )

        return properties
