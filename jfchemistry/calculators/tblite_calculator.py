"""TBLite calculator for extended tight-binding calculations.

This module provides integration with TBLite (Tight-Binding Library) for
performing GFN-xTB semi-empirical quantum chemistry calculations.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ase import Atoms
from ase.units import Bohr, Hartree

from .ase_calculator import ASECalculator


@dataclass
class TBLiteCalculator(ASECalculator):
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
    method: Literal["GFN1-xTB", "GFN2-xTB", "IPEA1-xTB"] = "GFN2-xTB"
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    max_iterations: int = 250
    initial_guess: Literal["sad", "eeq"] = "sad"
    mixer_damping: float = 0.4
    verbosity: int = 0

    def set_calculator(self, atoms: Atoms, charge: int = 0, spin_multiplicity: int = 1) -> Atoms:
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
            from tblite.ase import TBLite
        except ImportError as e:
            raise ImportError(
                "The 'tblite' package is required to use TBLiteCalculator but is not available. "
                "Please install it with: pip install tblite or conda install tblite-python"
            ) from e
        if self.charge is not None:
            charge = self.charge
        if self.multiplicity is not None:
            spin_multiplicity = self.multiplicity
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
        )

        return atoms

    def get_properties(self, atoms: Atoms) -> dict[str, Any]:
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
        wbo = []
        for i in range(len(bond_orders)):
            for j in range(len(bond_orders[i])):
                wbo.append(
                    {
                        "i": i,
                        "j": j,
                        "value": bond_orders[i][j],
                    }
                )
        properties = {
            "Global": {
                "Total Energy [eV]": energy,
                "Dipole Moment [D]": dipole,
                "Quadrupole Moment [D]": {
                    "xx": quadrupole[0],
                    "yy": quadrupole[1],
                    "zz": quadrupole[2],
                    "xy": quadrupole[3],
                    "xz": quadrupole[4],
                    "yz": quadrupole[5],
                },
                "Density Matrix": density_matrix,
                "HOMO-LUMO Gap [eV]": homo_lumo_gap,
                "HOMO Energy [eV]": homo_energy,
                "LUMO Energy [eV]": lumo_energy,
            },
            "Atomic": {
                "Mulliken Partial Charges [e]": charges,
            },
            "Bond": {"Wiberg Bond Order": wbo},
            "Orbital": {
                "Orbital Energies [eV]": orbital_energies,
                "Orbital Occupations": orbital_occupations,
                "Orbital Coefficients": orbital_coefficients,
            },
        }

        return properties
