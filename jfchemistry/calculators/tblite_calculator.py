"""Apply the AimNet2 calculator to a structure."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ase import Atoms
from ase.units import Bohr, Hartree

from .ase_calculator import ASECalculator


@dataclass
class TBLiteCalculator(ASECalculator):
    """Apply the AimNet2 calculator to a structure."""

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
        """Set the calculator for the atoms."""
        try:
            from tblite.ase import TBLite
        except ImportError as e:
            raise ImportError(
                "The 'aimnet' package is required to use AimNet2Calculator but is not available. "
                "Please install it from: https://github.com/cfarm6/aimnetcentral.git"
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
        """Return the properties of the structure."""
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
