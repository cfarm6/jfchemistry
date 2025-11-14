"""Chain generator for polymers."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core.structure import (
    Lattice,
    Molecule,
    Site,
    Structure,
)
from rdkit.Chem import (
    rdchem,
    rdDistGeom,
    rdForceFieldHelpers,
    rdmolfiles,
    rdmolops,
    rdMolTransforms,
)

from jfchemistry.base_classes import Polymer


@dataclass
class MonomerConnections:
    """Monomer Connections."""

    connecting_points: list[int]
    connected_atoms: list[int]
    connection_strings: list[str]


def fetch_connection_points_and_atoms(m: rdchem.Mol, smiles) -> MonomerConnections:
    """Fetch the connection points and atoms from the monomer."""
    pattern = r"\[\*:\d+\]"
    connection_points_strings = re.findall(pattern, smiles)
    connection_points = []
    atom: rdchem.Atom
    bond: rdchem.Bond
    for atom in m.GetAtoms():
        if atom.GetSymbol() == "*":
            connection_points.append(atom.GetIdx())
    connected_atoms = []

    for idx in connection_points:
        for bond in m.GetBonds():
            starting_atom = bond.GetBeginAtomIdx() == idx
            ending_atom = bond.GetEndAtomIdx() == idx
            if starting_atom or ending_atom:
                other_idx = bond.GetOtherAtomIdx(idx)
                if other_idx not in connected_atoms:
                    connected_atoms.append(other_idx)
    return MonomerConnections(
        connected_atoms=connected_atoms,
        connecting_points=connection_points,
        connection_strings=connection_points_strings,
    )


def optimize_constrained_dihedral(  # noqa: PLR0913
    monomer: rdchem.Mol,
    conf_id: int,
    atom_i: int,
    atom_j: int,
    atom_k: int,
    atom_l: int,
    rotation_angle: float,
) -> tuple[float, float]:
    """Optimize a constrained dihedral angle."""
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer, confId=conf_id)
    ff.UFFAddTorsionConstraint(
        atom_i,
        atom_j,
        atom_k,
        atom_l,
        False,
        rotation_angle,
        rotation_angle,
        1000,
    )
    ff.Minimize()
    dihedral_angle = rdMolTransforms.GetDihedralDeg(
        monomer.GetConformer(conf_id),
        atom_i,
        atom_j,
        atom_k,
        atom_l,
    )
    return ff.CalcEnergy(), dihedral_angle


def infinite_chain_generator(  # noqa: PLR0913, PLR0915
    monomer: rdchem.Mol,
    chain_length: int,
    rotation_angle: list[float],
    number_conformers: int,
    interchain_distance: float,
    remove_cap_sites: bool = True,
) -> Structure:
    """Generate a chain of monomers.

    Args:
        monomer: The monomer to generate a chain of.
        chain_length: The length of the chain.
        rotation_angle: The angle to rotate the chain by.
        number_conformers: The number of conformers to generate.
        interchain_distance: The distance between the chains.
        remove_cap_sites: Whether to remove the cap sites.

    Returns:
        A chain of monomers.
    """
    monomer_no_H = rdmolops.RemoveHs(monomer)
    monomer_no_H_smiles = rdmolfiles.MolToSmiles(monomer_no_H)
    monomer_no_H_clean = rdmolfiles.MolFromSmiles(monomer_no_H_smiles)
    initial_monomer_connection = fetch_connection_points_and_atoms(
        monomer_no_H_clean, monomer_no_H_smiles
    )
    # dimerized = False
    if len(initial_monomer_connection.connected_atoms) == 1:
        _smiles_1 = monomer_no_H_smiles.replace(
            initial_monomer_connection.connection_strings[1], "[10*]"
        )
        _smiles_2 = monomer_no_H_smiles.replace(
            initial_monomer_connection.connection_strings[0], "[10*]"
        )
        _params = rdmolops.MolzipParams()
        _params.label = rdmolops.MolzipLabel.Isotope
        _smiles = ".".join([_smiles_1, _smiles_2])
        dimer_mol = rdmolfiles.MolFromSmiles(_smiles)
        dimer_connection = rdmolops.molzip(dimer_mol, _params)
        monomer_no_H_smiles = rdmolfiles.MolToSmiles(dimer_connection)
        monomer_no_H_clean = dimer_connection
        # dimerized = True
    monomer_connection = fetch_connection_points_and_atoms(monomer_no_H_clean, monomer_no_H_smiles)
    cap_1 = monomer_no_H_clean.GetAtomWithIdx(monomer_connection.connected_atoms[1]).GetSymbol()
    cap_2 = monomer_no_H_clean.GetAtomWithIdx(monomer_connection.connected_atoms[0]).GetSymbol()

    monomer_capped_smiles = monomer_no_H_smiles.replace(
        "[*:0]", f"[{cap_1}]" if len(cap_1) > 1 else cap_1
    ).replace("[*:1]", f"[{cap_2}]" if len(cap_2) > 1 else cap_2)
    monomer_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer_capped_smiles))

    rdDistGeom.EmbedMultipleConfs(monomer_capped_mol, numConfs=number_conformers)
    energies_dihedrals = [
        optimize_constrained_dihedral(
            monomer_capped_mol,
            confId,
            monomer_connection.connecting_points[0],
            monomer_connection.connected_atoms[0],
            monomer_connection.connected_atoms[1],
            monomer_connection.connecting_points[1],
            180,
            # rotation_angle if not dimerized else rotation_angle * 2,
        )
        for confId in range(monomer_capped_mol.GetNumConformers())
    ]

    df = pd.DataFrame(
        {
            "confId": range(monomer_capped_mol.GetNumConformers()),
            "energy": [e[0] for e in energies_dihedrals],
            "dihedral_angle": [np.abs(e[1]) for e in energies_dihedrals],
        }
    )
    df = df.sort_values(by="energy")
    lowest_energy_conf_id: int = int(df.iloc[0]["confId"])
    conformer = monomer_capped_mol.GetConformer(lowest_energy_conf_id)
    sites = [
        Site(atom.GetSymbol(), conformer.GetAtomPosition(atom.GetIdx()))
        for atom in monomer_capped_mol.GetAtoms()
    ]

    molecule = Molecule.from_sites(sites)
    extra_atoms = [
        i
        for i in range(monomer_capped_mol.GetNumAtoms())
        if (
            monomer_capped_mol.GetBondBetweenAtoms(i, monomer_connection.connecting_points[0])
            is not None
            or monomer_capped_mol.GetBondBetweenAtoms(i, monomer_connection.connecting_points[1])
            is not None
        )
        and monomer_capped_mol.GetAtomWithIdx(i).GetSymbol() == "H"
    ]
    molecule.remove_sites(extra_atoms)
    chain = molecule.to_ase_atoms()
    chain.translate(-chain[monomer_connection.connecting_points[0]].position)
    chain.rotate(chain[monomer_connection.connected_atoms[1]].position, [0, 0, 1])
    atoms_per_monomer = len(chain)
    monomer = chain.copy()

    for i in range(chain_length - 1):
        _monomer = monomer.copy()
        # Align Chain-A1 with Monomer-CP0
        c_a1 = chain[monomer_connection.connected_atoms[1] + i * atoms_per_monomer]
        _monomer.translate(c_a1.position)
        # Align Chain-CP1 with Monomer-A0
        c_cp1 = chain[monomer_connection.connecting_points[1] + i * atoms_per_monomer].position
        m_cp0 = _monomer[monomer_connection.connecting_points[0]].position
        m_a0 = _monomer[monomer_connection.connected_atoms[0]].position
        _monomer.rotate(m_a0 - m_cp0, c_cp1 - m_cp0, center=m_cp0)
        chain += _monomer
        print("New Length: ", len(chain))
        print("Start Index: ", (i + 1) * atoms_per_monomer)
        print("Stop Length: ", (i + 2) * atoms_per_monomer)
        chain.set_dihedral(
            monomer_connection.connected_atoms[0] + i * atoms_per_monomer,  # C-A0
            monomer_connection.connected_atoms[1] + i * atoms_per_monomer,  # C-A1
            monomer_connection.connected_atoms[0] + (i + 1) * atoms_per_monomer,  # C-A1
            monomer_connection.connected_atoms[1] + (i + 1) * atoms_per_monomer,  # C-A1
            rotation_angle[i] - 180,
            indices=range((i + 1) * atoms_per_monomer, (i + 2) * atoms_per_monomer),
        )

    # Move Chain-A0 to origin
    c_a0 = chain[monomer_connection.connected_atoms[0]].position
    chain.translate(-c_a0)

    # Align CP-1 with A0 on the z-axis
    c_cp1 = chain[
        monomer_connection.connected_atoms[1] + (chain_length - 1) * atoms_per_monomer
    ].position
    chain.rotate(c_cp1, [0, 0, 1])
    chain = Molecule.from_ase_atoms(chain)

    # Build Latticecap
    coords = chain.cart_coords
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])  # Max-min of chain
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])  # Max-min of chain

    # O->CP1 length
    z_max = chain[
        monomer_connection.connecting_points[1] + (chain_length - 1) * atoms_per_monomer
    ].coords[2]

    # Build a,b,c lengths
    a = x_max - x_min + interchain_distance * 2
    b = y_max - y_min + interchain_distance * 2
    c = z_max
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
    structure = Structure(lattice=lattice, species=[], coords=[])
    for site in chain.sites:
        structure.append(
            species=site.species,
            coords=site.coords,
            coords_are_cartesian=True,
        )

    sites_to_remove = []
    for i in range(chain_length):
        sites_to_remove.append(monomer_connection.connecting_points[0] + atoms_per_monomer * i)
        sites_to_remove.append(monomer_connection.connecting_points[1] + atoms_per_monomer * i)

    # Monomer 0
    m0_cp0 = chain[monomer_connection.connecting_points[0]]
    m0_a0 = chain[monomer_connection.connected_atoms[0]]
    m0_a1 = chain[monomer_connection.connected_atoms[1]]
    m0_cp1 = chain[monomer_connection.connecting_points[1]]

    # Monomer N
    mn_cp0 = chain[monomer_connection.connecting_points[0] + atoms_per_monomer * (chain_length - 1)]
    mn_a0 = chain[monomer_connection.connected_atoms[0] + atoms_per_monomer * (chain_length - 1)]
    mn_a1 = chain[monomer_connection.connected_atoms[1] + atoms_per_monomer * (chain_length - 1)]
    mn_cp1 = chain[monomer_connection.connecting_points[1] + atoms_per_monomer * (chain_length - 1)]

    sites_to_remove = []
    for i in range(chain_length):
        sites_to_remove.append(monomer_connection.connecting_points[0] + atoms_per_monomer * i)
        sites_to_remove.append(monomer_connection.connecting_points[1] + atoms_per_monomer * i)
    ## Append A0 and A1
    sites_to_remove.append(monomer_connection.connected_atoms[0])
    sites_to_remove.append(monomer_connection.connected_atoms[1])
    sites_to_remove.append(
        monomer_connection.connected_atoms[0] + atoms_per_monomer * (chain_length - 1)
    )
    sites_to_remove.append(
        monomer_connection.connected_atoms[1] + atoms_per_monomer * (chain_length - 1)
    )
    structure.remove_sites(sites_to_remove)
    # Append A1 to the end of the structure
    # Append mn-CP0
    if not remove_cap_sites:
        structure.append(species=mn_cp0.species, coords=mn_cp0.coords, coords_are_cartesian=True)
    # Append mn-A0
    structure.append(species=mn_a0.species, coords=mn_a0.coords, coords_are_cartesian=True)
    # Append n-A1
    structure.append(species=mn_a1.species, coords=mn_a1.coords, coords_are_cartesian=True)
    # Append mn-CP1
    if not remove_cap_sites:
        structure.append(species=mn_cp1.species, coords=mn_cp1.coords, coords_are_cartesian=True)
    # Reverse the structures
    structure.reverse()
    # Append m0-CP1
    if not remove_cap_sites:
        structure.append(species=m0_cp1.species, coords=m0_cp1.coords, coords_are_cartesian=True)
    # Append m0-A1
    structure.append(species=m0_a1.species, coords=m0_a1.coords, coords_are_cartesian=True)
    # Append m0-A0
    structure.append(species=m0_a0.species, coords=m0_a0.coords, coords_are_cartesian=True)
    # Append m0-CP0
    if not remove_cap_sites:
        structure.append(species=m0_cp0.species, coords=m0_cp0.coords, coords_are_cartesian=True)
    # Return structure to original order
    structure.reverse()
    structure.translate_sites(
        indices=range(len(structure)), vector=[0.5, 0.5, 0], to_unit_cell=False
    )
    # chain = chain.to_ase_atoms()
    return structure


def finite_chain_generator(  # noqa: PLR0913, PLR0915
    polymer: Polymer,
    chain_length: int,
    rotation_angles: list[float],
    head_angle: float,
    tail_angle: float,
    number_conformers: int,
) -> Molecule:
    """Attach the head and tail to the chain."""
    monomer_no_H = rdmolops.RemoveHs(polymer.monomer)
    monomer_no_H_smiles = rdmolfiles.MolToSmiles(monomer_no_H)
    monomer_no_H_clean = rdmolfiles.MolFromSmiles(monomer_no_H_smiles)
    initial_monomer_connection = fetch_connection_points_and_atoms(
        monomer_no_H_clean, monomer_no_H_smiles
    )
    # dimerized = False
    if len(initial_monomer_connection.connected_atoms) == 1:
        _smiles_1 = monomer_no_H_smiles.replace(
            initial_monomer_connection.connection_strings[0], "[10*]"
        )
        _smiles_2 = monomer_no_H_smiles.replace(
            initial_monomer_connection.connection_strings[1], "[10*]"
        )
        _params = rdmolops.MolzipParams()
        _params.label = rdmolops.MolzipLabel.Isotope
        _smiles = ".".join([_smiles_1, _smiles_2])
        dimer_mol = rdmolfiles.MolFromSmiles(_smiles)
        dimer_connection = rdmolops.molzip(dimer_mol, _params)
        monomer_no_H_smiles = rdmolfiles.MolToSmiles(dimer_connection)
        monomer_no_H_clean = dimer_connection
    monomer_connection = fetch_connection_points_and_atoms(monomer_no_H_clean, monomer_no_H_smiles)
    cap_0 = monomer_no_H_clean.GetAtomWithIdx(monomer_connection.connected_atoms[1]).GetSymbol()
    cap_1 = monomer_no_H_clean.GetAtomWithIdx(monomer_connection.connected_atoms[0]).GetSymbol()
    monomer_capped_smiles = monomer_no_H_smiles.replace(
        initial_monomer_connection.connection_strings[0], f"[{cap_0}]" if len(cap_0) > 1 else cap_0
    ).replace(
        initial_monomer_connection.connection_strings[1], f"[{cap_1}]" if len(cap_1) > 1 else cap_1
    )
    monomer_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer_capped_smiles))
    rdDistGeom.EmbedMultipleConfs(monomer_capped_mol, numConfs=number_conformers)
    energies_dihedrals = [
        optimize_constrained_dihedral(
            monomer_capped_mol,
            confId,
            monomer_connection.connecting_points[0],
            monomer_connection.connected_atoms[0],
            monomer_connection.connected_atoms[1],
            monomer_connection.connecting_points[1],
            180,
            # rotation_angle if not dimerized else rotation_angle * 2,
        )
        for confId in range(monomer_capped_mol.GetNumConformers())
    ]
    df = pd.DataFrame(
        {
            "confId": range(monomer_capped_mol.GetNumConformers()),
            "energy": [e[0] for e in energies_dihedrals],
            "dihedral_angle": [np.abs(e[1]) for e in energies_dihedrals],
        }
    )
    df = df.sort_values(by="energy")
    lowest_energy_conf_id: int = int(df.iloc[0]["confId"])
    conformer = monomer_capped_mol.GetConformer(lowest_energy_conf_id)
    sites = [
        Site(atom.GetSymbol(), conformer.GetAtomPosition(atom.GetIdx()))
        for atom in monomer_capped_mol.GetAtoms()
    ]
    molecule = Molecule.from_sites(sites)
    extra_atoms = [
        i
        for i in range(monomer_capped_mol.GetNumAtoms())
        if (
            monomer_capped_mol.GetBondBetweenAtoms(i, monomer_connection.connecting_points[0])
            is not None
            or monomer_capped_mol.GetBondBetweenAtoms(i, monomer_connection.connecting_points[1])
            is not None
        )
        and monomer_capped_mol.GetAtomWithIdx(i).GetSymbol() == "H"
    ]
    molecule.remove_sites(extra_atoms)
    chain = molecule.to_ase_atoms()
    chain.translate(-chain[monomer_connection.connecting_points[0]].position)
    chain.rotate(chain[monomer_connection.connected_atoms[1]].position, [0, 0, 1])
    atoms_per_monomer = len(chain)
    monomer = chain.copy()
    for i in range(chain_length - 1):
        _monomer = monomer.copy()
        # Align Chain-A1 with Monomer-CP0
        c_a1 = chain[monomer_connection.connected_atoms[1] + i * atoms_per_monomer]
        _monomer.translate(c_a1.position)
        # Align Chain-CP1 with Monomer-A0
        c_cp1 = chain[monomer_connection.connecting_points[1] + i * atoms_per_monomer].position
        m_cp0 = _monomer[monomer_connection.connecting_points[0]].position
        m_a0 = _monomer[monomer_connection.connected_atoms[0]].position
        _monomer.rotate(m_a0 - m_cp0, c_cp1 - m_cp0, center=m_cp0)
        chain += _monomer
        chain.set_dihedral(
            monomer_connection.connected_atoms[0] + i * atoms_per_monomer,  # C-A0
            monomer_connection.connected_atoms[1] + i * atoms_per_monomer,  # C-A1
            monomer_connection.connected_atoms[0] + (i + 1) * atoms_per_monomer,  # C-A1
            monomer_connection.connected_atoms[1] + (i + 1) * atoms_per_monomer,  # C-A1
            rotation_angles[i] - 180,
            indices=range((i + 1) * atoms_per_monomer, (i + 2) * atoms_per_monomer),
        )
    # Move Chain-A0 to origin
    c_a0 = chain[monomer_connection.connected_atoms[0]].position
    chain.translate(-c_a0)
    # Align CP-1 with A0 on the z-axis
    c_cp1 = chain[
        monomer_connection.connected_atoms[1] + (chain_length - 1) * atoms_per_monomer
    ].position
    chain.rotate(c_cp1, [0, 0, 1])
    chain = Molecule.from_ase_atoms(chain)
    # # Build Lattice
    coords = chain.cart_coords
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])  # Max-min of chain
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])  # Max-min of chain
    # # O->CP1 length
    z_max = chain[
        monomer_connection.connecting_points[1] + (chain_length - 1) * atoms_per_monomer
    ].coords[2]
    # Build a,b,c lengths
    a = x_max - x_min + 0.001 * 2
    b = y_max - y_min + 0.001 * 2
    c = z_max
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
    structure = Structure(lattice=lattice, species=[], coords=[])
    for site in chain.sites:
        structure.append(
            species=site.species,
            coords=site.coords,
            coords_are_cartesian=True,
        )

    # Monomer 0
    m0_cp0 = chain[monomer_connection.connecting_points[0]]
    m0_a0 = chain[monomer_connection.connected_atoms[0]]
    m0_a1 = chain[monomer_connection.connected_atoms[1]]
    m0_cp1 = chain[monomer_connection.connecting_points[1]]

    # Monomer N
    mn_cp0 = chain[monomer_connection.connecting_points[0] + atoms_per_monomer * (chain_length - 1)]
    mn_a0 = chain[monomer_connection.connected_atoms[0] + atoms_per_monomer * (chain_length - 1)]
    mn_a1 = chain[monomer_connection.connected_atoms[1] + atoms_per_monomer * (chain_length - 1)]
    mn_cp1 = chain[monomer_connection.connecting_points[1] + atoms_per_monomer * (chain_length - 1)]

    sites_to_remove = []
    for i in range(chain_length):
        sites_to_remove.append(monomer_connection.connecting_points[0] + atoms_per_monomer * i)
        sites_to_remove.append(monomer_connection.connecting_points[1] + atoms_per_monomer * i)
    ## Append A0 and A1
    sites_to_remove.append(monomer_connection.connected_atoms[0])
    sites_to_remove.append(monomer_connection.connected_atoms[1])
    sites_to_remove.append(
        monomer_connection.connected_atoms[0] + atoms_per_monomer * (chain_length - 1)
    )
    sites_to_remove.append(
        monomer_connection.connected_atoms[1] + atoms_per_monomer * (chain_length - 1)
    )
    structure.remove_sites(sites_to_remove)
    # Append A1 to the end of the structure
    # Append mn-CP0
    structure.append(species=mn_cp0.species, coords=mn_cp0.coords, coords_are_cartesian=True)
    # Append mn-A0
    structure.append(species=mn_a0.species, coords=mn_a0.coords, coords_are_cartesian=True)
    # Append n-A1
    structure.append(species=mn_a1.species, coords=mn_a1.coords, coords_are_cartesian=True)
    # Append mn-CP1
    structure.append(species=mn_cp1.species, coords=mn_cp1.coords, coords_are_cartesian=True)
    # Reverse the structures
    structure.reverse()
    # Append m0-CP1
    structure.append(species=m0_cp1.species, coords=m0_cp1.coords, coords_are_cartesian=True)
    # Append m0-A1
    structure.append(species=m0_a1.species, coords=m0_a1.coords, coords_are_cartesian=True)
    # Append m0-A0
    structure.append(species=m0_a0.species, coords=m0_a0.coords, coords_are_cartesian=True)
    # Append m0-CP0
    structure.append(species=m0_cp0.species, coords=m0_cp0.coords, coords_are_cartesian=True)
    # Return structure to original order
    structure.reverse()

    head_no_H = rdmolops.RemoveHs(polymer.head)
    head_no_H_smiles = rdmolfiles.MolToSmiles(head_no_H)
    head_no_H_clean = rdmolfiles.MolFromSmiles(head_no_H_smiles)
    head_connection = fetch_connection_points_and_atoms(head_no_H_clean, head_no_H_smiles)
    head_capped_smiles = head_no_H_smiles.replace(
        head_connection.connection_strings[0], f"[{cap_1}]" if len(cap_1) > 1 else cap_1
    )

    head_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(head_capped_smiles))
    rdDistGeom.EmbedMultipleConfs(head_capped_mol)

    head_conformer = head_capped_mol.GetConformer()
    head_sites = [
        Site(atom.GetSymbol(), head_conformer.GetAtomPosition(atom.GetIdx()))
        for atom in head_capped_mol.GetAtoms()
    ]
    head_molecule = Molecule.from_sites(head_sites)
    extra_head_atoms = [
        i
        for i in range(head_capped_mol.GetNumAtoms())
        if (
            head_capped_mol.GetBondBetweenAtoms(i, head_connection.connecting_points[0]) is not None
        )
        and head_capped_mol.GetAtomWithIdx(i).GetSymbol() == "H"
        and i not in head_connection.connected_atoms
    ]
    head_molecule.remove_sites(extra_head_atoms)
    head = head_molecule.to_ase_atoms()
    # Move Head-CP-0 to Chain-A0
    head_cp0 = head[head_connection.connecting_points[0]].position
    chain_a0 = structure[1].coords
    head.translate(chain_a0 - head_cp0)
    head_cp0 = head[head_connection.connecting_points[0]].position
    # Align Head-A0 with Chain-CP-0
    chain_cp0 = structure[0].coords
    chain_a0 = structure[1].coords
    head_a0 = head[head_connection.connected_atoms[0]].position
    head.rotate(a=head_a0 - chain_a0, v=chain_cp0 - chain_a0, center=chain_a0)
    head_chain = structure.to_ase_atoms() + head
    head_chain.set_dihedral(
        3,  # C-CP1-monomer-1
        2,  # C-A1-monomer-1
        len(structure) + head_connection.connecting_points[0],  # H-CP0-monomer-1
        len(structure) + head_connection.connected_atoms[0],  # H-A0
        head_angle,
        indices=range(len(structure), len(head_chain)),
    )
    head = head_chain[len(structure) :]
    head = Molecule.from_ase_atoms(head)
    head.remove_sites(
        [
            head_connection.connecting_points[0],  # H-CP0
        ]
    )

    tail_no_H = rdmolops.RemoveHs(polymer.tail)
    tail_no_H_smiles = rdmolfiles.MolToSmiles(tail_no_H)
    tail_no_H_clean = rdmolfiles.MolFromSmiles(tail_no_H_smiles)
    tail_connection = fetch_connection_points_and_atoms(tail_no_H_clean, tail_no_H_smiles)
    tail_capped_smiles = tail_no_H_smiles.replace(
        tail_connection.connection_strings[0], f"[{cap_0}]" if len(cap_0) > 1 else cap_0
    )

    tail_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(tail_capped_smiles))
    rdDistGeom.EmbedMultipleConfs(tail_capped_mol)

    tail_conformer = tail_capped_mol.GetConformer()
    tail_sites = [
        Site(atom.GetSymbol(), tail_conformer.GetAtomPosition(atom.GetIdx()))
        for atom in tail_capped_mol.GetAtoms()
    ]
    tail_molecule = Molecule.from_sites(tail_sites)
    extra_tail_atoms = [
        i
        for i in range(tail_capped_mol.GetNumAtoms())
        if (
            tail_capped_mol.GetBondBetweenAtoms(i, tail_connection.connecting_points[0]) is not None
        )
        and tail_capped_mol.GetAtomWithIdx(i).GetSymbol() == "H"
        and i not in tail_connection.connected_atoms
    ]
    tail_molecule.remove_sites(extra_tail_atoms)
    tail = tail_molecule.to_ase_atoms()
    # Move tail-CP-0 to Chain-A1
    tail_cp0 = tail[tail_connection.connecting_points[0]].position
    chain_a1 = structure[-2].coords
    tail.translate(chain_a1 - tail_cp0)
    tail_cp0 = tail[tail_connection.connecting_points[0]].position
    # Align tail-A0 with Chain-CP-1
    chain_cp1 = structure[-1].coords
    chain_a1 = structure[-2].coords
    tail_a0 = tail[tail_connection.connected_atoms[0]].position
    tail.rotate(a=tail_a0 - chain_a1, v=chain_cp1 - chain_a1, center=chain_a1)
    tail_chain = tail + structure.to_ase_atoms()
    tail_chain.set_dihedral(
        -4,  # C-CP0-monomer-1
        -3,  # C-A0-monomer-1
        -2,  # C-A1-monomer-1
        tail_connection.connected_atoms[0],  # T-A0
        tail_angle,
        indices=range(len(structure), len(tail_chain)),
    )
    tail = tail_chain[: len(tail)]
    tail = Molecule.from_ase_atoms(tail)
    tail.remove_sites(
        [
            tail_connection.connecting_points[0],  # H-CP0
        ]
    )
    structure.remove_sites([0, 3, len(structure) - 1, len(structure) - 4])

    return Molecule(
        coords=[*head.cart_coords, *structure.cart_coords, *tail.cart_coords],
        species=[*head.species, *structure.species, *tail.species],
    )
