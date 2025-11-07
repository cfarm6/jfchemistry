"""Chain generator for polymers."""

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


def fetch_dummy_index_and_bond_type(m) -> tuple[list[int], list[int]]:
    """Fetch the dummy atom index and bond type from a smiles string."""
    dummy_index = []
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == "*":
                dummy_index.append(atom.GetIdx())
    # Get the atoms connected to the dummy atoms
    connected_atoms = []
    for dummy_idx in dummy_index:
        for bond in m.GetBonds():
            if bond.GetBeginAtomIdx() == dummy_idx or bond.GetEndAtomIdx() == dummy_idx:
                idx = bond.GetOtherAtomIdx(dummy_idx)
                if idx not in connected_atoms:
                    connected_atoms.append(idx)
    return (
        dummy_index,
        connected_atoms,
    )


def process_monomer(monomer: rdchem.Mol) -> tuple[str, rdchem.Mol, bool]:
    """Process the monomer from Polymer.monomer.

    1. Fetch the connection points from the Mol object.
    2. Return the smiles string and the monomer object.
    """
    monomer = rdmolops.RemoveAllHs(monomer)
    monomer_smiles = rdmolfiles.MolToSmiles(monomer)
    monomer = rdmolfiles.MolFromSmiles(monomer_smiles)
    _, connected_atoms = fetch_dummy_index_and_bond_type(monomer)
    dimerized = False
    if len(connected_atoms) == 1:
        _smiles_1 = monomer_smiles.replace("[2*]", "[10*]")
        _smiles_2 = monomer_smiles.replace("[1*]", "[10*]")
        _params = rdmolops.MolzipParams()
        _params.label = rdmolops.MolzipLabel.Isotope
        _smiles = ".".join([_smiles_1, _smiles_2])
        dimer = rdmolfiles.MolFromSmiles(_smiles)
        dimer = rdmolops.molzip(dimer, _params)
        monomer = dimer
        monomer_smiles = rdmolfiles.MolToSmiles(monomer)
        dimerized = True
    return monomer_smiles, monomer, dimerized


def optimize_constrained_dihedral(  # noqa: PLR0913
    monomer: rdchem.Mol,
    conf_id: int,
    atom_i: int,
    atom_j: int,
    atom_k: int,
    atom_l: int,
    rotation_angle: float,
) -> float:
    """Optimize a constrained dihedral angle."""
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer, confId=conf_id)
    ff.UFFAddTorsionConstraint(
        atom_i,
        atom_j,
        atom_k,
        atom_l,
        False,
        rotation_angle * 0.9,
        rotation_angle * 1.1,
        100,
    )
    ff.Minimize()
    dihedral_angle = rdMolTransforms.GetDihedralDeg(
        monomer.GetConformer(conf_id),
        atom_i,
        atom_j,
        atom_k,
        atom_l,
    )
    print(dihedral_angle)
    return ff.CalcEnergy()


def chain_generator(  # noqa: PLR0913, PLR0915
    monomer: rdchem.Mol,
    chain_length: int,
    rotation_angle: float,
    num_conformers: int,
    dihedral_angle_cutoff: float,
    inter_chain_distance: float,
) -> Structure:
    """Generate a polymer chain."""
    monomer_smiles, monomer, dimerized = process_monomer(monomer)
    connection_points, connected_atoms = fetch_dummy_index_and_bond_type(monomer)
    connection_point_1, connection_point_2 = connection_points
    atom1, atom2 = connected_atoms
    # Replace the dummy atom with the appropriate bond type
    atom_types = {
        1: monomer.GetAtomWithIdx(atom2).GetSymbol(),
        2: monomer.GetAtomWithIdx(atom1).GetSymbol(),
    }

    for i in [1, 2]:
        monomer_smiles = monomer_smiles.replace(f"[{i}*]", atom_types[i])

    monomer = rdmolfiles.MolFromSmiles(monomer_smiles)
    num_atoms = monomer.GetNumAtoms()
    monomer = rdmolops.AddHs(monomer)
    num_H_atoms = monomer.GetNumAtoms()

    extra_atoms = [
        i
        for i in range(num_atoms, num_H_atoms)
        if monomer.GetBondBetweenAtoms(i, connection_points[0]) is not None
        or monomer.GetBondBetweenAtoms(i, connection_points[1]) is not None
    ]

    rdDistGeom.EmbedMultipleConfs(monomer, numConfs=num_conformers)

    if dimerized:
        energies = [
            optimize_constrained_dihedral(
                monomer,
                confId,
                connection_point_1,
                atom1,
                atom2,
                connection_point_2,
                rotation_angle if not dimerized else rotation_angle * 2,
            )
            for confId in range(monomer.GetNumConformers())
        ]
    else:
        results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(monomer)
        energies = [r[1] for r in results]

    dihedrals = [
        rdMolTransforms.GetDihedralDeg(
            monomer.GetConformer(i),
            connection_point_1,
            atom1,
            atom2,
            connection_point_2,
        )
        for i in range(num_conformers)
    ]
    df = pd.DataFrame(
        {
            "confId": range(num_conformers),
            "energy": energies,
            "dihedral_angle": np.abs(dihedrals),
        }
    )
    df = df.sort_values(by="dihedral_angle")
    largest_dihedral_angle: float = df.iloc[-1]["dihedral_angle"]
    df = df[df["dihedral_angle"] > largest_dihedral_angle - dihedral_angle_cutoff]
    df = df.sort_values(by="energy")
    lowest_energy_conf: int = int(df.iloc[0]["confId"])
    conformer = monomer.GetConformer(lowest_energy_conf)

    sites = [
        Site(atom.GetSymbol(), conformer.GetAtomPosition(atom.GetIdx()))
        for atom in monomer.GetAtoms()
    ]

    molecule = Molecule.from_sites(sites)
    atoms = molecule.to_ase_atoms()

    atoms.translate([0, 0, -atoms[connection_point_1].position[2]])
    atoms.rotate(atoms[connection_point_2].position, [0, 0, 1])

    molecule = Molecule.from_ase_atoms(atoms)

    molecule.remove_sites([connection_point_2, *extra_atoms])

    v_min = np.min(molecule.cart_coords, axis=0)
    v_max = np.max(molecule.cart_coords, axis=0)

    delta = v_max - v_min
    a = delta[0] + inter_chain_distance
    b = delta[1] + inter_chain_distance
    c = delta[2] + 0.00001

    structure = molecule.get_boxed_structure(a=a, b=b, c=c, reorder=False, no_cross=False)

    offset = molecule[connection_point_1].coords - structure[connection_point_1].coords
    structure.translate_sites(
        indices=range(len(structure)),
        vector=offset,
        frac_coords=False,
        to_unit_cell=False,
    )

    structure.remove_sites([connection_point_1])

    theta = rotation_angle

    if dimerized:
        theta *= 2
        copies = chain_length // 2
        _c = c * copies
        if chain_length % 2 == 1:
            _c += c * 0.5
    else:
        copies = chain_length
        _c = c * copies
    print(copies)
    print(
        c,
        _c,
    )
    lattice = Lattice.from_parameters(a=a, b=b, c=_c, alpha=90, beta=90, gamma=90)

    final_structure = Structure(species=[], coords=[], lattice=lattice)

    for i in range(copies):
        print(i)
        _structure = structure.copy()
        _structure.rotate_sites(
            theta=np.deg2rad(theta) * i,
            axis=[0, 0, 1],
            anchor=[0, 0, 0],
            to_unit_cell=False,
        )
        for site in _structure:
            final_structure.append(
                species=site.species,
                coords=site.coords + [0, 0, c * i],  # noqa: RUF005
                coords_are_cartesian=True,
            )

    if chain_length % 2 == 1:
        num_dimer_sites = len(structure)
        num_monomer_sites = num_dimer_sites // 2
        num_atoms = len(final_structure)
        final_structure.remove_sites(range(num_atoms - num_monomer_sites, num_atoms))

    final_structure.translate_sites(
        indices=range(len(final_structure)),
        vector=[a / 2, b / 2, 0],
        to_unit_cell=True,
        frac_coords=False,
    )
    return final_structure


def attach_head_and_tail(chain: Structure, head: Molecule, tail: Molecule) -> Molecule:
    """Attach the head and tail to the chain."""
    # Find the connection point of the head and tail
    return Molecule(species=[], coords=[])
