
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core.structure import Molecule, Site
from rdkit.Chem import (
    AllChem,
    rdchem,
    rdDetermineBonds,
    rdDistGeom,
    rdForceFieldHelpers,
    rdmolfiles,
    rdmolops,
    rdMolTransforms,
)

from jfchemistry.core.structures import Polymer


@dataclass
class MonomerAtomCap:
    atom_type: str
    atom_index: int
    connection_index: int | list[int]
    connected_atom_index: int | list[int]


def pymatgen_to_rdkit_with_bonds(pmg_mol: Molecule, charge: int = 0) -> rdchem.Mol:
    """Convert a pymatgen Molecule to an RDKit Mol object with bonding information.

    This function uses RDKit's bond perception algorithm to determine connectivity
    from 3D coordinates.

    Args:
        pmg_mol: pymatgen Molecule object
        charge: Total charge of the molecule (default: 0)

    Returns:
        RDKit Mol object with bonds determined from geometry

    Examples:
        >>> from pymatgen.core import Molecule
        >>> pmg_mol = Molecule(["C", "H", "H", "H", "H"],
        ...                    [[0, 0, 0], [0.63, 0.63, 0.63],
        ...                     [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
        ...                     [0.63, -0.63, -0.63]])
        >>> rdkit_mol = pymatgen_to_rdkit_with_bonds(pmg_mol)
        >>> rdkit_mol.GetNumBonds()
        4
    """
    # Create an editable RDKit molecule
    rd_mol = rdchem.RWMol()

    # Add atoms
    for site in pmg_mol.sites:
        atom = rdchem.Atom(site.species_string)
        rd_mol.AddAtom(atom)

    # Add a conformer with 3D coordinates
    conf = rdchem.Conformer(len(pmg_mol))
    for i, site in enumerate(pmg_mol.sites):
        conf.SetAtomPosition(i, site.coords)

    rd_mol.AddConformer(conf)

    # Convert to regular Mol (needed for rdDetermineBonds)
    rd_mol = rd_mol.GetMol()

    # Determine bonds from 3D coordinates
    try:
        rdDetermineBonds.DetermineBonds(rd_mol, useVdw=True)
    except Exception as e:
        print(f"Warning: Bond determination failed with error: {e}")
        print("Returning molecule without bonds.")
        return rd_mol
    return rd_mol


def get_atom_caps(m: rdchem.Mol) -> tuple[list[MonomerAtomCap], bool]:
    smiles = rdmolfiles.MolToSmiles(m)
    pattern = r"\[\*:\d+\]"
    connection_points_strings = re.findall(pattern, smiles)
    numbers = [re.findall(r"\d+", s)[0] for s in connection_points_strings]
    connected_indices: list[int] = []
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            connected_indices.append(atom.GetIdx())
    A = np.array(rdmolops.GetAdjacencyMatrix(m))
    symbols = np.array([x.GetSymbol() for x in m.GetAtoms()])
    atomCaps: list[MonomerAtomCap] = []
    for index, connection_index in zip(connected_indices, numbers):
        atom_type = str(symbols[A[index, :] >= 1][0])
        if len(atom_type) != 1:
            atom_type = f"[{atom_type}]"
        atom_index = int(np.argwhere(A[index, :] >= 1)[0, 0])
        connected_atom_index = index
        connection_index = int(connection_index)
        atomCaps.append(
            MonomerAtomCap(atom_type, atom_index, connection_index, connected_atom_index)
        )
    dimer = all(obj.atom_index == atomCaps[0].atom_index for obj in atomCaps)
    atomCaps.sort(key=lambda x: x.connection_index)

    return atomCaps, dimer


def generate_monomer(
    m: rdchem.Mol, monomer_dihedral: float, dihedral_cutoff: float, number_conformers: int
) -> tuple[Molecule, bool]:
    """Generate a capped monomer for creating a polymer chain.
    For a monomer, `m`, the unit is characterized by the following points:
    C0 - A0 - R - A1 - C1
    or
    C0 - A0 - C1
         |
         R
    Where C0 and C1 are the connection points, and A0 and A1 are the atoms that are the head and tail connections for the monomer. For a repeating chain, the atom type of C0 = A1, and C1 = A0,
    """
    # Canonicalize the SMILES for the molecule
    m = rdmolops.RemoveHs(m)
    m_smiles = rdmolfiles.MolToSmiles(m)
    m = rdmolfiles.MolFromSmiles(m_smiles)
    # Get the monomer connections
    atom_caps, dimer = get_atom_caps(m)
    head_cap, tail_cap = atom_caps
    # Replace First Atom Cap Type:
    m_capped_smiles = m_smiles.replace(
        f"[*:{head_cap.connection_index}]", tail_cap.atom_type
    ).replace(f"[*:{tail_cap.connection_index}]", head_cap.atom_type)

    monomer_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(m_capped_smiles))
    # Embed conformers for the capped monomer
    rdDistGeom.EmbedMultipleConfs(monomer_capped_mol, numConfs=number_conformers)
    # Get the connecting dihedral
    if not dimer:
        atom_i = atom_caps[0].connected_atom_index
        atom_j = atom_caps[0].atom_index
        atom_k = atom_caps[1].atom_index
        atom_l = atom_caps[1].connected_atom_index
        # atom_i, atom_l = connections.connecting_points
        # atom_j, atom_k = connections.connected_atoms
    # Setup the forcefield
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer_capped_mol)
    # Apply dihedral constraints if applicable
    if not dimer and monomer_dihedral:
        ff.UFFAddTorsionConstraint(
            atom_i,
            atom_j,
            atom_k,
            atom_l,
            False,
            monomer_dihedral,
            monomer_dihedral,
            1000,
        )
    # Optimize the molecule
    results = rdForceFieldHelpers.OptimizeMoleculeConfs(mol=monomer_capped_mol, ff=ff)
    # retrieve the energies
    energies = np.array([r[1] for r in results])
    # create a dataframe
    df = pd.DataFrame({"confId": range(monomer_capped_mol.GetNumConformers()), "energy": energies})
    # If dihedrals are constrained, then filter by the dihedrals
    if not dimer:
        dihedrals = np.array(
            [
                rdMolTransforms.GetDihedralDeg(conf, atom_i, atom_j, atom_k, atom_l)
                for conf in monomer_capped_mol.GetConformers()
            ]
        )
        dihedrals = np.abs(dihedrals)
        df["dihedral"] = np.max(dihedrals) - dihedrals
        df = df[df["dihedral"] < dihedral_cutoff]
    # Find the minimum energy structure
    df.sort_values(by=["energy"], inplace=True)
    # Get the conformer
    confId = int(df["confId"].values[0])
    conformer = monomer_capped_mol.GetConformer(confId)
    # Convert to pymatgen molecule
    sites = np.array(
        [
            Site(atom.GetSymbol(), conformer.GetAtomPosition(atom.GetIdx()))
            for atom in monomer_capped_mol.GetAtoms()
        ]
    )
    # molecule = Molecule.from_sites(sites[order])
    # Remove extra hydrogens connected to capped atoms.
    monomer_atoms = np.array(monomer_capped_mol.GetAtoms())
    A = rdmolops.GetAdjacencyMatrix(monomer_capped_mol)
    extra_Hs = []
    for ac in atom_caps:
        connected_atoms = monomer_atoms[A[ac.connected_atom_index] >= 1]
        for atom in connected_atoms:
            if atom.GetSymbol() == "H":
                extra_Hs.append(atom.GetIdx())
    sites = np.delete(sites, extra_Hs)
    # Reorder the sites so that C0, A0, ..., A1, C1
    indices = np.array(list(range(len(sites))))
    if dimer:
        idx1 = atom_caps[1].connected_atom_index
        idx3 = atom_caps[0].atom_index
        idx4 = atom_caps[0].connected_atom_index
        indices = list(np.delete(indices, [idx1, idx3, idx4]))
        order = [idx4, idx3] + indices + [idx1]
    else:
        idx1 = atom_caps[1].connected_atom_index
        idx2 = atom_caps[1].atom_index
        idx3 = atom_caps[0].atom_index
        idx4 = atom_caps[0].connected_atom_index
        indices = list(np.delete(indices, [idx1, idx2, idx3, idx4]))
        order = [idx4, idx3] + indices + [idx2, idx1]
    print(order)
    # Form Molecule
    print(sites[order])
    molecule = Molecule.from_sites(sites[order])
    # Translate C0 to the origin
    # Rotate to align A1 or A0 (dimer) with the z-axis
    _m = molecule.to_ase_atoms()
    _m.translate(-molecule.sites[0].coords)
    if dimer:
        a = molecule.sites[1].coords
    else:
        a = molecule.sites[-2].coords
    _m.rotate(a, [0, 0, 1])
    molecule = Molecule.from_ase_atoms(_m)
    return molecule, dimer


def rotate_monomer(
    last_monomer: Molecule, next_monomer: Molecule, dihedral_angle: float, dimer: bool
) -> Molecule:
    # Align New-C0 with Old-A1/A0(dimer)
    old = last_monomer.copy().to_ase_atoms()
    new = next_monomer.copy().to_ase_atoms()
    if dimer:
        connection_point = old[1].position
    else:
        connection_point = old[-2].position
    new.translate(connection_point - new[0].position)

    # Align Connection Vector
    if dimer:
        # Align Old - A0-C1 with New C0-A0
        V1 = old[1].position - old[-1].position
        V2 = new[0].position - new[1].position
        center = new[0].position
    else:
        # Align Old - A1-C1 with New C0 - A0
        V1 = old[-1].position - old[-2].position
        V2 = new[1].position - new[0].position
        center = new[0].position
    new.rotate(V2, V1, center=center)

    # Fix Dihedrals
    atoms = old + new
    if dimer:
        atom_i = 0  # Chain - C0
        atom_j = 1  # Chain - A0
        atom_k = len(old) + 1  # Monomer - A0
        atom_l = len(old) + len(new) - 1  # Monomer - C1
    else:
        atom_i = 1  # Chain - A0
        atom_j = len(old) - 2  # Chain - A1
        atom_k = len(old) + 1  # Monomer - A0
        atom_l = len(old) + len(new) - 2  # Monomer - A1

    indices = range(len(old), len(atoms))

    atoms.set_dihedral(
        atom_i,
        atom_j,
        atom_k,
        atom_l,
        dihedral_angle if dimer else dihedral_angle - 180,
        indices=range(len(new), len(atoms)),
    )

    return Molecule.from_ase_atoms(atoms[len(new) :])


def build_chain(
    monomer: Molecule, dihedral_angle: float | list[float], dimer: bool, chain_length: int
) -> Molecule:
    monomer_start = monomer.copy()
    monomer_list = [monomer.copy()]
    if isinstance(dihedral_angle, list):
        assert len(dihedral_angle) == chain_length
    else:
        dihedral_angle = [dihedral_angle] * chain_length
    for i in range(chain_length - 1):
        next_monomer = rotate_monomer(monomer_list[-1], monomer_start, dihedral_angle[i], dimer)
        monomer_list.append(next_monomer)

    atoms_list = [mol for mol in monomer_list]
    # atoms = Atoms()
    # for mer in atoms_list:
    #     atoms += mer
    # chain = Molecule.from_ase_atoms(atoms)
    return atoms_list


def add_cap(p: Polymer, last_monomer: Molecule, head: bool = True):
    # 1. Standardize head unit and get the atom caps
    if head:
        h = rdmolops.RemoveHs(p.head)
    else:
        h = rdmolops.RemoveHs(p.tail)
    ## Get the unit connections
    atom_caps, dimer = get_atom_caps(h)
    [h_head_cap] = atom_caps
    h = rdmolops.AddHs(h)
    # 2. Repeat for Monomer Unit
    m = rdmolops.RemoveHs(p.monomer)
    m = rdmolops.AddHs(m)
    m_smiles = rdmolfiles.MolToSmiles(m)
    atom_caps, dimer = get_atom_caps(m)
    m_head_cap, m_tail_cap = atom_caps
    # Update the monomer graph
    if head:
        m_smiles = m_smiles.replace(f"[*:{m_tail_cap.connection_index}]", m_head_cap.atom_type)
    else:
        m_smiles = m_smiles.replace(f"[*:{m_head_cap.connection_index}]", m_tail_cap.atom_type)

    m = rdmolfiles.MolFromSmiles(m_smiles, sanitize=False)
    # 3. Update Head smiles to have the correct end_cap
    hm = rdmolops.molzip(h, m)
    # 4. Create the monomer standard graph w/ caps
    if head:
        m_core_smiles = m_smiles.replace(f"[*:{m_head_cap.connection_index}]", m_tail_cap.atom_type)
    else:
        m_core_smiles = m_smiles.replace(f"[*:{m_tail_cap.connection_index}]", m_head_cap.atom_type)

    m_core = rdmolfiles.MolFromSmiles(m_core_smiles, sanitize=False)
    # 5. Reorder the atoms to match the pmg Molecule
    indices = list(range(m_core.GetNumAtoms()))
    if dimer:
        idx1 = atom_caps[1].connected_atom_index
        idx3 = atom_caps[0].atom_index
        idx4 = atom_caps[0].connected_atom_index
        indices = list(np.delete(indices, [idx1, idx3, idx4]))
        order = [idx4, idx3] + indices + [idx1]
    else:
        idx1 = atom_caps[1].connected_atom_index
        idx2 = atom_caps[1].atom_index
        idx3 = atom_caps[0].atom_index
        idx4 = atom_caps[0].connected_atom_index
        indices = list(np.delete(indices, [idx1, idx2, idx3, idx4]))
        order = [idx4, idx3] + indices + [idx2, idx1]
    rdmolops.RenumberAtoms(m_core, [int(i) for i in order])
    # 6. Convert the pmg Molecule to rdkit
    pmg_mol = pymatgen_to_rdkit_with_bonds(last_monomer)
    # 7. Match the substructure and add conformer with correct bonds.
    match = pmg_mol.GetSubstructMatch(m_core)
    mol_conf = rdchem.Conformer()
    pmg_conf = pmg_mol.GetConformer()
    for atom in m_core.GetAtoms():
        i = atom.GetIdx()
        mol_conf.SetAtomPosition(i, pmg_conf.GetAtomPosition(match[i]))
    m_core.AddConformer(mol_conf)
    # 8. Perform the constrained embedding
    AllChem.ConstrainedEmbed(hm, m_core, useTethers=True)
    # 9. Get the head unit match
    sites = []
    conf = hm.GetConformer()
    for _i in range(h.GetNumAtoms() - 1):
        sites.append(
            Site(coords=conf.GetAtomPosition(_i), species=hm.GetAtomWithIdx(_i).GetSymbol())
        )
    head_mol = Molecule.from_sites(sites)
    return head_mol


def make_finite_chain(
    p: Polymer,
    chain_length: int,
    monomer_dihedral: float,
    dihedral_cutoff: float,
    number_conformers: int,
    chain_dihedral: float | list[float],
) -> Molecule:
    M, D = generate_monomer(p.monomer, monomer_dihedral, dihedral_cutoff, number_conformers)
    C = build_chain(M, chain_dihedral, D, chain_length)

    # Add head and tail groups
    # if D:
    #     # Dimer case: pass first two monomers for head, last two for tail
    H = add_cap(p, C[0], head=True)
    #     T = add_tail(p.tail, [C[-1], C[-2]], tail_dihedral, D)
    # else:
    #     # Non-dimer case: pass single monomers
    #     H = add_head(p.head, C[0], head_dihedral, D)
    T = add_cap(p, C[-1], head=False)
    # Assemble the full chain
    # Head minus last atom (connection point)
    _c = H.to_ase_atoms()
    # Add all middle monomers (without first and last connection points)
    for mer in C:
        _c += mer.to_ase_atoms()[1:-1]
    # Tail minus last atom (connection point)
    _c += T.to_ase_atoms()

    return Molecule.from_ase_atoms(_c)
