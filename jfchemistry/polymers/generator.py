"""Polymer Generator."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core.structure import Molecule, Site
from rdkit import Geometry
from rdkit.Chem import (
    rdchem,
    rdDistGeom,
    rdForceFieldHelpers,
    rdmolfiles,
    rdmolops,
    rdMolTransforms,
)

from jfchemistry.core.structures import Polymer


@dataclass
class MonomerAtomCap:
    """Represents a connection point (cap) in a monomer unit.

    Attributes:
        atom_type: The atomic symbol of the atom connected to the dummy atom.
        atom_index: The index of the atom connected to the dummy atom.
        connection_index: The connection point label (e.g., from SMILES [*:1], [*:2]).
        connected_atom_index: The index of the dummy atom (atomic number 0).
    """

    atom_type: str
    atom_index: int
    connection_index: int
    connected_atom_index: int


def get_atom_caps(m: rdchem.Mol) -> tuple[list[MonomerAtomCap], bool]:
    """Extract connection point information from a monomer molecule.

    Identifies dummy atoms (atomic number 0) in the molecule and extracts
    information about their connection points, including the atom types they
    connect to and their connection indices from SMILES notation.

    Args:
        m: RDKit molecule object containing dummy atoms as connection points.

    Returns:
        A tuple containing:
            - List of MonomerAtomCap objects sorted by connection_index.
            - Boolean indicating if this is a dimer (both connection points
              connect to the same atom).
    """
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
    for index, connection_index in zip(connected_indices, numbers, strict=False):
        atom_type = str(symbols[A[index, :] >= 1][0])
        if len(atom_type) != 1:
            atom_type = f"[{atom_type}]"
        atom_index = int(np.argwhere(A[index, :] >= 1)[0, 0])
        connected_atom_index = index
        atomCaps.append(
            MonomerAtomCap(atom_type, atom_index, int(connection_index), connected_atom_index)
        )
    dimer = all(obj.atom_index == atomCaps[0].atom_index for obj in atomCaps)
    atomCaps.sort(key=lambda x: x.connection_index)

    return atomCaps, dimer


def generate_dimer_smiles(m_smiles: str, caps: list[MonomerAtomCap]) -> str:
    """Generate a dimer SMILES string by connecting two monomer units.

    Creates a dimer by replacing both connection points with a common label
    and using molzip to connect them.

    Args:
        m_smiles: SMILES string of the monomer with connection points.
        caps: List of MonomerAtomCap objects (expected to have 2 elements).

    Returns:
        SMILES string of the dimer formed by connecting two monomer units.
    """
    smiles_1 = m_smiles.replace(f"[*:{caps[0].connection_index}]", "[*:100]")
    smiles_2 = m_smiles.replace(f"[*:{caps[1].connection_index}]", "[*:100]")
    mol1 = rdmolfiles.MolFromSmiles(smiles_1)
    mol2 = rdmolfiles.MolFromSmiles(smiles_2)
    mol = rdmolops.molzip(mol1, mol2)
    return rdmolfiles.MolToSmiles(mol)


def generate_monomer(  # noqa: PLR0915
    m: rdchem.Mol,
    monomer_dihedral: float | None = None,
    dihedral_cutoff: float = 10,
    number_conformers: int = 100,
) -> tuple[Molecule, bool]:
    """Generate a capped monomer for creating a polymer chain.

    For a monomer, `m`, the unit is characterized by the following points:
    C0 - A0 - R - A1 - C1
    or
    C0 - A0 - C1
         |
         R
    Where C0 and C1 are the connection points, and A0 and A1 are the atoms that are the head and
    tail connections for the monomer. For a repeating chain, the atom type of C0 = A1, and C1 = A0.

    The function generates multiple conformers, optimizes them with UFF force field,
    and selects the lowest energy conformer that meets the dihedral angle constraints.
    The resulting molecule is translated and rotated to a standard orientation.

    Args:
        m: RDKit molecule object representing the monomer with connection points.
        monomer_dihedral: Optional target dihedral angle (in degrees) for the connecting
            dihedral. If provided, conformers are filtered to those within dihedral_cutoff
            of this value.
        dihedral_cutoff: Maximum deviation (in degrees) from monomer_dihedral for
            conformer selection. Only used if monomer_dihedral is provided.
        number_conformers: Number of conformers to generate and optimize.

    Returns:
        A tuple containing:
            - pymatgen Molecule object of the capped monomer in its optimal conformation,
              oriented with C0 at origin and aligned along z-axis.
            - Boolean indicating if this is a dimer (both connection points connect
              to the same atom).
    """
    # Canonicalize the SMILES for the molecule
    m = rdmolops.RemoveHs(m)
    m_smiles = rdmolfiles.MolToSmiles(m)
    m = rdmolfiles.MolFromSmiles(m_smiles)
    # Get the monomer connections
    atom_caps, dimer = get_atom_caps(m)

    if dimer:
        m_smiles = generate_dimer_smiles(m_smiles, atom_caps)
        m = rdmolfiles.MolFromSmiles(m_smiles)
        atom_caps, _ = get_atom_caps(m)

    head_cap, tail_cap = atom_caps
    # Replace First Atom Cap Type:
    m_capped_smiles = m_smiles.replace(
        f"[*:{head_cap.connection_index}]", tail_cap.atom_type
    ).replace(f"[*:{tail_cap.connection_index}]", head_cap.atom_type)

    monomer_capped_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(m_capped_smiles))
    # Embed conformers for the capped monomer
    rdDistGeom.EmbedMultipleConfs(monomer_capped_mol, numConfs=number_conformers)
    # Get the connecting dihedral
    atom_i = atom_caps[0].connected_atom_index
    atom_j = atom_caps[0].atom_index
    atom_k = atom_caps[1].atom_index
    atom_l = atom_caps[1].connected_atom_index

    energies = np.zeros(monomer_capped_mol.GetNumConformers())
    for _i, confId in enumerate(range(monomer_capped_mol.GetNumConformers())):
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer_capped_mol, confId=confId)
        if monomer_dihedral is not None:
            ff.UFFAddTorsionConstraint(
                atom_i,
                atom_j,
                atom_k,
                atom_l,
                False,
                monomer_dihedral,
                monomer_dihedral,
                10_000,
            )
        rdForceFieldHelpers.OptimizeMolecule(ff=ff)
        energies[_i] = ff.CalcEnergy()
    df = pd.DataFrame({"confId": range(monomer_capped_mol.GetNumConformers()), "energy": energies})
    if monomer_dihedral is not None:
        dihedrals = np.array(
            [
                rdMolTransforms.GetDihedralDeg(conf, atom_i, atom_j, atom_k, atom_l)
                for conf in monomer_capped_mol.GetConformers()
            ]
        )
        dihedrals = np.abs(dihedrals)
        df["dihedral"] = np.max(dihedrals) - dihedrals
        df.sort_values(by=["dihedral"], inplace=True, ascending=False)
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
            if atom.GetAtomicNum() == 1:
                extra_Hs.append(atom.GetIdx())
    sites = np.delete(sites, extra_Hs)
    # Reorder the sites so that C0, A0, ..., A1, C1
    indices = np.array(list(range(len(sites))))
    idx1 = atom_caps[1].connected_atom_index
    idx2 = atom_caps[1].atom_index
    idx3 = atom_caps[0].atom_index
    idx4 = atom_caps[0].connected_atom_index
    indices = list(np.delete(indices, [idx1, idx2, idx3, idx4]))
    order = [idx4, idx3, *indices, idx2, idx1]
    # Form Molecule
    molecule = Molecule.from_sites(sites[order].tolist())
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
    last_monomer: Molecule, next_monomer: Molecule, dihedral_angle: float
) -> Molecule:
    """Rotate and position a monomer to connect to the previous monomer in a chain.

    Aligns the new monomer's connection point (C0) with the previous monomer's
    connection point, rotates to align the bond vectors, and sets the dihedral
    angle between the two monomers.

    Args:
        last_monomer: The previous monomer in the chain (pymatgen Molecule).
        next_monomer: The monomer to be added to the chain (pymatgen Molecule).
        dihedral_angle: Target dihedral angle (in degrees) for the connection
            between the two monomers.

    Returns:
        A new pymatgen Molecule object representing the rotated and positioned
        next_monomer, ready to be appended to the chain.
    """
    # Align New-C0 with Old-A1/A0(dimer)
    old = last_monomer.copy().to_ase_atoms()
    new = next_monomer.copy().to_ase_atoms()
    connection_point = old[-2].position
    new.translate(connection_point - new[0].position)

    # Align Old - A1-C1 with New C0 - A0
    V1 = old[-1].position - old[-2].position
    V2 = new[1].position - new[0].position
    center = new[0].position
    new.rotate(V2, V1, center=center)

    # Fix Dihedrals
    atoms = old + new
    atom_i = 1  # Chain - A0
    atom_j = len(old) - 2  # Chain - A1
    atom_k = len(old) + 1  # Monomer - A0
    atom_l = len(old) + len(new) - 2  # Monomer - A1

    atoms.set_dihedral(
        atom_i,
        atom_j,
        atom_k,
        atom_l,
        dihedral_angle,
        indices=range(len(new), len(atoms)),
    )

    # Return only the transformed "new" monomer block.
    # Important: when copolymer monomers have different sizes, slicing must
    # be based on len(old), not len(new).
    return Molecule.from_ase_atoms(atoms[len(old) :])


def add_cap(p: Polymer, last_monomer: Molecule, head: bool = True) -> Molecule:  # noqa: PLR0915
    """Add a head or tail cap group to a polymer chain.

    Connects either the head or tail cap group from the Polymer object to
    the first or last monomer in the chain. The cap is positioned using
    constrained embedding to match the geometry of the adjacent monomer.

    Args:
        p: Polymer object containing the monomer, head, and tail cap molecules.
        last_monomer: The monomer at the end of the chain where the cap will
            be attached (first monomer for head=True, last monomer for head=False).
        head: If True, add the head cap; if False, add the tail cap.

    Returns:
        A pymatgen Molecule object representing the cap group positioned and
        ready to be connected to the chain.

    Raises:
        AssertionError: If the constrained embedding of the cap fails.
    """
    if head:
        h = rdmolops.RemoveHs(p.head)
    else:
        h = rdmolops.RemoveHs(p.tail)
    head_atom_caps, _ = get_atom_caps(h)
    [h_head_cap] = head_atom_caps
    h = rdmolops.AddHs(h)
    indices = np.array(list(range(h.GetNumAtoms())))
    idx1 = h_head_cap.connected_atom_index
    idx2 = h_head_cap.atom_index
    indices = list(np.delete(indices, [idx1, idx2]))
    order = [idx1, idx2, *indices]
    h = rdmolops.RenumberAtoms(mol=h, newOrder=np.array(order).tolist())
    # 2. Repeat for Monomer Unit
    m = rdmolops.RemoveHs(p.monomer)
    m_smiles = rdmolfiles.MolToSmiles(m)
    atom_caps, dimer = get_atom_caps(m)
    m_head_cap, m_tail_cap = atom_caps
    if dimer:
        m_smiles = generate_dimer_smiles(m_smiles, atom_caps)
        m = rdmolfiles.MolFromSmiles(m_smiles)
        atom_caps, dimer = get_atom_caps(m)
        m_head_cap, m_tail_cap = atom_caps
    # Update the monomer graph
    m_smiles = m_smiles.replace(f"[*:{m_head_cap.connection_index}]", m_tail_cap.atom_type).replace(
        f"[*:{m_tail_cap.connection_index}]", m_head_cap.atom_type
    )
    m = rdmolops.AddHs(rdmolfiles.MolFromSmiles(m_smiles))
    # Remove connected H atoms
    monomer_atoms = np.array(m.GetAtoms())
    A = rdmolops.GetAdjacencyMatrix(m)
    extra_Hs = []
    for ac in atom_caps:
        connected_atoms = monomer_atoms[A[ac.connected_atom_index] >= 1]
        for atom in connected_atoms:
            if atom.GetAtomicNum() == 1:
                extra_Hs.append(atom.GetIdx())
    m = rdchem.EditableMol(m)
    extra_Hs.sort(reverse=True)
    for _i in extra_Hs:
        m.RemoveAtom(_i)
    m = m.GetMol()
    m = rdchem.RWMol(m)
    if head:
        dummy_atom_1 = rdchem.Atom(0)  # Atomic number 0 = dummy atom
        dummy_atom_1.SetProp("molAtomMapNumber", str(m_head_cap.connection_index))
        m.ReplaceAtom(m_head_cap.connected_atom_index, dummy_atom_1)
    else:
        dummy_atom_2 = rdchem.Atom(0)  # Atomic number 0 = dummy atom
        dummy_atom_2.SetProp("molAtomMapNumber", str(m_tail_cap.connection_index))
        m.ReplaceAtom(m_tail_cap.connected_atom_index, dummy_atom_2)
    indices = np.array(list(range(m.GetNumAtoms())))
    idx1 = atom_caps[1].connected_atom_index
    idx2 = atom_caps[1].atom_index
    idx3 = atom_caps[0].atom_index
    idx4 = atom_caps[0].connected_atom_index
    indices = list(np.delete(indices, [idx1, idx2, idx3, idx4]))
    order = [idx4, idx3, *indices, idx2, idx1]
    m = rdmolops.RenumberAtoms(mol=m, newOrder=[int(i) for i in order])
    # 3. Update Head smiles to have the correct end_cap
    hm = rdmolops.molzip(m, h)
    # hm = rdmolops.AddHs(hm)
    coordMap = {}
    if head:
        _sites = [*last_monomer.sites[1:], last_monomer.sites[0]]
    else:
        _sites = last_monomer.sites
    for i, site in enumerate(_sites):
        coordMap[i] = Geometry.Point3D(*site.coords)
    # 9. Perform the constrained embedding
    res = rdDistGeom.EmbedMolecule(hm, coordMap=coordMap, useRandomCoords=True)
    assert res == 0, "The monomer embedding failed. Please file an issue to resolve"
    # # 10. Extract the head/tail unit
    sites = []
    conf = hm.GetConformer()
    for _i in range(m.GetNumAtoms() - 1, hm.GetNumAtoms()):
        sites.append(
            Site(coords=conf.GetAtomPosition(_i), species=hm.GetAtomWithIdx(_i).GetSymbol())
        )
    head_mol = Molecule.from_sites(sites)
    return head_mol


def build_chain(
    monomer: Molecule,
    dihedral_angle: list[float],
) -> list[Molecule]:
    """Build a polymer chain by connecting multiple monomer units.

    Creates a chain of monomers by repeatedly rotating and connecting monomer
    units with the specified dihedral angles between each pair.

    Args:
        monomer: The base monomer unit (pymatgen Molecule) to repeat in the chain.
        dihedral_angle: List of dihedral angles (in degrees) for each connection
            between monomers. The chain will have len(dihedral_angle) + 1 monomers.

    Returns:
        List of pymatgen Molecule objects representing each monomer in the chain,
        positioned and oriented for connection.
    """
    monomer_start = monomer.copy()
    monomer_list: list[Molecule] = [monomer.copy()]

    for phi in dihedral_angle:
        next_monomer = rotate_monomer(monomer_list[-1], monomer_start, phi)
        monomer_list.append(next_monomer)

    return monomer_list


def build_dimer_chain(
    p: Polymer,
    dihedral_cutoff: float,
    number_conformers: int,
    chain_dihedrals: list[float],
) -> list[Molecule]:
    """Build a polymer chain for dimer-type monomers.

    For dimer monomers (where both connection points attach to the same atom),
    this function alternates between generating monomers with specific internal
    dihedrals and connecting them with chain dihedrals.

    Args:
        p: Polymer object containing the monomer molecule.
        dihedral_cutoff: Maximum deviation (in degrees) from target dihedral
            for conformer selection during monomer generation.
        number_conformers: Number of conformers to generate for each monomer.
        chain_dihedrals: List of dihedral angles alternating between monomer
            internal dihedrals (even indices) and chain connection dihedrals
            (odd indices). Expected to have even length.

    Returns:
        List of pymatgen Molecule objects representing each monomer in the chain,
        positioned and oriented for connection.
    """
    monomer_dihedrals = chain_dihedrals[::2]
    chain_dihedrals = chain_dihedrals[1::2]
    C = []
    monomers = [
        generate_monomer(p.monomer, phi, dihedral_cutoff, number_conformers)[0]
        for phi in monomer_dihedrals
    ]
    C.append(monomers.pop(0))
    for next_monomer, phi in zip(monomers, chain_dihedrals, strict=False):
        M = rotate_monomer(C[-1], next_monomer, phi)
        C.append(M)
    return C


def make_finite_chain(
    p: Polymer,
    dihedrals: list[float],
    dihedral_cutoff: float = 10,
    number_conformers: int = 100,
    monomer_dihedral: float | None = None,
) -> Molecule:
    """Construct a complete finite polymer chain with head and tail caps.

    Generates a full polymer chain starting from a monomer, building the chain
    with specified dihedral angles, and adding head and tail cap groups.
    Handles both regular monomers and dimer-type monomers automatically.

    Args:
        p: Polymer object containing monomer, head, and tail cap molecules.
        dihedrals: List of dihedral angles (in degrees) for connections between
            monomers. For dimer monomers, this alternates between monomer internal
            dihedrals and chain dihedrals.
        dihedral_cutoff: Maximum deviation (in degrees) from target dihedral
            for conformer selection during monomer generation.
        number_conformers: Number of conformers to generate and optimize for
            each monomer unit.
        monomer_dihedral: Optional target dihedral angle (in degrees) for the
            first monomer's internal dihedral. If None, uses the lowest energy
            conformer without dihedral constraints.

    Returns:
        A complete pymatgen Molecule object representing the full polymer chain
        including head cap, all monomer units, and tail cap.
    """
    M, D = generate_monomer(p.monomer, monomer_dihedral, dihedral_cutoff, number_conformers)

    if not D:
        C = build_chain(M, dihedrals)
    else:
        C = build_dimer_chain(p, dihedral_cutoff, number_conformers, dihedrals)

    # Add head and tail groups
    H = add_cap(p, C[0], head=True)
    T = add_cap(p, C[-1], head=False)
    # Assemble the full chain
    # Head minus last atom (connection point)
    _c = H.to_ase_atoms()
    for mer in C:
        _c += mer.to_ase_atoms()[1:-1]
    _c += T.to_ase_atoms()
    return Molecule.from_ase_atoms(_c)


def generate_alternating_sequence(chain_length: int, units: list[int]) -> list[int]:
    """Generate an alternating sequence from a set/order of unit indices."""
    if chain_length < 1:
        raise ValueError("chain_length must be >= 1")
    minimum_units = 2
    if len(units) < minimum_units:
        raise ValueError("alternating units must include at least two entries")
    return [units[i % len(units)] for i in range(chain_length)]


def generate_periodic_sequence(chain_length: int, motif: list[int]) -> list[int]:
    """Generate a periodic sequence by repeating a motif."""
    if chain_length < 1:
        raise ValueError("chain_length must be >= 1")
    if len(motif) < 1:
        raise ValueError("motif must contain at least one unit index")
    return [motif[i % len(motif)] for i in range(chain_length)]


def generate_block_sequence(
    block_units: list[int],
    block_lengths: list[int],
    chain_length: int | None = None,
) -> list[int]:
    """Generate a block copolymer sequence from unit IDs and block lengths."""
    if len(block_units) < 1 or len(block_lengths) < 1:
        raise ValueError("block_units and block_lengths must be non-empty")
    if len(block_units) != len(block_lengths):
        raise ValueError("block_units and block_lengths must have same length")
    if any(x < 1 for x in block_lengths):
        raise ValueError("all block lengths must be >= 1")
    seq: list[int] = []
    for u, n in zip(block_units, block_lengths, strict=False):
        seq.extend([u] * n)
    if chain_length is not None:
        if chain_length < 1:
            raise ValueError("chain_length must be >= 1")
        if len(seq) < chain_length:
            raise ValueError("block sequence shorter than requested chain_length")
        seq = seq[:chain_length]
    return seq


def generate_weighted_random_sequence(
    chain_length: int,
    unit_weights: list[float],
    seed: int | None = None,
) -> list[int]:
    """Generate a weighted random sequence of monomer indices.

    Args:
        chain_length: Number of monomer units in the chain.
        unit_weights: Relative weights for each monomer type.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of monomer indices of length ``chain_length``.
    """
    if chain_length < 1:
        raise ValueError("chain_length must be >= 1")
    if len(unit_weights) < 1:
        raise ValueError("unit_weights must contain at least one weight")
    weights = np.array(unit_weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("unit_weights must be non-negative")
    if float(weights.sum()) <= 0:
        raise ValueError("unit_weights must have positive total weight")
    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(len(unit_weights), size=chain_length, p=probs).astype(int).tolist()


def make_finite_copolymer_chain(  # noqa: PLR0913
    polymers: list[Polymer],
    sequence: list[int],
    dihedrals: list[float],
    dihedral_cutoff: float = 10,
    number_conformers: int = 100,
    monomer_dihedral: float | None = None,
) -> Molecule:
    """Construct a finite capped co-polymer chain from a sequence of monomer templates.

    This implementation builds a connected molecular graph by remapping dummy atom
    labels and using RDKit ``molzip`` to stitch head cap, monomer sequence, and
    tail cap together. 3D coordinates are then generated with ETKDG + UFF.

    Notes:
        ``dihedrals``, ``dihedral_cutoff``, ``number_conformers``, and
        ``monomer_dihedral`` are currently accepted for API compatibility, but
        the graph-first co-polymer path does not yet enforce target torsions.
    """
    if len(sequence) < 1:
        raise ValueError("sequence must contain at least one monomer index")
    if len(dihedrals) != len(sequence) - 1:
        raise ValueError("dihedrals length must equal len(sequence)-1")

    del dihedral_cutoff, number_conformers, monomer_dihedral
    if max(sequence) >= len(polymers):
        raise ValueError("sequence includes polymer index out of range")

    def _dummy_labels(smiles: str) -> list[int]:
        labels = re.findall(r"\[\*:(\d+)\]", smiles)
        if not labels:
            raise ValueError("Expected monomer/cap smiles with mapped dummy atoms [*:n]")
        return [int(x) for x in labels]

    def _remap_dummy_labels(smiles: str, mapping: dict[int, int]) -> str:
        out = smiles
        for old, new in mapping.items():
            out = out.replace(f"[*:{old}]", f"[*:{new}]")
        return out

    first_polymer = polymers[sequence[0]]
    last_polymer = polymers[sequence[-1]]

    if first_polymer.head is None or last_polymer.tail is None:
        raise ValueError("Copolymer generation requires head cap on first and tail cap on last")

    head_smiles = rdmolfiles.MolToSmiles(first_polymer.head)
    tail_smiles = rdmolfiles.MolToSmiles(last_polymer.tail)

    parts: list[str] = []
    head_label = _dummy_labels(head_smiles)[0]
    parts.append(_remap_dummy_labels(head_smiles, {head_label: 1000}))

    expected_dummy_count = 2
    for i, polymer_idx in enumerate(sequence):
        monomer_smiles = rdmolfiles.MolToSmiles(polymers[polymer_idx].monomer)
        labels = sorted(_dummy_labels(monomer_smiles))
        if len(labels) != expected_dummy_count:
            raise ValueError("Each monomer must contain exactly two mapped dummy atoms")
        mapping = {labels[0]: 1000 + i, labels[1]: 1000 + i + 1}
        parts.append(_remap_dummy_labels(monomer_smiles, mapping))

    tail_label = _dummy_labels(tail_smiles)[0]
    parts.append(_remap_dummy_labels(tail_smiles, {tail_label: 1000 + len(sequence)}))

    zipped_input = rdmolfiles.MolFromSmiles(".".join(parts))
    if zipped_input is None:
        raise ValueError("Failed to build RDKit input molecule for copolymer zipping")

    mol = rdmolops.molzip(zipped_input)
    # Ensure internal caches (e.g. ring info) are initialized before force-field use.
    mol.UpdatePropertyCache(strict=False)
    rdmolops.SanitizeMol(mol)
    mol = rdmolops.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 7
    if rdDistGeom.EmbedMolecule(mol, params) != 0:
        raise ValueError("RDKit embedding failed for generated copolymer")
    rdForceFieldHelpers.UFFOptimizeMolecule(mol)

    return Molecule.from_str(rdmolfiles.MolToXYZBlock(mol), fmt="xyz")
