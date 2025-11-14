import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    from os import path
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from ase.visualize import view
    from pymatgen.core.structure import (
        Lattice,
        Molecule,
        Site,
        Structure,
    )
    from rdkit.Chem import (
        rdDistGeom,
        rdForceFieldHelpers,
        rdmolfiles,
        rdmolops,
        rdMolTransforms,
    )

    return (
        Lattice,
        Molecule,
        Path,
        Site,
        Structure,
        np,
        path,
        pd,
        rdDistGeom,
        rdForceFieldHelpers,
        rdMolTransforms,
        rdmolfiles,
        rdmolops,
        view,
    )


@app.cell
def _(Path, path):
    working_dir = Path(path.split(__file__)[0])
    return (working_dir,)


@app.function
def fetch_dummy_index_and_bond_type(m) -> tuple[list[int], list[int], str]:
    """Fetch the dummy atom index and bond type from a smiles string."""
    dummy_index = []
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == "*":
                dummy_index.append(atom.GetIdx())
        for bond in m.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == "*" or bond.GetEndAtom().GetSymbol() == "*":
                bond_type = bond.GetBondType()
                break
    # Get the atoms connected to the dummy atoms
    connected_atoms = []
    for dummy_idx in dummy_index:
        for bond in m.GetBonds():
            if bond.GetBeginAtomIdx() == dummy_idx or bond.GetEndAtomIdx() == dummy_idx:
                idx = bond.GetOtherAtomIdx(dummy_idx)
                if idx not in connected_atoms:
                    connected_atoms.append(idx)
    return dummy_index, connected_atoms, str(bond_type)


@app.cell
def _():
    num_conformers = 100
    dihedral_angle_cutoff = 8
    interchain_distance = 12
    chain_length = 150
    angle = 60
    return (
        angle,
        chain_length,
        dihedral_angle_cutoff,
        interchain_distance,
        num_conformers,
    )


@app.cell
def _(rdmolfiles, rdmolops):
    monomer_smiles = "[1*]C(F)(F)[2*]"
    monomer = rdmolfiles.MolFromSmiles(monomer_smiles)
    dimerized = False
    # ----
    monomer = rdmolops.AddHs(monomer)
    monomer = rdmolops.RemoveAllHs(monomer)
    monomer_smiles = rdmolfiles.MolToSmiles(monomer)
    monomer = rdmolfiles.MolFromSmiles(monomer_smiles)
    connection_points, connected_atoms, bond_type = fetch_dummy_index_and_bond_type(monomer)

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
        connection_points, connected_atoms, bond_type = fetch_dummy_index_and_bond_type(monomer)
        dimerized = True
    atom1, atom2 = connected_atoms
    connection_point_1, connection_point_2 = connection_points
    # Replace the dummy atom with the appropriate bond type
    atom_types = {
        1: monomer.GetAtomWithIdx(atom2).GetSymbol(),
        2: monomer.GetAtomWithIdx(atom1).GetSymbol(),
    }

    for _i in [1, 2]:
        monomer_smiles = monomer_smiles.replace(f"[{_i}*]", atom_types[_i])

    monomer = rdmolfiles.MolFromSmiles(monomer_smiles)
    num_atoms = monomer.GetNumAtoms()
    monomer = rdmolops.AddHs(monomer)

    return (
        atom1,
        atom2,
        connection_point_1,
        connection_point_2,
        connection_points,
        dimerized,
        monomer,
        num_atoms,
    )


app._unparsable_cell(
    r"""
    def optimize_constrained_dihedral(monomer: rdchem.Mol, confId: int, i,j,k,l) -> float:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer, confId=confId)
         ff.UFFAddTorsionConstraint(
            i, j, k, l, False, 170, 190, 100
        )
        ff.Minimize()
        energy = ff.CalcEnergy()
        return energy
    """,
    name="_",
)


@app.cell
def _(
    Lattice,
    Molecule,
    Site,
    Structure,
    angle,
    atom1,
    atom2,
    chain_length,
    connection_point_1,
    connection_point_2,
    connection_points,
    dihedral_angle_cutoff,
    dimerized,
    interchain_distance,
    monomer,
    np,
    num_atoms,
    num_conformers,
    optimize_constrained_dihedral,
    pd,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolTransforms,
    view,
    working_dir,
):
    extra_atoms = []
    for bond in monomer.GetBonds():
        if bond.GetBeginAtomIdx() in connection_points:
            extra_atoms.append(bond.GetEndAtomIdx())
        elif bond.GetEndAtomIdx() in connection_points:
            extra_atoms.append(bond.GetBeginAtomIdx())
    extra_atoms = [index for index in extra_atoms if index >= num_atoms]
    rdDistGeom.EmbedMultipleConfs(monomer, numConfs=num_conformers)
    if dimerized:
        energies = [
            optimize_constrained_dihedral(
                monomer, confId, connection_point_1, atom1, atom2, connection_point_2
            )
            for confId in range(monomer.GetNumConformers())
        ]
    else:
        results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(monomer)
        energies = [r[1] for r in results]
    dihedrals = [
        rdMolTransforms.GetDihedralDeg(
            monomer.GetConformer(i), connection_point_1, atom1, atom2, connection_point_2
        )
        for i in range(monomer.GetNumConformers())
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

    sites = []
    for atom in monomer.GetAtoms():
        sites.append(Site(atom.GetSymbol(), conformer.GetAtomPosition(atom.GetIdx())))

    molecule = Molecule.from_sites(sites)
    atoms = molecule.to_ase_atoms()

    atoms.translate([0, 0, -atoms[connection_point_1].position[2]])
    atoms.rotate(atoms[connection_point_2].position, [0, 0, 1])

    molecule = Molecule.from_ase_atoms(atoms)

    c1 = molecule[connection_point_1].coords
    c2 = molecule[connection_point_2].coords

    # molecule.translate_sites(vector=[0,0,-molecule[atom1].coords[2]])
    # ud = c2 - molecule[atom2].coords
    molecule.remove_sites([connection_point_2, *extra_atoms])

    v_min = np.min(molecule.cart_coords, axis=0)
    v_max = np.max(molecule.cart_coords, axis=0)

    delta = v_max - v_min
    a = delta[0] + interchain_distance
    b = delta[1] + interchain_distance
    c = delta[2] + 0.001

    structure = molecule.get_boxed_structure(a=a, b=b, c=c, reorder=False, no_cross=False)

    print(structure[connection_point_1].coords)
    print(molecule[connection_point_1].coords)

    offset = molecule[connection_point_1].coords - structure[connection_point_1].coords
    structure.translate_sites(
        indices=range(len(structure)), vector=offset, frac_coords=False, to_unit_cell=False
    )

    print(structure[connection_point_1].coords)
    print(molecule[connection_point_1].coords)

    structure.remove_sites([connection_point_1])

    lattice = Lattice.from_parameters(a=a, b=b, c=c * chain_length, alpha=90, beta=90, gamma=90)
    final_structure = Structure(species=[], coords=[], lattice=lattice)
    for i in range(chain_length):
        _structure = structure.copy()
        if dimerized:
            theta_degrees = angle - 180
        else:
            theta_degrees = angle
        _structure.rotate_sites(
            theta=np.deg2rad(theta_degrees) * i,
            axis=[0, 0, 1],
            anchor=[0, 0, 0],
            to_unit_cell=False,
        )
        for site in _structure:
            final_structure.append(
                species=site.species,
                coords=site.coords + [0, 0, c * i],
                coords_are_cartesian=True,
            )
    final_structure.translate_sites(
        indices=range(len(final_structure)),
        vector=[a / 2, b / 2, 0],
        to_unit_cell=True,
        frac_coords=False,
    )
    view(final_structure.to_ase_atoms())
    final_structure.to(working_dir / "chain.cif")


@app.cell
def _(monomer, rdForceFieldHelpers):
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer, confId=10)
    return (ff,)


@app.cell
def _(atom1, atom2, connection_point_1, connection_point_2, ff):
    ff.UFFAddTorsionConstraint(
        connection_point_1, atom1, atom2, connection_point_2, False, 170, 190, 100
    )


@app.cell
def _(ff):
    ff.Minimize()
    ff.CalcEnergy()


@app.cell
def _(monomer, rdForceFieldHelpers):
    _ff = rdForceFieldHelpers.UFFGetMoleculeForceField(monomer, confId=10)
    _ff.CalcEnergy()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
