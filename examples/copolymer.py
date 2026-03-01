"""Example: build a finite co-polymer chain and write XYZ output."""

from pathlib import Path

from rdkit.Chem import rdmolfiles, rdmolops

from jfchemistry.core.structures import Polymer, RDMolMolecule
from jfchemistry.polymers import GenerateFiniteCopolymerChain


def polymer_from_smiles(head: str, monomer: str, tail: str) -> Polymer:
    """Create a Polymer object from head/monomer/tail SMILES templates."""
    head_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(head))
    monomer_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer))
    tail_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(tail))
    return Polymer(
        head=RDMolMolecule(head_mol),
        monomer=RDMolMolecule(monomer_mol),
        tail=RDMolMolecule(tail_mol),
    )


def main() -> None:
    """Generate and save a weighted-random co-polymer chain as XYZ."""
    poly_a = polymer_from_smiles("C[*:1]", "[*:1]CON[*:2]", "C[*:2]")
    poly_b = polymer_from_smiles("C[*:1]", "[*:1]CC(C=CC=C)[*:2]", "C[*:2]")

    maker = GenerateFiniteCopolymerChain(
        sequence_mode="weighted_random",
        chain_length=8,
        unit_weights=[0.65, 0.35],
        random_seed=42,
        dihedral_angles=175.0,
        num_conformers=5,
        monomer_dihedral=180.0,
    )

    response = maker.make.original(maker, [poly_a, poly_b])
    xyz = response.output.files["chain.xyz"]

    out = Path("examples/copolymer_weighted_random.xyz")
    out.write_text(xyz)
    print(f"Wrote {out}")
    print(f"Sequence: {maker.sequence}")


if __name__ == "__main__":
    main()
