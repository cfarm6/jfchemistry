"""Tests for polymer unit extraction."""

from __future__ import annotations

import pytest
from pymatgen.core.structure import Molecule
from rdkit.Chem import rdmolfiles, rdmolops

from jfchemistry.core.structures import Polymer, RDMolMolecule
from jfchemistry.polymers.extract_units import extract_units
from jfchemistry.polymers.generator import make_finite_chain


def _polymer_from_smiles(head: str | None, monomer: str, tail: str | None) -> Polymer:
    """Build a Polymer with explicit hydrogen atoms from SMILES templates."""
    head_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(head)) if head else None
    monomer_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer))
    tail_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(tail)) if tail else None
    return Polymer(
        head=RDMolMolecule(head_mol) if head_mol is not None else None,
        monomer=RDMolMolecule(monomer_mol),
        tail=RDMolMolecule(tail_mol) if tail_mol is not None else None,
    )


def _count_heavy_atoms(mol: Molecule) -> int:
    return sum(1 for site in mol if site.specie.Z > 1)


NUMBER_UNITS = 3
NUMBER_HEAVY_ATOMS = 3
NUMBER_HEAVY_ATOMS_PER_HEAD = 1
NUMBER_HEAVY_ATOMS_PER_TAIL = 1


def test_extract_units_canonical_equivalent_caps() -> None:
    """Extract head, monomer units, and tail from the canonical test polymer."""
    polymer = _polymer_from_smiles(
        head="C[*:1]",
        monomer="[*:1]C(C)(C)[*:2]",
        tail="C[*:2]",
    )
    # 3 monomers in chain: len(dihedrals)+1
    chain = make_finite_chain(polymer, dihedrals=[180.0, 180.0], number_conformers=1)

    units = extract_units(polymer=polymer, molecule=chain)

    assert len(units) >= NUMBER_UNITS
    # End caps are methyl groups => 1 heavy atom each
    assert _count_heavy_atoms(units[0]) == NUMBER_HEAVY_ATOMS_PER_HEAD
    assert _count_heavy_atoms(units[-1]) == NUMBER_HEAVY_ATOMS_PER_TAIL
    # Monomer units are tert-carbon with two methyl substituents => 3 heavy atoms each
    assert all(_count_heavy_atoms(m) == NUMBER_HEAVY_ATOMS for m in units[1:-1])


def test_extract_units_missing_caps_raises() -> None:
    """Head and tail are required for extraction."""
    polymer = _polymer_from_smiles(
        head=None,
        monomer="[*:1]C(C)(C)[*:2]",
        tail=None,
    )
    chain = Molecule(["C", "C"], [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])

    with pytest.raises(ValueError, match="requires both head and tail"):
        extract_units(polymer=polymer, molecule=chain)


def test_extract_units_branched_chain_rejected() -> None:
    """Branched heavy-atom graph is not supported."""
    polymer = _polymer_from_smiles(
        head="C[*:1]",
        monomer="[*:1]C(C)(C)[*:2]",
        tail="C[*:2]",
    )
    # Isobutane heavy-atom skeleton (branched): central carbon has degree 3.
    branched = Molecule(
        ["C", "C", "C", "C"],
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-1.54, 0.0, 0.0],
            [0.0, 1.54, 0.0],
        ],
    )

    with pytest.raises(ValueError, match="single linear chain"):
        extract_units(polymer=polymer, molecule=branched)
