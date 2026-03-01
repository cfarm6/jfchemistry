"""Tests for finite co-polymer chain generation."""

from __future__ import annotations

import pytest
from pymatgen.core.structure import Molecule
from rdkit.Chem import rdmolfiles, rdmolops

from jfchemistry.core.structures import Polymer, RDMolMolecule
from jfchemistry.polymers.finite_chain import GenerateFiniteCopolymerChain
from jfchemistry.polymers.generator import (
    generate_alternating_sequence,
    generate_block_sequence,
    generate_periodic_sequence,
    generate_weighted_random_sequence,
    make_finite_copolymer_chain,
)


def _polymer_from_smiles(head: str, monomer: str, tail: str) -> Polymer:
    head_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(head))
    monomer_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(monomer))
    tail_mol = rdmolops.AddHs(rdmolfiles.MolFromSmiles(tail))
    return Polymer(
        head=RDMolMolecule(head_mol),
        monomer=RDMolMolecule(monomer_mol),
        tail=RDMolMolecule(tail_mol),
    )


def test_make_finite_copolymer_chain_smoke() -> None:
    """Generate a simple ABAB co-polymer chain."""
    poly_a = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")
    poly_b = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")

    chain = make_finite_copolymer_chain(
        polymers=[poly_a, poly_b],
        sequence=[0, 1, 0, 1],
        dihedrals=[180.0, 170.0, 160.0],
        number_conformers=1,
        monomer_dihedral=180.0,
    )

    assert isinstance(chain, Molecule)
    assert len(chain) > 0


def test_make_finite_copolymer_chain_validates_lengths() -> None:
    """Dihedral count must be len(sequence)-1."""
    poly_a = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")

    with pytest.raises(ValueError, match="dihedrals length"):
        make_finite_copolymer_chain(
            polymers=[poly_a],
            sequence=[0, 0, 0],
            dihedrals=[180.0],
            number_conformers=1,
        )


def test_generate_finite_copolymer_chain_maker() -> None:
    """Maker node should construct a job with an output reference."""
    poly_a = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")
    poly_b = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")

    maker = GenerateFiniteCopolymerChain(
        sequence=[0, 1, 0],
        dihedral_angles=[180.0, 175.0],
        num_conformers=1,
    )
    response = maker.make.original(maker, [poly_a, poly_b])

    assert response.output is not None
    assert response.output.structure is not None
    assert "chain.xyz" in response.output.files


def test_generate_weighted_random_sequence_reproducible() -> None:
    """Weighted random sequence should be reproducible with a seed."""
    chain_length = 10
    s1 = generate_weighted_random_sequence(chain_length, [0.8, 0.2], seed=7)
    s2 = generate_weighted_random_sequence(chain_length, [0.8, 0.2], seed=7)
    assert s1 == s2
    assert len(s1) == chain_length
    assert set(s1).issubset({0, 1})


def test_generate_finite_copolymer_chain_maker_random_mode() -> None:
    """Maker should support random sequence generation from unit weights."""
    poly_a = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")
    poly_b = _polymer_from_smiles("C[*:1]", "[*:1]CC[*:2]", "C[*:2]")
    chain_length = 5

    maker = GenerateFiniteCopolymerChain(
        sequence_mode="weighted_random",
        chain_length=chain_length,
        unit_weights=[0.7, 0.3],
        random_seed=11,
        dihedral_angles=180.0,
        num_conformers=1,
    )
    response = maker.make.original(maker, [poly_a, poly_b])

    assert len(maker.sequence) == chain_length
    assert response.output is not None
    assert response.output.structure is not None


def test_copolymer_maker_sequence_and_weights_mutually_exclusive() -> None:
    """Providing both explicit sequence and weighted mode should fail."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        GenerateFiniteCopolymerChain(
            sequence_mode="explicit",
            sequence=[0, 1, 0],
            chain_length=5,
            unit_weights=[0.5, 0.5],
            dihedral_angles=180.0,
        )


def test_generate_alternating_periodic_block_sequences() -> None:
    """Sequence helper functions should produce expected patterns."""
    assert generate_alternating_sequence(6, [0, 1]) == [0, 1, 0, 1, 0, 1]
    assert generate_periodic_sequence(7, [0, 0, 1]) == [0, 0, 1, 0, 0, 1, 0]
    assert generate_block_sequence([0, 1, 0], [2, 3, 1]) == [0, 0, 1, 1, 1, 0]


def test_maker_alternating_periodic_block_modes() -> None:
    """Maker should generate sequence in alternating/periodic/block modes."""
    m_alt = GenerateFiniteCopolymerChain(
        sequence_mode="alternating",
        chain_length=6,
        alternating_units=[0, 1],
        dihedral_angles=180.0,
    )
    assert m_alt.sequence == [0, 1, 0, 1, 0, 1]

    m_per = GenerateFiniteCopolymerChain(
        sequence_mode="periodic",
        chain_length=7,
        periodic_motif=[0, 0, 1],
        dihedral_angles=180.0,
    )
    assert m_per.sequence == [0, 0, 1, 0, 0, 1, 0]

    m_blk = GenerateFiniteCopolymerChain(
        sequence_mode="block",
        block_units=[0, 1, 0],
        block_lengths=[2, 3, 1],
        dihedral_angles=180.0,
    )
    assert m_blk.sequence == [0, 0, 1, 1, 1, 0]
