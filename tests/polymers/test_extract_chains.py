"""Tests for extracting unwrapped polymer chains from periodic structures."""

from __future__ import annotations

import pytest
from pymatgen.core.structure import Lattice, Structure

from jfchemistry.polymers.extract_chains import extract_chains_from_structure

NUM_SITES = 2


def _simple_1d_chain_across_boundary() -> Structure:
    """Build a two-atom chain crossing the periodic boundary in x."""
    # Cubic box with side length 10 Å.
    lattice = Lattice.cubic(10.0)
    # Place one atom near x = 1 Å and the other near x = 9 Å. With PBC and
    # minimum-image convention, the true bond length should be ~2 Å, not ~8 Å.
    frac_coords = [
        [0.1, 0.5, 0.5],  # x = 1.0 Å
        [0.9, 0.5, 0.5],  # x = 9.0 Å
    ]
    return Structure(lattice, ["C", "C"], frac_coords)


def test_extract_chains_unwraps_across_pbc() -> None:
    """Chains that cross PBC should be unwrapped to the shortest image."""
    structure = _simple_1d_chain_across_boundary()

    molecules = extract_chains_from_structure(structure, bond_cutoff=2.1)

    # One connected component → one molecule with two atoms.
    assert len(molecules) == 1
    mol = molecules[0]
    assert len(mol.sites) == NUM_SITES

    # With correct unwrapping, the C-C distance should be close to 2 Å.
    dist = mol.get_distance(0, 1)
    assert dist == pytest.approx(2.0, rel=1e-3)
