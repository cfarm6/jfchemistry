"""MolBar screening."""

import numpy as np
from molbar.helper.ensemble_splitter import EnsembleSplitter
from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ


def molbar_screening(filename: str, threads: int = 1) -> list[Molecule]:
    """Screen an ensemble of structures using MolBar."""
    mols = XYZ.from_file(filename).all_molecules
    if len(mols) == 1:
        return mols
    coords = np.array([mol.cart_coords for mol in mols])
    elements = np.array([[s.name for s in m.species] for m in mols])

    ensemble_splitter = EnsembleSplitter(list_of_coordinates=coords, list_of_elements=elements)

    ensemble_splitter.split_ensemble(threads=threads)
    ensembles = ensemble_splitter.return_ensembles()
    mols = [
        Molecule(
            ensemble[0][1],
            ensemble[0][0],
        )
        for ensemble in ensembles
    ]
    return mols
