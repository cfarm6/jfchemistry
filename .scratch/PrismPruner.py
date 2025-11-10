import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from prism_pruner.conformer_ensemble import ConformerEnsemble
    from pymatgen.io.xyz import XYZ

    return ConformerEnsemble, XYZ, np


@app.cell
def _(XYZ):
    mols = XYZ.from_file(
        "/home/carson/research/SoftwareDevelopment/jfchemistry/examples/crest_conformers.xyz"
    ).all_molecules
    return (mols,)


@app.cell
def _(mols, np):
    atoms = np.array([s.name for s in mols[0].species])
    coords = np.array([m.cart_coords for m in mols])
    return atoms, coords


@app.cell
def _(ConformerEnsemble, atoms, coords):
    ensemble = ConformerEnsemble(atoms=atoms, coords=coords)
    return (ensemble,)


@app.cell
def _():
    from prism_pruner.pruner import prune_by_moment_of_inertia

    return (prune_by_moment_of_inertia,)


@app.cell
def _(ensemble, prune_by_moment_of_inertia):
    pruned, mask = prune_by_moment_of_inertia(
        ensemble.coords,
        ensemble.atoms,
        max_deviation=0.0001,  # 1% difference
        debugfunction=print,
    )
    return (mask,)


@app.cell
def _(mask):
    mask


@app.cell
def _(mols):
    mols[1]


@app.cell
def _(mask, mols):
    mols[mask]


@app.cell
def _():
    from molbar.helper.ensemble_splitter import EnsembleSplitter

    return (EnsembleSplitter,)


@app.cell
def _(XYZ):
    filename = "/home/carson/research/SoftwareDevelopment/jfchemistry/examples/launcher_2025-11-10-16-31-54-045749/tautomers.xyz"
    m = XYZ.from_file(filename).all_molecules
    return (m,)


@app.cell
def _(m, np):
    cs = np.array([_m.cart_coords for _m in m])
    return (cs,)


@app.cell
def _(m, np):
    es = np.array([[s.name for s in _m.species] for _m in m])
    return (es,)


@app.cell
def _(EnsembleSplitter, cs, es):
    ES = EnsembleSplitter(list_of_coordinates=cs, list_of_elements=es)
    return (ES,)


@app.cell
def _(es):
    es.get_ensemble_energies()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
