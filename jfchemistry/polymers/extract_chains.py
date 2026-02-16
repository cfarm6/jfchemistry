"""Extract polymer chains from periodic structures as pymatgen Molecules.

Supports unwrapping chains that cross periodic boundary conditions (e.g. from
molecular dynamics simulations) and returns one Molecule per chain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from jobflow.core.job import Response
from pymatgen.core.structure import Molecule, Structure

from jfchemistry.core.input_types import RecursiveStructureList
from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output


def _get_bonds(
    structure: Structure,
    bond_cutoff: float,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Identify bonds from a periodic structure using distance cutoff.

    Returns list of (site_i, site_j, image) where image is the integer
    periodic image (i, j, k) of site_j when bonded to site_i (so that
    the unwrapped position of j is structure.lattice.get_cartesian_coords(
    structure[j].frac_coords + image)).
    """
    neighbors = structure.get_all_neighbors(r=bond_cutoff)
    bonds: list[tuple[int, int, tuple[int, ...]]] = []
    seen = set()
    for i, nlist in enumerate(neighbors):
        for n in nlist:
            j = int(n.index)
            if i == j:
                continue
            # Avoid duplicate (i,j) / (j,i)
            edge = (min(i, j), max(i, j))
            if edge in seen:
                continue
            seen.add(edge)
            # Image of neighbor: frac_coords can be outside [0,1)
            image = tuple(int(np.floor(x)) for x in n.frac_coords)
            bonds.append((i, j, image))
    return bonds


def _build_adjacency(
    n_sites: int,
    bonds: list[tuple[int, int, tuple[int, ...]]],
) -> list[list[tuple[int, tuple[int, ...]]]]:
    """Build adjacency list: adj[i] = [(j, image), ...] for i -> j with image of j."""
    adj: list[list[tuple[int, tuple[int, ...]]]] = [[] for _ in range(n_sites)]
    for i, j, image in bonds:
        adj[i].append((j, image))
        # Reverse direction: when going j -> i, the image of i is -image
        adj[j].append((i, tuple(-x for x in image)))
    return adj


def _find_connected_components(
    n_sites: int,
    adj: list[list[tuple[int, tuple[int, ...]]]],
) -> list[list[int]]:
    """Find connected components (chains) from adjacency list."""
    visited = [False] * n_sites
    components: list[list[int]] = []
    for seed in range(n_sites):
        if visited[seed]:
            continue
        comp: list[int] = []
        queue = deque([seed])
        visited[seed] = True
        while queue:
            i = queue.popleft()
            comp.append(i)
            for j, _ in adj[i]:
                if not visited[j]:
                    visited[j] = True
                    queue.append(j)
        components.append(comp)
    return components


def _unwrap_chain(
    structure: Structure,
    site_indices: list[int],
    adj: list[list[tuple[int, tuple[int, ...]]]],
) -> np.ndarray:
    """Compute unwrapped fractional coordinates for one chain (connected component).

    Uses BFS from the first site; each site is assigned the fractional coords
    of the periodic image that connects it to the current "unwrapped" frame.
    """
    n_sites = len(structure)
    unwrapped_frac = np.zeros((n_sites, 3))
    # Mark which sites we care about and their order for this component
    in_component = set(site_indices)
    # Start from first site in unit cell coords
    seed = site_indices[0]
    unwrapped_frac[seed] = structure[seed].frac_coords
    visited = {seed}
    queue = deque([seed])
    while queue:
        i = queue.popleft()
        for j, image in adj[i]:
            if j not in in_component or j in visited:
                continue
            visited.add(j)
            # Neighbor n's frac_coords are the unwrapped position of j
            # when we reached j from i (that image of j)
            neighbor_frac = structure[j].frac_coords + np.array(image, dtype=float)
            unwrapped_frac[j] = neighbor_frac
            queue.append(j)
    return unwrapped_frac


def _chain_to_molecule(
    structure: Structure,
    site_indices: list[int],
    unwrapped_frac: np.ndarray,
) -> Molecule:
    """Build a pymatgen Molecule from a list of site indices and their unwrapped frac coords."""
    species = [structure[i].specie for i in site_indices]
    frac_coords = unwrapped_frac[site_indices]
    cart = structure.lattice.get_cartesian_coords(frac_coords)
    return Molecule(species, cart)


def extract_chains_from_structure(
    structure: Structure,
    bond_cutoff: float = 2.0,
) -> list[Molecule]:
    """Extract polymer chains from a periodic structure as unwrapped Molecules.

    Identifies bonds via distance cutoff, finds connected components (chains),
    unwraps each chain across periodic boundaries, and returns one Molecule per chain.

    Args:
        structure: Periodic structure (e.g. from MD) containing one or more chains.
        bond_cutoff: Maximum distance (Angstrom) for two atoms to be considered bonded.

    Returns:
        List of Molecules, one per chain, with coordinates unwrapped (no PBC).
    """
    if len(structure) == 0:
        return []
    bonds = _get_bonds(structure, bond_cutoff)
    if not bonds:
        # No bonds found: treat each site as its own "chain"
        return [
            Molecule(
                [structure[i].specie],
                structure.lattice.get_cartesian_coords([structure[i].frac_coords]),
            )
            for i in range(len(structure))
        ]
    adj = _build_adjacency(len(structure), bonds)
    components = _find_connected_components(len(structure), adj)
    molecules: list[Molecule] = []
    for comp in components:
        if not comp:
            continue
        unwrapped_frac = _unwrap_chain(structure, comp, adj)
        mol = _chain_to_molecule(structure, comp, unwrapped_frac)
        molecules.append(mol)
    return molecules


def _structures_from_input(input: RecursiveStructureList) -> list[Structure]:
    """Flatten RecursiveStructureList to a list of Structure for processing."""
    if isinstance(input, Structure):
        return [input]
    return [s for item in input for s in _structures_from_input(item)]


@dataclass
class ExtractPolymerChains(
    PymatGenMaker[RecursiveStructureList, list[Molecule]],
):
    """Extract polymer chains from periodic Structure(s) as unwrapped pymatgen Molecules.

    Accepts either a single Structure (e.g. one MD snapshot) or a list of Structures
    (e.g. multiple snapshots or pre-split chain structures). Uses bond detection
    (distance cutoff) to identify connected components, then unwraps each chain
    so that molecules are continuous across periodic boundaries.

    Attributes:
        name: Job name (default: "Extract Polymer Chains").
        bond_cutoff: Maximum distance (Angstrom) for a bond (default: 2.0).
    """

    name: str = "Extract Polymer Chains"
    bond_cutoff: float = field(
        default=2.0,
        metadata={"description": "Maximum distance (Angstrom) for two atoms to be bonded."},
    )
    _output_model: type[Output] = Output

    @jfchem_job()
    def make(self, input: RecursiveStructureList) -> Response[Output]:
        """Extract chains from Structure(s) and return unwrapped Molecules.

        Args:
            input: A single Structure (e.g. one periodic box from MD) or a list of
                Structures (e.g. one per chain or one per frame).

        Returns:
            Response with output.structure = list[Molecule], one per chain.
            If input is a list of Structures, chains from all are concatenated.
        """
        structures = _structures_from_input(input)
        molecules: list[Molecule] = []
        for s in structures:
            molecules.extend(
                extract_chains_from_structure(s, bond_cutoff=self.bond_cutoff),
            )
        return Response(output=self._output_model(structure=molecules))
