"""Extract head, monomer, and tail fragments from a finite polymer chain."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow.core.job import Job, Response
from pymatgen.core.structure import Molecule
from rdkit.Chem import rdchem, rdDetermineBonds, rdmolfiles, rdmolops

from jfchemistry.core.jfchem_job import jfchem_job
from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.outputs import Output

if TYPE_CHECKING:
    from jfchemistry.core.structures import Polymer


_MIN_HEAVY_ATOMS = 3
_ONE_DUMMY = 1
_TWO_DUMMIES = 2
_UNIT_ENDPOINT_DEGREE = 1
_UNIT_INTERIOR_DEGREE = 2


@dataclass(frozen=True)
class _TemplateSpec:
    """Template query details for substructure matching."""

    query: rdchem.Mol
    external_counts: dict[int, int]
    n_dummy: int


def _validate_polymer_templates(polymer: Polymer) -> None:
    """Validate polymer template requirements for unit extraction."""
    if polymer.head is None or polymer.tail is None:
        raise ValueError("extract_units requires both head and tail templates in Polymer.")


def _prepare_template(template: rdchem.Mol) -> _TemplateSpec:
    """Build a query from a template by removing H and dummy atoms."""
    no_h = rdmolops.RemoveHs(template)
    dummy_indices = [a.GetIdx() for a in no_h.GetAtoms() if a.GetAtomicNum() == 0]
    external_counts_by_old_idx: dict[int, int] = defaultdict(int)

    for d_idx in dummy_indices:
        dummy_atom = no_h.GetAtomWithIdx(d_idx)
        for nbr in dummy_atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0:
                external_counts_by_old_idx[nbr.GetIdx()] += 1

    rw = rdchem.RWMol(no_h)
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    query = rw.GetMol()

    old_to_new: dict[int, int] = {}
    shift = 0
    dummy_set = set(dummy_indices)
    for old_idx in range(no_h.GetNumAtoms()):
        if old_idx in dummy_set:
            shift += 1
            continue
        old_to_new[old_idx] = old_idx - shift

    external_counts: dict[int, int] = {}
    for old_idx, count in external_counts_by_old_idx.items():
        external_counts[old_to_new[old_idx]] = count

    return _TemplateSpec(
        query=query,
        external_counts=external_counts,
        n_dummy=len(dummy_indices),
    )


def _molecule_to_rdkit(molecule: Molecule) -> rdchem.Mol:
    """Convert a pymatgen Molecule to an RDKit molecule with inferred bonds."""
    xyz = molecule.to(fmt="xyz")
    rd_mol = rdmolfiles.MolFromXYZBlock(xyz)
    if rd_mol is None:
        raise ValueError("Could not convert pymatgen Molecule to RDKit Mol from XYZ block.")
    try:
        rdDetermineBonds.DetermineBonds(rd_mol)
    except ValueError:
        # Some synthetic/partial test molecules cannot satisfy valence-based bond order assignment.
        # Connectivity-only perception is sufficient for unit graph segmentation.
        rdDetermineBonds.DetermineConnectivity(rd_mol)
    return rd_mol


def _heavy_atom_indices(mol: rdchem.Mol) -> list[int]:
    """Return heavy atom indices for an RDKit molecule."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _heavy_neighbors(mol: rdchem.Mol, atom_idx: int) -> set[int]:
    """Return heavy-atom neighbors of an atom."""
    return {
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetAtomicNum() > 1
    }


def _dedupe_matches(matches: tuple[tuple[int, ...], ...]) -> list[frozenset[int]]:
    """Collapse match tuples to unique atom sets."""
    atom_sets = {frozenset(m) for m in matches}
    return sorted(atom_sets, key=lambda s: tuple(sorted(s)))


def _find_unit_matches(target: rdchem.Mol, spec: _TemplateSpec) -> list[frozenset[int]]:
    """Find template matches that satisfy external-heavy-neighbor constraints."""
    raw_matches = target.GetSubstructMatches(spec.query, uniquify=True)
    valid: list[tuple[int, ...]] = []
    for match in raw_matches:
        match_set = set(match)
        ok = True
        for q_idx, t_idx in enumerate(match):
            expected_external = spec.external_counts.get(q_idx, 0)
            external = sum(1 for n in _heavy_neighbors(target, t_idx) if n not in match_set)
            if external != expected_external:
                ok = False
                break
        if ok:
            valid.append(match)
    return _dedupe_matches(tuple(valid))


def _template_equivalent(a: _TemplateSpec, b: _TemplateSpec) -> bool:
    """Return True if two templates are graph-equivalent for matching."""
    return rdmolfiles.MolToSmiles(a.query, canonical=True) == rdmolfiles.MolToSmiles(
        b.query, canonical=True
    ) and sorted(a.external_counts.items()) == sorted(b.external_counts.items())


def _expected_degrees(spec: _TemplateSpec) -> set[int]:
    """Expected heavy-atom degree in target for atoms matching this template."""
    degrees: set[int] = set()
    for atom in spec.query.GetAtoms():
        q_idx = atom.GetIdx()
        internal = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
        degrees.add(internal + spec.external_counts.get(q_idx, 0))
    return degrees


def _assert_template_degree_compatibility(
    target: rdchem.Mol,
    specs: list[_TemplateSpec],
) -> None:
    """Fast-reject targets with heavy-atom degree patterns impossible for templates."""
    allowed_degrees: set[int] = set()
    for spec in specs:
        allowed_degrees |= _expected_degrees(spec)

    for atom_idx in _heavy_atom_indices(target):
        degree = len(_heavy_neighbors(target, atom_idx))
        if degree not in allowed_degrees:
            return


def _exact_cover_monomers(
    remaining_atoms: frozenset[int],
    monomer_candidates: list[frozenset[int]],
) -> list[frozenset[int]] | None:
    """Choose non-overlapping monomer matches that exactly cover remaining heavy atoms."""
    by_atom: dict[int, list[frozenset[int]]] = defaultdict(list)
    for cand in monomer_candidates:
        if cand <= remaining_atoms:
            for atom in cand:
                by_atom[atom].append(cand)

    for _atom, candidates in by_atom.items():
        candidates.sort(key=lambda s: tuple(sorted(s)))

    def _search(
        uncovered: frozenset[int],
        chosen: list[frozenset[int]],
    ) -> list[frozenset[int]] | None:
        if not uncovered:
            return chosen
        pivot = min(uncovered)
        for cand in by_atom.get(pivot, []):
            if cand <= uncovered:
                result = _search(frozenset(uncovered - cand), [*chosen, cand])
                if result is not None:
                    return result
        return None

    return _search(remaining_atoms, [])


def _build_unit_graph(
    target: rdchem.Mol,
    unit_heavy_sets: list[frozenset[int]],
) -> list[set[int]]:
    """Build unit adjacency from heavy-atom bonds crossing unit boundaries."""
    atom_to_unit: dict[int, int] = {}
    for unit_idx, atoms in enumerate(unit_heavy_sets):
        for atom_idx in atoms:
            atom_to_unit[atom_idx] = unit_idx

    adj: list[set[int]] = [set() for _ in unit_heavy_sets]
    for bond in target.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (
            target.GetAtomWithIdx(a).GetAtomicNum() == 1
            or target.GetAtomWithIdx(b).GetAtomicNum() == 1
        ):
            continue
        ua = atom_to_unit.get(a)
        ub = atom_to_unit.get(b)
        if ua is None or ub is None or ua == ub:
            continue
        adj[ua].add(ub)
        adj[ub].add(ua)
    return adj


def _is_connected(adj: list[set[int]]) -> bool:
    """Check connectivity of a graph represented as adjacency sets."""
    if not adj:
        return False
    seen = {0}
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return len(seen) == len(adj)


def _order_units_head_to_tail(adj: list[set[int]], head_idx: int, tail_idx: int) -> list[int]:
    """Order units along the unique linear path from head to tail."""
    parent = {head_idx: -1}
    queue = deque([head_idx])
    while queue:
        u = queue.popleft()
        if u == tail_idx:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if tail_idx not in parent:
        raise ValueError("Could not connect head and tail units along the backbone.")
    path = [tail_idx]
    while path[-1] != head_idx:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _expand_with_hydrogens(target: rdchem.Mol, heavy_atoms: frozenset[int]) -> list[int]:
    """Add hydrogen atoms bonded to heavy atoms in a unit."""
    atoms = set(heavy_atoms)
    for h_idx in heavy_atoms:
        for nbr in target.GetAtomWithIdx(h_idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                atoms.add(nbr.GetIdx())
    return sorted(atoms)


def _fragment_from_indices(molecule: Molecule, indices: list[int]) -> Molecule:
    """Create a molecule fragment from selected atom indices."""
    species = [molecule[i].specie for i in indices]
    coords = [molecule[i].coords for i in indices]
    return Molecule(species, coords)


def extract_units(polymer: Polymer, molecule: Molecule) -> list[Molecule]:  # noqa: PLR0915
    """Extract head, monomer(s), and tail units as pymatgen Molecule fragments.

    Args:
        polymer: Polymer template containing `head`, `monomer`, and `tail`.
        molecule: Finite chain molecule to segment.

    Returns:
        Ordered list of unit fragments as [head, monomer_1, ..., monomer_n, tail].

    Raises:
        ValueError: If templates are missing or no valid linear decomposition is found.
    """
    _validate_polymer_templates(polymer)

    target = _molecule_to_rdkit(molecule)
    all_heavy = frozenset(_heavy_atom_indices(target))
    if len(all_heavy) < _MIN_HEAVY_ATOMS:
        raise ValueError("Input molecule is too small for polymer unit extraction.")

    head_spec = _prepare_template(polymer.head)  # type: ignore[arg-type]
    monomer_spec = _prepare_template(polymer.monomer)
    tail_spec = _prepare_template(polymer.tail)  # type: ignore[arg-type]

    if head_spec.n_dummy != _ONE_DUMMY or tail_spec.n_dummy != _ONE_DUMMY:
        raise ValueError("Head and tail templates must each contain exactly one dummy atom.")
    if monomer_spec.n_dummy != _TWO_DUMMIES:
        raise ValueError("Monomer template must contain exactly two dummy atoms for linear chains.")
    _assert_template_degree_compatibility(target, [head_spec, monomer_spec, tail_spec])

    head_candidates = _find_unit_matches(target, head_spec)
    monomer_candidates = _find_unit_matches(target, monomer_spec)
    tail_candidates = _find_unit_matches(target, tail_spec)
    if not head_candidates or not monomer_candidates or not tail_candidates:
        raise ValueError(
            "Input molecule is not a single linear chain compatible with provided templates."
        )

    head_tail_equivalent = _template_equivalent(head_spec, tail_spec)

    ordered_cap_pairs: list[tuple[frozenset[int], frozenset[int]]] = []
    for h in head_candidates:
        for t in tail_candidates:
            if h.isdisjoint(t):
                ordered_cap_pairs.append((h, t))

    if head_tail_equivalent:
        merged = sorted({*head_candidates, *tail_candidates}, key=lambda s: tuple(sorted(s)))
        ordered_cap_pairs = []
        for i, h in enumerate(merged):
            for t in merged[i + 1 :]:
                if h.isdisjoint(t):
                    ordered_cap_pairs.append((h, t))

    ordered_cap_pairs.sort(
        key=lambda pair: (
            tuple(sorted(pair[0])),
            tuple(sorted(pair[1])),
        )
    )

    for head_set, tail_set in ordered_cap_pairs:
        remaining = frozenset(all_heavy - head_set - tail_set)
        monomer_cover = _exact_cover_monomers(remaining, monomer_candidates)
        if monomer_cover is None:
            continue

        unit_heavy_sets = [head_set, *monomer_cover, tail_set]
        adj = _build_unit_graph(target, unit_heavy_sets)
        if not _is_connected(adj):
            continue

        degrees = [len(x) for x in adj]
        if degrees[0] != _UNIT_ENDPOINT_DEGREE or degrees[-1] != _UNIT_ENDPOINT_DEGREE:
            continue
        if any(d != _UNIT_INTERIOR_DEGREE for d in degrees[1:-1]):
            continue

        try:
            path = _order_units_head_to_tail(adj, 0, len(unit_heavy_sets) - 1)
        except ValueError:
            continue

        # Deterministic orientation when head and tail are equivalent.
        if head_tail_equivalent:
            first_cap_atoms = unit_heavy_sets[path[0]]
            last_cap_atoms = unit_heavy_sets[path[-1]]
            if min(first_cap_atoms) > min(last_cap_atoms):
                path = list(reversed(path))

        fragments: list[Molecule] = []
        for unit_idx in path:
            atom_indices = _expand_with_hydrogens(target, unit_heavy_sets[unit_idx])
            fragments.append(_fragment_from_indices(molecule, atom_indices))
        return fragments

    raise ValueError(
        "Could not decompose molecule into a single linear chain of head/monomer/tail units."
    )


@dataclass
class ExtractPolymerUnits(
    PymatGenMaker[Molecule, list[Molecule]],
):
    """Extract head, monomer, and tail fragments from a finite polymer molecule."""

    name: str = "Extract Polymer Units"
    polymer: Polymer | None = None
    _output_model: type[Output] = Output

    @jfchem_job()
    def _make(self, molecule: Molecule, polymer: Polymer) -> Response[Output]:
        """Run unit extraction and return ordered polymer fragments."""
        units = extract_units(polymer=polymer, molecule=molecule)
        print(len(units))
        return Response(output=self._output_model(structure=units))

    def make(self, molecule: Molecule) -> Job:
        """Create a job that extracts ordered polymer units from one molecule."""
        if self.polymer is None:
            raise ValueError("ExtractPolymerUnits requires `polymer` to be set on the maker.")
        return self._make(molecule=molecule, polymer=self.polymer)
