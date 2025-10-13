"""RDKit generation."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

from pymatgen.core.structure import Molecule, SiteCollection

from jfchemistry.generation.base import StructureGeneration
from jfchemistry.jfchemistry import RDMolMolecule


@dataclass
class RDKitGeneration(StructureGeneration):
    """Maker for generating a structure using RDKit."""

    # Name of the job
    name: str = "rdKit Generation"
    # Method to use for generating the structure
    method: Literal["ETDG", "ETKDG", "ETKDGv2", "ETKDGv3", "KDG", "srETKDGv3"] = "ETKDGv3"
    basin_thresh: Optional[float] = None
    bounds_mat_force_scaling: Optional[float] = None
    box_size_mult: Optional[float] = None
    clear_confs: Optional[bool] = None
    embed_fragments_separately: Optional[bool] = None
    enable_sequential_random_seeds: Optional[bool] = None
    enforce_chirality: Optional[bool] = None
    force_trans_amides: Optional[bool] = None
    ignore_smoothing_failures: Optional[bool] = None
    max_iterations: Optional[Annotated[int, "positive"]] = None
    num_threads: Annotated[int, "positive"] = 1
    num_zero_fail: Optional[Annotated[int, "positive"]] = None
    only_heavy_atoms_for_rms: Optional[bool] = None
    optimizer_force_tol: Optional[float] = None
    prune_rms_thresh: Optional[float] = None
    rand_neg_eig: Optional[bool] = None
    random_seed: Optional[Annotated[int, "positive"]] = None
    symmetrize_conjugated_terminal_groups_for_pruning: Optional[bool] = None
    timeout: Optional[Annotated[int, "positive"]] = None
    track_failures: Optional[bool] = None
    use_basic_knowledge: Optional[bool] = None
    use_exp_torsion_angle_prefs: Optional[bool] = None
    use_macrocycle_14config: Optional[bool] = None
    use_macrocycle_torsions: Optional[bool] = None
    use_random_coords: Optional[bool] = None
    use_small_ring_torsions: Optional[bool] = None
    use_symmetry_for_pruning: Optional[bool] = None
    num_conformers: Annotated[int, "positive"] = 1

    def operation(
        self, mol: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], dict[str, Any]]:
        """Generate a structure using RDKit."""
        import inspect

        from rdkit.Chem import rdDistGeom, rdmolfiles, rdmolops

        params = getattr(rdDistGeom, self.method)()
        param_keys = [x[0] for x in inspect.getmembers(params)]
        for key, value in vars(self).items():

            def _to_camel_case(s: str) -> str:
                parts = s.split("_")
                return parts[0] + "".join(word.capitalize() for word in parts[1:])

            camel_key = _to_camel_case(key)
            if camel_key not in param_keys or value is None:
                continue
            setattr(params, camel_key, value)
        rdDistGeom.EmbedMultipleConfs(mol, self.num_conformers, params)
        molecules: list[SiteCollection] = []
        for confId in range(mol.GetNumConformers()):
            molecule: Molecule = Molecule.from_str(
                rdmolfiles.MolToV3KMolBlock(mol, confId=int(confId)),
                fmt="sdf",  # type: ignore[arg-type]
            )
            charge = rdmolops.GetFormalCharge(mol)
            spin = int(2 * (abs(charge) // 2) + 1)
            molecule.set_charge_and_spin(charge, spin)
            molecules.append(molecule)

        return molecules, {}
