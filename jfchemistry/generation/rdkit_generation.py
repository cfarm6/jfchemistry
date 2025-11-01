"""RDKit-based 3D structure generation using distance geometry.

This module provides integration with RDKit's distance geometry embedding
methods for generating 3D molecular structures from 2D representations.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Optional

from pymatgen.core.structure import Molecule, SiteCollection

from jfchemistry.base_classes import RDMolMolecule
from jfchemistry.generation.base import StructureGeneration


@dataclass
class RDKitGeneration(StructureGeneration):
    """Generate 3D structures using RDKit distance geometry methods.

    This class uses RDKit's distance geometry embedding algorithms (ETKDG family)
    to generate 3D conformers from molecular graphs. The methods use distance
    bounds derived from experimental torsion angle preferences and small ring
    conformations to produce chemically reasonable 3D structures.

    The implementation supports all RDKit embedding parameters, allowing fine
    control over the generation process including optimization settings, pruning
    criteria, and random seed control.

    Attributes:
        name: Name of the job (default: "rdKit Generation").
        method: Distance geometry method to use:
            - "ETKDGv3": ETKDG version 3 (default, recommended)
            - "ETKDGv2": ETKDG version 2
            - "ETKDG": Original ETKDG method
            - "ETDG": Experimental torsion distance geometry
            - "KDG": Basic distance geometry
            - "srETKDGv3": Small ring ETKDG version 3
        basin_thresh: Energy threshold for basin hopping (default: None).
        bounds_mat_force_scaling: Scaling factor for distance bounds (default: None).
        box_size_mult: Box size multiplier for embedding (default: None).
        clear_confs: Clear existing conformers before embedding (default: None).
        embed_fragments_separately: Embed molecular fragments separately (default: None).
        enable_sequential_random_seeds: Use sequential random seeds (default: None).
        enforce_chirality: Enforce stereochemistry from molecular graph (default: None).
        force_trans_amides: Force amide bonds to trans configuration (default: None).
        ignore_smoothing_failures: Continue if smoothing fails (default: None).
        max_iterations: Maximum optimization iterations per conformer (default: None).
        num_threads: Number of parallel threads for embedding (default: 1).
        num_zero_fail: Number of failures before reporting (default: None).
        only_heavy_atoms_for_rms: Use only heavy atoms for RMSD pruning (default: None).
        optimizer_force_tol: Force tolerance for optimization (default: None).
        prune_rms_thresh: RMSD threshold for conformer pruning in Angstroms (default: None).
        rand_neg_eig: Randomize negative eigenvalues (default: None).
        random_seed: Random seed for reproducibility (default: None).
        symmetrize_conjugated_terminal_groups_for_pruning: Symmetrize terminal groups
            for RMSD calculations (default: None).
        timeout: Timeout in seconds for embedding (default: None).
        track_failures: Track embedding failures (default: None).
        use_basic_knowledge: Use basic chemical knowledge (default: None).
        use_exp_torsion_angle_prefs: Use experimental torsion preferences (default: None).
        use_macrocycle_14config: Use 1,4 distance bounds for macrocycles (default: None).
        use_macrocycle_torsions: Use macrocycle torsion preferences (default: None).
        use_random_coords: Start from random coordinates (default: None).
        use_small_ring_torsions: Use small ring torsion preferences (default: None).
        use_symmetry_for_pruning: Use symmetry when pruning conformers (default: None).
        num_conformers: Number of conformers to generate (default: 1).

    """

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

    def operation(self, mol: RDMolMolecule) -> tuple[SiteCollection | list[SiteCollection], None]:
        """Generate 3D structure(s) using RDKit distance geometry embedding.

        Embeds 3D coordinates into the molecule using the specified ETKDG method
        and parameters. Automatically configures the RDKit embedding parameters
        from the instance attributes and generates the requested number of conformers.

        The method:
        1. Creates RDKit embedding parameters object for the selected method
        2. Transfers all non-None attributes to the parameters
        3. Embeds conformers using distance geometry
        4. Converts conformers to Pymatgen Molecule objects
        5. Sets charge and spin multiplicity based on formal charge

        Args:
            mol: RDKit molecule without 3D coordinates.

        Returns:
            Tuple containing:
                - List of Pymatgen Molecule objects with 3D coordinates
                - Empty dictionary (no additional properties)

        Examples:
            >>> from rdkit import Chem # doctest: +SKIP
            >>> from jfchemistry import RDMolMolecule # doctest: +SKIP
            >>> from jfchemistry.generation import RDKitGeneration # doctest: +SKIP
            >>>
            >>> # Create molecule from SMILES
            >>> mol = Chem.MolFromSmiles("CCO") # doctest: +SKIP
            >>> rdmol = RDMolMolecule(Chem.AddHs(mol)) # doctest: +SKIP
            >>>
            >>> # Generate 10 conformers
            >>> gen = RDKitGeneration(num_conformers=10, random_seed=42) # doctest: +SKIP
            >>> structures, props = gen.operation(rdmol) # doctest: +SKIP
            >>> print(f"Generated {len(structures)} conformers") # doctest: +SKIP
        """
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
        if self.num_conformers == 1:
            return molecules[0], None
        else:
            return molecules, None
