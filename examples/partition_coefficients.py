"""Example: partition coefficient workflow (new molecule-first interface).

This example assumes you already provide a 3D molecule (here via XYZ file).
"""

from pathlib import Path

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from pymatgen.core.structure import Molecule

from jfchemistry.conformers import CRESTConformers
from jfchemistry.filters import EnergyFilter, PrismPrunerFilter
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.single_point import ORCASinglePointCalculator
from jfchemistry.workflows.partition_coefficient import PartitionCoefficientWorkflow


def load_structure() -> Molecule:
    """Load a 3D molecule from local XYZ file."""
    xyz_path = Path("examples/ibuprofen.xyz")
    if not xyz_path.exists():
        raise FileNotFoundError(
            f"Expected input structure at {xyz_path}. Provide a 3D XYZ molecule first."
        )
    return Molecule.from_file(str(xyz_path))


def main() -> None:
    """Run partition workflow from one 3D structure across two phases."""
    structure = load_structure()

    conformer_generator = CRESTConformers(
        threads=8,
        crest_executable="crest",
        energy_window=6.0,
    )

    optimizer = ORCAOptimizer(
        cores=8,
        xc_functional="r2scan-3c",
        basis_set="def2-mTZVPP",
        solvent=None,  # workflow assigns per-phase solvent
    )

    single_point = ORCASinglePointCalculator(
        cores=8,
        xc_functional="r2scan-3c",
        basis_set="def2-mTZVPP",
        solvent=None,  # workflow assigns per-phase solvent
    )

    workflow = PartitionCoefficientWorkflow(
        name="Partition Coefficient Workflow (molecule-first)",
        threads=8,
        temperature=298.15,
        alpha_phase="OCTANOL",
        beta_phase="WATER",
        tautomer_generator=None,
        conformer_generator=conformer_generator,
        geometry_optimizer=optimizer,
        single_point=single_point,
        conformer_energy_filter=EnergyFilter(energy_window=6.0),
        conformer_structural_filter=PrismPrunerFilter(max_n_confs=30),
        optimized_energy_filter=EnergyFilter(energy_window=4.0),
        optimized_structural_filter=PrismPrunerFilter(max_n_confs=15),
    )

    partition_job = workflow.make(structure)
    flow = Flow([partition_job], name="partition_coefficient_new_interface")

    run_locally(flow, create_folders=True)


if __name__ == "__main__":
    main()
