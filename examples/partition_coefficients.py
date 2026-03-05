"""Example: partition coefficient workflow (new molecule-first interface).

This script can start from either:
- a SMILES string, or
- a PubChem CID (resolved to isomeric SMILES),
then generates a 3D molecule and runs the partition workflow.
"""

from __future__ import annotations

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.conformers import CRESTConformers
from jfchemistry.filters import EnergyFilter, PrismPrunerFilter
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.single_point import ORCASinglePointCalculator
from jfchemistry.workflows.partition_coefficient import PartitionCoefficientWorkflow


def main() -> None:
    """Run partition workflow from SMILES or PubChem CID.

    Set exactly one of `smiles` or `pubchem_cid`.
    """
    smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # ibuprofen

    smiles_job = Smiles(remove_salts=False).make(input=smiles)

    generation_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    conformer_generator = CRESTConformers(
        threads=8,
        executable="crest",
        ewin=1.0,
    )

    optimizer = ORCAOptimizer(
        cores=8,
        xc_functional="PBE",
        basis_set="sto-3g",
        solvent=None,  # workflow assigns per-phase solvent
    )

    single_point = ORCASinglePointCalculator(
        cores=8,
        xc_functional="PBE",
        basis_set="sto-3g",
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
        conformer_energy_filter=EnergyFilter(threshold=6.0),
        conformer_structural_filter=PrismPrunerFilter(),
        optimized_energy_filter=EnergyFilter(threshold=4.0),
        optimized_structural_filter=PrismPrunerFilter(),
    )

    partition_job = workflow.make(generation_job.output.structure)

    flow = Flow(
        [smiles_job, generation_job, partition_job],
        name="partition_coefficient_new_interface",
    )

    run_locally(flow, create_folders=True)


if __name__ == "__main__":
    main()
