"""Example: conformer ensemble workflow from a single Molecule."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.single_point.ase import ASESinglePoint
from jfchemistry.workflows.conformer_ensemble import ConformerEnsembleWorkflow

SMILES = "CCO"  # ethanol


def main() -> None:
    """Build and run conformer ensemble workflow using RDKit + single point."""
    smiles_job = Smiles(remove_salts=False).make(input=SMILES)
    structure_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    conformer_generator = RDKitGeneration(num_conformers=5)
    single_point = ASESinglePoint(calculator=AimNet2Calculator(model="aimnet2_2025"))

    ensemble_job = ConformerEnsembleWorkflow(
        conformer_generator=conformer_generator,
        single_point=single_point,
        temperature=298.15,
    ).make(structure_job.output.structure)

    flow = Flow(
        [smiles_job, structure_job, ensemble_job],
        name="Conformer Ensemble Workflow (single molecule)",
    )
    responses = run_locally(flow)
    print("Ran flow with", len(responses), "responses")


if __name__ == "__main__":
    main()
