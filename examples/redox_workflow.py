"""Example: redox workflow from a single Molecule input."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.single_point.ase import ASESinglePoint
from jfchemistry.workflows.redox import RedoxPropertyWorkflow

SMILES = "c1ccccc1N"  # aniline


def main() -> None:
    """Build and run redox workflow using optimizer + single-point makers."""
    smiles_job = Smiles(remove_salts=False).make(input=SMILES)
    structure_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    calculator = AimNet2Calculator(model="aimnet2_2025")
    optimizer = ASEOptimizer(
        calculator=calculator,
        optimizer="BFGS",
        fmax=0.02,
        steps=300,
        name="AimNet2 ASE BFGS",
    )
    single_point = ASESinglePoint(calculator=calculator)

    redox_job = RedoxPropertyWorkflow(
        optimizer=optimizer,
        single_point=single_point,
    ).make(structure_job.output.structure)

    flow = Flow(
        [smiles_job, structure_job, redox_job],
        name="Redox Workflow (single molecule + AimNet2)",
    )

    responses = run_locally(flow)

    print("Ran flow with", len(responses), "responses")


if __name__ == "__main__":
    main()
