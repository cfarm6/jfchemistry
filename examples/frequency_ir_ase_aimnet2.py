"""Example: ASE frequency/IR workflow using AimNet2 calculator."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.workflows.frequency_ir import FrequencyIRASEWorkflow

SMILES = "CCO"  # ethanol


def main() -> None:
    """Build and run ASE frequency/IR workflow with AimNet2 backend."""
    smiles_job = Smiles(remove_salts=False).make(input=SMILES)
    structure_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    freq_job = FrequencyIRASEWorkflow(
        calculator=AimNet2Calculator(model="aimnet2_2025"),
        temperature=298.15,
        sigma_cm1=20.0,
        displacement=0.01,
        nfree=2,
    ).make(structure_job.output.structure)

    flow = Flow(
        [smiles_job, structure_job, freq_job],
        name="ASE Frequency/IR Workflow (AimNet2)",
    )
    responses = run_locally(flow)
    print("Ran flow with", len(responses), "responses")


if __name__ == "__main__":
    main()
