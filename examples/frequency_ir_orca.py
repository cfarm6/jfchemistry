"""Example: ORCA frequency/IR workflow."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.orca import ORCACalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.workflows.frequency_ir import FrequencyIRORCAWorkflow

SMILES = "CCO"  # ethanol


def main() -> None:
    """Build and run ORCA-specific frequency/IR workflow."""
    smiles_job = Smiles(remove_salts=False).make(input=SMILES)
    structure_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    orca_calc = ORCACalculator(
        xc_functional="B3LYP",
        basis_set="def2-SVP",
        cores=4,
        additional_keywords=["FREQ"],
    )

    freq_job = FrequencyIRORCAWorkflow(
        calculator=orca_calc,
        temperature=298.15,
        sigma_cm1=20.0,
    ).make(structure_job.output.structure)

    flow = Flow(
        [smiles_job, structure_job, freq_job],
        name="ORCA Frequency/IR Workflow",
    )
    responses = run_locally(flow)
    print("Ran flow with", len(responses), "responses")


if __name__ == "__main__":
    main()
