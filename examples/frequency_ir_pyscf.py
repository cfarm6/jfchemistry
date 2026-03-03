"""Example: PySCF frequency/IR workflow."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.pyscfgpu import PySCFCalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.workflows.frequency_ir import FrequencyIRPySCFWorkflow

SMILES = "CCO"  # ethanol


def main() -> None:
    """Build and run PySCF-specific frequency/IR workflow."""
    smiles_job = Smiles(remove_salts=False).make(input=SMILES)
    structure_job = RDKitGeneration(num_conformers=1).make(smiles_job.output.structure)

    pyscf_calc = PySCFCalculator(
        mode="cpu",
        xc_functional="b3lyp",
        basis_set="def2-svp",
        cores=4,
    )

    freq_job = FrequencyIRPySCFWorkflow(
        calculator=pyscf_calc,
        temperature=298.15,
        pressure_pa=101325.0,
        sigma_cm1=20.0,
    ).make(structure_job.output.structure)

    flow = Flow(
        [smiles_job, structure_job, freq_job],
        name="PySCF Frequency/IR Workflow",
    )
    responses = run_locally(flow)
    print("Ran flow with", len(responses), "responses")


if __name__ == "__main__":
    main()
