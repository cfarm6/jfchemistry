"""Example workflow for combined Nelson's four-point method with AimNet2 + ASE BFGS."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.workflows import NelsonsFourPointMethod

DONOR_SMILES = "c1ccccc1N"  # aniline
ACCEPTOR_SMILES = "O=C1NC(=O)C=CC1=O"  # maleimide

base_optimizer = ORCAOptimizer(steps=500, xc_functional="R2SCAN_3C", cores=16)

donor_smiles_job = Smiles(remove_salts=False).make(input=DONOR_SMILES)
donor_structure_job = RDKitGeneration(num_conformers=1).make(donor_smiles_job.output.structure)

acceptor_smiles_job = Smiles(remove_salts=False).make(input=ACCEPTOR_SMILES)
acceptor_structure_job = RDKitGeneration(num_conformers=1).make(
    acceptor_smiles_job.output.structure
)

nelson_job = NelsonsFourPointMethod(
    optimizer=base_optimizer,
).make(
    donor=donor_structure_job.output.structure,
    acceptor=acceptor_structure_job.output.structure,
)

flow = Flow(
    [
        donor_smiles_job,
        donor_structure_job,
        acceptor_smiles_job,
        acceptor_structure_job,
        nelson_job,
    ],
    name="Nelsons Four Point Method (SMILES + AimNet2 ASE BFGS)",
)

response = run_locally(flow)
