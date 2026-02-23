"""Example workflow for donor-acceptor free energy difference with AimNet2 + ASE BFGS."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.workflows import FreeEnergyDifferenceWorkflow

# Example donor/acceptor SMILES
DONOR_SMILES = "c1ccccc1N"  # aniline
ACCEPTOR_SMILES = "O=C1NC(=O)C=CC1=O"  # maleimide

base_optimizer = ASEOptimizer(
    calculator=AimNet2Calculator(model="aimnet2_2025"),
    optimizer="BFGS",
    fmax=0.02,
    steps=500,
    name="AimNet2 ASE BFGS",
)

# Donor input pipeline
donor_smiles_job = Smiles(remove_salts=False).make(input=DONOR_SMILES)
donor_structure_job = RDKitGeneration(num_conformers=1).make(donor_smiles_job.output.structure)

# Acceptor input pipeline
acceptor_smiles_job = Smiles(remove_salts=False).make(input=ACCEPTOR_SMILES)
acceptor_structure_job = RDKitGeneration(num_conformers=1).make(
    acceptor_smiles_job.output.structure
)

# Free energy difference workflow (defaults: donor final charge = +1, acceptor final charge = -1)
free_energy_job = FreeEnergyDifferenceWorkflow(
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
        free_energy_job,
    ],
    name="Free Energy Difference (SMILES + AimNet2 ASE BFGS)",
)

run_locally(flow)
