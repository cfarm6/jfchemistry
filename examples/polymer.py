"""Example of using the polymer nodes to run a 21-step MD simulation."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from numpy.random import choice

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.polymers import GenerateFinitePolymerChain
from jfchemistry.utilities import SaveToDisk

chain_length = 20
number_chains = 5
calculator = AimNet2Calculator(model="aimnet2")

# BEGIN WORKFLOW

polymer_job = PolymerInput().make(head="C[*:1]", monomer="[*:1]C[*:2]", tail="[*:2]C")

rotation_angles = choice([180, 240, -240], size=chain_length, p=[0.6, 0.2, 0.2]).tolist()
job = GenerateFinitePolymerChain(dihedral_angles=rotation_angles, monomer_dihedral=0.0).make(
    polymer_job.output.structure
)


opt_job = ASEOptimizer(
    calculator=calculator, trajectory="pe_short_chain.traj", logfile="pe_short_chain.log", fmax=0.01
).make(job.output.structure)

save_to_disk_job = SaveToDisk(filename="pe_short_chain.xyz").make(opt_job.output.structure)

flow = Flow(
    [
        polymer_job,
        job,
        opt_job,
        save_to_disk_job,
    ]
)

run_locally(flow)
