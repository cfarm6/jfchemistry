"""Example of using the polymer nodes to run a 21-step MD simulation."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from numpy.random import choice

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.packing.packmol import PackmolPacking
from jfchemistry.polymers import GenerateFinitePolymerChain

chain_length = 50
number_chains = 20
finite_chain_jobs = []
finite_chain_structures = []
calculator = AimNet2Calculator(model="aimnet2_2025")

# BEGIN WORKFLOW

polymer_job = PolymerInput().make(head="C[*:1]", monomer="[*:1]C[*:2]", tail="C[*:2]")

for _ in range(number_chains):
    rotation_angles = choice([180, 240, -240], size=chain_length, p=[0.6, 0.2, 0.2]).tolist()
    job = GenerateFinitePolymerChain(dihedral_angles=rotation_angles, monomer_dihedral=0.0).make(
        polymer_job.output.structure
    )
    finite_chain_jobs.append(job)
    finite_chain_structures.append(job.output.structure)

pack_job = PackmolPacking(
    packing_mode="box",
    num_molecules=[1] * number_chains,
    density=0.1,
).make(finite_chain_structures)

flow = Flow(
    [
        polymer_job,
        *finite_chain_jobs,
        pack_job,
    ]
)

run_locally(flow)
