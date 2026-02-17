"""Example of using the polymer nodes to run a 21-step MD simulation."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from numpy.random import choice

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.polymers import GenerateFinitePolymerChain

chain_length = 50
number_chains = 5
finite_chain_jobs = []
finite_chain_structures = []
calculator = AimNet2Calculator(model="aimnet2_2025")

# BEGIN WORKFLOW

polymer_job = PolymerInput().make(
    head="[Si](C)(C)(C)O[*:1]", monomer="[*:1][Si](C)(C)O[*:2]", tail="[*:2][Si](C)(C)(C)"
)

rotation_angles = choice([180, 240, -240], size=chain_length, p=[0.6, 0.2, 0.2]).tolist()
job = GenerateFinitePolymerChain(dihedral_angles=rotation_angles, monomer_dihedral=0.0).make(
    polymer_job.output.structure
)
finite_chain_jobs.append(job)
finite_chain_structures.append(job.output.structure)


flow = Flow(
    [
        polymer_job,
        *finite_chain_jobs,
        # pack_job,
        # opt_job,
        # nvt_job_1,
        # nvt_job_2,
        # npt_job_3,
        # nvt_job_4,
        # nvt_job_5,
        # npt_job_6,
        # nvt_job_7,
        # nvt_job_8,
        # npt_job_9,
        # nvt_job_10,
        # nvt_job_11,
        # npt_job_12,
        # nvt_job_13,
        # nvt_job_14,
        # npt_job_15,
        # nvt_job_16,
        # nvt_job_17,
        # npt_job_18,
        # nvt_job_19,
        # nvt_job_20,
        # npt_job_21,
    ]
)

run_locally(flow)
