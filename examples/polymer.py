"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers import ORBModelOptimizer
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(
    head="[H][*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="[H][*:1]"
)

infinite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=180,
    chain_length=4,
).make(polymer_job.output.structure)

opt_job = ORBModelOptimizer(
    model="orb-v3-direct-omol",  # ORB OMol Model
    optimizer="FIRE",  # optimizer
    unit_cell_optimizer="ExpCellFilter",  # unit cell optimizer
    trajectory="opt.traj",  # trajectory
).make(infinite_chain_job.output.structure)

flow = Flow([polymer_job, infinite_chain_job, opt_job])

run_locally(flow)
