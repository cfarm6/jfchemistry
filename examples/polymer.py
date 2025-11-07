"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers import TBLiteOptimizer
from jfchemistry.polymers.generation import PolymerInfiniteChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(monomer="[1*]C(F)(F)[2*]")

infinite_chain_job = PolymerInfiniteChain(
    rotation_angle=60,
    chain_length=6,
).make(polymer_job.output.structure)

opt_job = TBLiteOptimizer(
    optimizer="FIRE", unit_cell_optimizer="ExpCellFilter", trajectory="opt.traj"
).make(infinite_chain_job.output.structure)

flow = Flow(
    [
        polymer_job,
        infinite_chain_job,  # opt_job
    ]
)

run_locally(flow)
