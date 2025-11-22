"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers.torchsim.fairchem import FairChemTorchSimOptimizer
from jfchemistry.packing import PackmolPacking
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(head="C[*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="C[*:1]")

finite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=180,
    chain_length=10,
).make(polymer_job.output.structure)

packing = PackmolPacking(
    packing_mode="box",
    num_molecules=10,
    density=0.2,  # g/cm^3
).make(finite_chain_job.output.structure)

opt = FairChemTorchSimOptimizer(
    model="uma-s-1",
    task="omol",
    device="cuda",
    optimizer="FIRE",
).make(packing.output.structure)

flow = Flow(
    [
        polymer_job,
        finite_chain_job,
        packing,
        opt,
    ]
)

response = run_locally(flow)
