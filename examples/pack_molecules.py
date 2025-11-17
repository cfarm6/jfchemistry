"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers import ORBModelOptimizer
from jfchemistry.packing import PackmolPacking
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(head="C[*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="C[*:1]")

finite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=np.array([180] * 6) + np.random.randn(6) * 10,
    chain_length=6,
).make(polymer_job.output.structure)

packing = PackmolPacking(
    packing_mode="box",
    num_molecules=2,
    density=0.5,
).make(finite_chain_job.output.structure)

opt = ORBModelOptimizer(
    model="orb-v3-direct-20-omat",
    optimizer="FIRE",
    trajectory="opt.traj",
    logfile="opt.log",
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
