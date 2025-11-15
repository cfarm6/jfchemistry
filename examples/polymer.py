"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers import TBLiteOptimizer
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(head="C[*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="C[*:1]")

infinite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=np.array([180] * 4) + np.random.randn(4) * 10,
    chain_length=4,
).make(polymer_job.output.structure)

opt_job = TBLiteOptimizer(
    # model="orb-v3-direct-omol",  # ORB OMol Model
    optimizer="FIRE",  # optimizer
    unit_cell_optimizer="ExpCellFilter",  # unit cell optimizer
    trajectory="opt.traj",  # trajectory
).make(infinite_chain_job.output.structure)

flow = Flow([polymer_job, infinite_chain_job, opt_job])

run_locally(flow)
