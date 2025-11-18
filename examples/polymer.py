"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.optimizers import TBLiteOptimizer
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput
from jfchemistry.single_point import ORCASinglePointCalculator

chain_length = 4
rotation_angles = np.array([180] * (chain_length - 2) + np.random.randn(chain_length - 2) * 10)

polymer_job = PolymerInput().make(head="C[*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="C[*:1]")

infinite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=rotation_angles,
    chain_length=chain_length - 2,
).make(polymer_job.output.structure)

opt_job = TBLiteOptimizer(
    # model="orb-v3-direct-omol",  # ORB OMol Model
    optimizer="FIRE",  # optimizer
    trajectory="opt.traj",  # trajectory
).make(infinite_chain_job.output.structure)

pr_job = ORCASinglePointCalculator(
    basis_set="DEF2_SVP",
    xc_functional="B3LYP",
    participation_ratio=True,
    homo_threshold=0.5,
    lumo_threshold=0.5,
    cores=16,
).make(opt_job.output.structure)

flow = Flow([polymer_job, infinite_chain_job, opt_job, pr_job])

run_locally(flow)
