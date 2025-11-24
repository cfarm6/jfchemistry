"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim import OrbCalculator
from jfchemistry.polymers.generation import GenerateInfinitePolymerChain
from jfchemistry.polymers.input import PolymerInput
from jfchemistry.single_point.torchsim import TorchSimSinglePoint

chain_length = 10
rotation_angles = np.array([60] * (chain_length - 2) + np.random.randn(chain_length - 2) * 10)

polymer_job = PolymerInput().make(head="C[*:0]", monomer="C(C(F)(F)[*:1])[*:0]", tail="C[*:1]")

infinite_chain_job = GenerateInfinitePolymerChain(
    rotation_angles=rotation_angles,
    chain_length=chain_length - 2,
).make(polymer_job.output.structure)

opt_job = TorchSimSinglePoint(
    calculator=OrbCalculator(model="orb_v3_direct_20_omat", compile=True, compute_stress=True),
).make(infinite_chain_job.output.structure)

flow = Flow(
    [
        polymer_job,
        infinite_chain_job,
        opt_job,
        # pr_job
    ]
)

run_locally(flow)
