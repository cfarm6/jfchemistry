"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

# from jfchemistry.single_point.torchsim import TorchSimSinglePoint
from jfchemistry.single_point.pyscfgpu import PySCFGPUSinglePoint

chain_length = 10
rotation_angles = np.array([60] * (chain_length - 2) + np.random.randn(chain_length - 2) * 10)

polymer_job = PolymerInput().make(
    head="[Si](C)(C)(C)O[*:0]", monomer="[*:0][Si](C)(C)O[*:1]", tail="[*:1][Si](C)(C)(C)"
)

finite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=rotation_angles.tolist(),
    chain_length=chain_length - 2,
).make(polymer_job.output.structure)

pyscf_gpu = PySCFGPUSinglePoint(
    basis_set="def2svp",
    xc_functional="r2scan",
    participation_ratio=True,
    homo_threshold=1.0,
    lumo_threshold=1.0,
).make(finite_chain_job.output.structure)

flow = Flow(
    [
        polymer_job,
        finite_chain_job,
        pyscf_gpu,
    ]
)

run_locally(flow)
