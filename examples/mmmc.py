"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim import OrbCalculator
from jfchemistry.conformers import MMMCConformers
from jfchemistry.optimizers.torchsim import TorchSimOptimizer
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

chain_length = 2
rotation_angles = np.array([180] * (chain_length) + np.random.randn(chain_length) * 10)


polymer = PolymerInput().make(head="C[*:0]", monomer="[*:0]C(F)(F)C(F)(F)[*:1]", tail="C[*:1]")

generate_structure = GenerateFinitePolymerChain(
    chain_length=chain_length,
    rotation_angles=rotation_angles.tolist(),
    head_angle=180.0,
    tail_angle=180.0,
    num_conformers=100,
).make(polymer.output.structure)

optimizer = TorchSimOptimizer(
    calculator=OrbCalculator(
        device="cuda", model="orb_v3_conservative_inf_omat", compile=True, compute_stress=True
    ),
)

conformers = MMMCConformers(
    optimizer=optimizer,
    angle_step=10.0,
).make(generate_structure.output.structure)

flow = Flow(
    [
        polymer,
        generate_structure,
        conformers,
    ]
)

response = run_locally(flow)
