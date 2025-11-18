"""Example of using the PubChemCID node to get a molecule from PubChem."""

import numpy as np
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators import ORBModelCalculator
from jfchemistry.conformers import MMMCConformers
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

chain_length = 32
rotation_angles = np.array([180] * (chain_length) + np.random.randn(chain_length) * 10)


polymer = PolymerInput().make(head="C[*:0]", monomer="[*:0]CC(C1=CC=CC=N1)[*:1]", tail="C[*:1]")

generate_structure = GenerateFinitePolymerChain(
    chain_length=chain_length,
    rotation_angles=rotation_angles,
    head_angle=180.0,
    tail_angle=180.0,
    num_conformers=100,
).make(polymer.output.structure)


calc = ORBModelCalculator(model="orb-v3-direct-20-omat", device="cuda", compile=True)

conformers = MMMCConformers(
    angle_step=10.0,
).make(generate_structure.output.structure, calc)

flow = Flow(
    [
        polymer,
        generate_structure,
        conformers,
    ]
)

response = run_locally(flow)
