"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators import ORBModelCalculator
from jfchemistry.conformers import MMMCConformers
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer = PolymerInput().make(
    head="[Si](C)(C)(C)[*:0]", monomer="[*:0]O[Si](C)(C)[*:1]", tail="[*:1]O[Si](C)(C)C"
)
generate_structure = GenerateFinitePolymerChain(
    chain_length=64,
    rotation_angles=180.0,
    head_angle=180.0,
    tail_angle=180.0,
    num_conformers=100,
).make(polymer.output.structure)


calc = ORBModelCalculator(model="orb-v3-direct-omol", device="cuda", compile=True)

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
