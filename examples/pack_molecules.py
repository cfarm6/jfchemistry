"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ORBModelOptimizer
from jfchemistry.packing import PackmolPacking

smiles = Smiles().make("C(C=O)Cl")

generate_structure = RDKitGeneration(num_conformers=1).make(smiles.output.structure)

packing = PackmolPacking(
    packing_mode="box",
    box_dimensions=(20, 20, 20),
    num_molecules=3,
).make(generate_structure.output.structure)

opt = ORBModelOptimizer(optimizer="FIRE").make(packing.output.structure)

flow = Flow(
    [
        smiles,
        generate_structure,
        packing,
        opt,
    ]
)

response = run_locally(flow)
