"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim import FairChemCalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.single_point.torchsim import TorchSimSinglePoint

pubchem_cid = Smiles().make("C(C=O)Cl")

generate_structure = RDKitGeneration(num_conformers=3).make(pubchem_cid.output.structure)

energies = TorchSimSinglePoint(
    calculator=FairChemCalculator(),
).make(generate_structure.output.structure)


flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        energies,
    ]
)

response = run_locally(flow)
