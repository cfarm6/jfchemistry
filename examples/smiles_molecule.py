"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.inputs import Smiles

smiles = Smiles().make("C1CCCCC1")

flow = Flow([smiles])

response = run_locally(flow)

print(response)
