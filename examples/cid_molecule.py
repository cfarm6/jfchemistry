"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.inputs import PubChemCID

pubchem_cid = PubChemCID().make(12345)

flow = Flow([pubchem_cid])

response = run_locally(flow)

print(response)
