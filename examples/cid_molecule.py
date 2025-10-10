"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.generation.rdkit_generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID

pubchem_cid = PubChemCID().make(12345)
generate_structure = RDKitGeneration(basin_thresh=3.4).make(pubchem_cid.output["structure"])
flow = Flow([pubchem_cid, generate_structure])

response = run_locally(flow)

print(response)
