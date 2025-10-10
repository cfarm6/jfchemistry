"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.generation.rdkit_generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID
from jfchemistry.optimizers.tblite_optimizer import TBLiteOptimizer

pubchem_cid = PubChemCID().make(21688863)
generate_structure = RDKitGeneration(basin_thresh=3.4).make(pubchem_cid.output["structure"])
optimize_structure = TBLiteOptimizer(method="GFN2-xTB", verbosity=3).make(
    generate_structure.output["structure"]
)
flow = Flow([pubchem_cid, generate_structure, optimize_structure])

response = run_locally(flow)

print(response)
