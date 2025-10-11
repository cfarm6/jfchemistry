"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.conformers.crest import CRESTConformers
from jfchemistry.generation.rdkit_generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID
from jfchemistry.modification.crest_deprotonation import CRESTDeprotonation
from jfchemistry.modification.crest_protonation import CRESTProtonation

pubchem_cid = PubChemCID().make(21688863)

generate_structure = RDKitGeneration(num_conformers=2).make(pubchem_cid.output["structure"])

# optimize_structure = AimNet2Optimizer().make(generate_structure.output["structure"])

crest_conformers = CRESTConformers(
    calculation_dynamics_method="gfnff", calculation_energy_method="gfnff"
).make(generate_structure.output["structure"])

deprotonation = CRESTDeprotonation().make(crest_conformers.output["structure"])
protonation = CRESTProtonation(ion="Na+").make(deprotonation.output["structure"])
flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        crest_conformers,
        deprotonation,
        protonation,
    ]
)

response = run_locally(flow)

print(response)
