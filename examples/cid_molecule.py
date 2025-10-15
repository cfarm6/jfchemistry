"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.conformers import CRESTConformers
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID
from jfchemistry.modification import CRESTDeprotonation, CRESTTautomers
from jfchemistry.optimizers.aimnet2 import AimNet2Optimizer

pubchem_cid = PubChemCID().make(8003)

generate_structure = RDKitGeneration(num_conformers=1).make(pubchem_cid.output["structure"])

optimize_structure = AimNet2Optimizer(optimizer="QuasiNewton").make(
    generate_structure.output["structure"]
)

crest_conformers = CRESTConformers(
    calculation_dynamics_method="gfnff",
    calculation_energy_method="gfnff",
).make(generate_structure.output["structure"])

deprotonation = CRESTDeprotonation().make(crest_conformers.output["structure"])
protonation = CRESTTautomers().make(deprotonation.output["structure"])
flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        crest_conformers,
        deprotonation,
        protonation,
        optimize_structure,
    ]
)

response = run_locally(flow)

print(response)
