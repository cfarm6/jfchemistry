"""Example of using the PubChemCID node to get a molecule from PubChem."""

from fireworks.core.launchpad import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from jobflow.managers.local import run_locally

from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID

# from jfchemistry.modification import CRESTDeprotonation, CRESTProtonation, CRESTTautomers
from jfchemistry.optimizers import AimNet2Optimizer, TBLiteOptimizer

pubchem_cid = PubChemCID().make(8003)

generate_structure = RDKitGeneration(num_conformers=2).make(pubchem_cid.output.structure)

optimize_structure = AimNet2Optimizer(optimizer="FIRE").make(generate_structure.output.structure)

optimize_structure2 = TBLiteOptimizer(method="GFN2-xTB", optimizer="FIRE").make(
    optimize_structure.output.structure
)

# crest_conformers = CRESTConformers(
#     calculation_dynamics_method="gfnff",
#     calculation_energy_method="gfnff",
# ).make(optimize_structure.output["structure"])

# deprotonation = CRESTDeprotonation(threads=16, energy_window=2.0).make(
#     crest_conformers.output["structure"]
# )

# protonation = CRESTProtonation(threads=16, energy_window=2.0).make(
#     deprotonation.output["structure"]
# )

# tautomers = CRESTTautomers(threads=16, energy_window=2.0).make(protonation.output["structure"])


flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        optimize_structure,
        optimize_structure2,
        # crest_conformers,
        # deprotonation,
        # protonation,
        # tautomers,
    ]
)

response = run_locally(flow)
print(response[optimize_structure.uuid])
wf = flow_to_workflow(flow)
lp = LaunchPad.from_file("my_launchpad.local.yaml")
lp.add_wf(wf)
