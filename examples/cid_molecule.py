"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import ORBCalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.utilities import CombineMolecules, RotateMolecule, SaveToDisk, TranslateMolecule

# Select Calculator
calculator = ORBCalculator(
    model="orb-v3-conservative-inf-omat", compile=True, d3_correction=True, device="cuda"
)


# Pull molecules from PubChem by CID
def generate_structure(cid):
    """Generate a structure from a PubChem CID."""
    jobs = []
    jobs.append(PubChemCID().make(input=cid))
    jobs.append(RDKitGeneration().make(jobs[-1].output.structure))
    jobs.append(ASEOptimizer(calculator=calculator).make(jobs[-1].output.structure))
    jobs.append(RotateMolecule(mode="principal_axes").make(jobs[-1].output.structure))
    return jobs


jobs1 = generate_structure(2244)
jobs2 = generate_structure(12)

translate_job = TranslateMolecule(mode="vector", translation=[0, 0, 10]).make(
    jobs1[-1].output.structure
)

combine_job = CombineMolecules().make([translate_job.output.structure, jobs2[-1].output.structure])

final_opt_job = ASEOptimizer(calculator=calculator).make(combine_job.output.structure)

save_job = SaveToDisk(filename="final_opt.xyz").make(final_opt_job.output.structure)
flow = Flow(
    [
        *jobs1,
        *jobs2,
        combine_job,
        translate_job,
        final_opt_job,
        save_job,
    ]
)
response = run_locally(flow)
