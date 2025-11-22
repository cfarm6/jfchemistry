"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim.orb_ts_calculator import OrbTSCalculator
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.molecular_dynamics.torchsim.npt.npt_langevin import (
    TorchSimMolecularDynamicsNPTLangevin,
)

pubchem_cid = Smiles().make("C(C=O)Cl")

generate_structure = RDKitGeneration(num_conformers=1).make(pubchem_cid.output.structure)


energies = TorchSimMolecularDynamicsNPTLangevin(
    duration=5.0,
    timestep=0.5,
    temperature=300.0,
    logfile="trajectory.h5",
    log_interval=1.0,
    log_potential_energy=True,
    log_kinetic_energy=True,
    log_temperature=True,
    log_volume=False,
).make(
    generate_structure.output.structure,
    OrbTSCalculator(device="cuda", model="orb_v3_conservative_inf_omat", compile=True),
)

flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        energies,
    ]
)

response = run_locally(flow)
