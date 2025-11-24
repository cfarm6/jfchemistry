"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim.orb_calculator import OrbCalculator
from jfchemistry.molecular_dynamics.torchsim.nvt.nvt_nose_hoover import (
    TorchSimMolecularDynamicsNVTNoseHoover,
)
from jfchemistry.packing import PackmolPacking
from jfchemistry.polymers.generation import GenerateFinitePolymerChain
from jfchemistry.polymers.input import PolymerInput

polymer_job = PolymerInput().make(
    head="[Si](C)(C)(C)O[*:0]", monomer="[*:0][Si](C)(C)O[*:1]", tail="[*:1][Si](C)(C)(C)"
)

finite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=180,
    chain_length=3,
).make(polymer_job.output.structure)

packing = PackmolPacking(
    packing_mode="box",
    num_molecules=2,
    density=0.5,  # g/cm^3
).make(finite_chain_job.output.structure)

nvt = TorchSimMolecularDynamicsNVTNoseHoover(
    duration=10.0,  # fs
    timestep=1.0,  # fs
    temperature=300.0,  # K
    autobatcher=False,
    logfile="trajectory_nvt",
    log_interval=1.0,  # fs
    log_potential_energy=True,
    log_kinetic_energy=True,
    log_temperature=True,
    log_volume=True,
    log_pressure=False,
).make(
    [packing.output.structure, packing.output.structure],
    OrbCalculator(
        device="cuda", model="orb_v3_conservative_inf_omat", compile=True, compute_stress=True
    ),
)


flow = Flow([polymer_job, finite_chain_job, packing, nvt])

response = run_locally(flow)
