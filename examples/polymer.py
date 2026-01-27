"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jfchemistry.calculators.torchsim.fairchem_calculator import FairChemCalculator
from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.torchsim import OrbCalculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.molecular_dynamics.torchsim import (
    TorchSimMolecularDynamicsNPTNoseHoover,
    TorchSimMolecularDynamicsNVTNoseHoover,
)
from jfchemistry.optimizers.torchsim import TorchSimOptimizer
from jfchemistry.packing.packmol import PackmolPacking
from jfchemistry.polymers import GenerateFinitePolymerChain

# chain_length = 11
rotation_angles = [180] * 5 + [60] * 5 + [180] * 5

polymer_job = PolymerInput().make(
    head="[Si](C)(C)(C)O[*:1]", monomer="[*:1][Si](C)(C)O[*:2]", tail="[Si](C)(C)(C)[*:2]"
)

finite_chain_job = GenerateFinitePolymerChain(
    rotation_angles=rotation_angles, chain_length=len(rotation_angles)
).make(polymer_job.output.structure)

optimize_job = TorchSimOptimizer(
    calculator=OrbCalculator(device="cuda", model="orb_v3_direct_20_omat"),
).make(finite_chain_job.output.structure)

pack_job = PackmolPacking(
    packing_mode="box",
    num_molecules=20,
    density=0.1,  # g/cm^3
).make(optimize_job.output.structure)

nvt_job = TorchSimMolecularDynamicsNVTNoseHoover(
    calculator=FairChemCalculator(device="cuda"),
    duration=20,
    timestep=1.0,
    temperature=300.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=10.0,
    tau=10.0,
).make(pack_job.output.structure)

npt_job = TorchSimMolecularDynamicsNPTNoseHoover(
    calculator=FairChemCalculator(device="cuda"),
    duration=20,
    timestep=1.0,
    temperature=300.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=10.0,
    log_volume=True,
    b_tau=10.0,
    t_tau=10.0,
).make(pack_job.output.structure)

flow = Flow(
    [
        polymer_job,
        finite_chain_job,
        optimize_job,
        pack_job,
        nvt_job,
        npt_job,
    ]
)

run_locally(flow)
