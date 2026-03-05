"""Example of using the polymer nodes to run a 21-step MD simulation."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from numpy.random import choice

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.calculators.torchsim import OrbCalculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.molecular_dynamics.torchsim import (
    TorchSimMolecularDynamicsNPTNoseHoover,
    TorchSimMolecularDynamicsNVTNoseHoover,
)
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.packing.packmol import PackmolPacking
from jfchemistry.polymers import ExtractPolymerChains, GenerateFinitePolymerChain
from jfchemistry.utilities import SaveToDisk

chain_length = 30
number_chains = 10
finite_chain_jobs = []
finite_chain_structures = []
ase_calculator = AimNet2Calculator(model="aimnet2_2025")
torchsim_calculator = OrbCalculator(model="orb_v3_conservative_20_omat", device="cuda")
# BEGIN WORKFLOW

polymer_job = PolymerInput().make(
    head="[Si](C)(C)(C)O[*:1]", monomer="[*:1][Si](C)(C)O[*:2]", tail="[*:2][Si](C)(C)(C)"
)

for _ in range(number_chains):
    rotation_angles = choice([180, 240, -240], size=chain_length, p=[0.6, 0.2, 0.2]).tolist()
    job = GenerateFinitePolymerChain(dihedral_angles=rotation_angles, monomer_dihedral=0.0).make(
        polymer_job.output.structure
    )
    finite_chain_jobs.append(job)
    opt_job = ASEOptimizer(calculator=ase_calculator).make(job.output.structure)
    finite_chain_jobs.append(opt_job)
    finite_chain_structures.append(opt_job.output.structure)


pack_job = PackmolPacking(
    packing_mode="box",
    num_molecules=[1] * number_chains,
    density=0.1,
).make(finite_chain_structures)

opt_job = ASEOptimizer(calculator=ase_calculator, trajectory="opt.traj", logfile="opt.log").make(
    pack_job.output.structure
)

# ####### CYCLE 1 #######

nvt_job_1 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=100,
    timestep=1.0,
    tau=100.0,
    log_interval=500.0,
    calculator=torchsim_calculator,
    log_temperature=True,
    log_potential_energy=True,
    logfile="nvt_1",
    log_trajectory=True,
).make(opt_job.output.structure)

nvt_job_2 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=50_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_trajectory=True,
    log_interval=500.0,
    logfile="nvt_2",
    tau=100.0,
).make(nvt_job_1.output.structure)

npt_job_3 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=1.0,
    duration=50_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=500.0,
    log_trajectory=True,
    log_volume=False,
    logfile="npt_3",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_2.output.structure)

# ####### CYCLE 2 #######

nvt_job_4 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=50_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_4",
    log_trajectory=True,
    tau=100.0,
).make(npt_job_3.output.structure)

nvt_job_5 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=100_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_5",
    tau=100.0,
).make(nvt_job_4.output.structure)

npt_job_6 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=30_000.0,
    duration=50_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_6",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_5.output.structure)

# ####### CYCLE 3 #######

nvt_job_7 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=50_000,
    log_interval=1_000.0,
    logfile="nvt_7",
    tau=100.0,
).make(npt_job_6.output.structure)

nvt_job_8 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=50_000,
    log_interval=1_000.0,
    logfile="nvt_7",
    tau=100.0,
).make(nvt_job_7.output.structure)

npt_job_9 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=50_000,
    duration=50_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_9",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_8.output.structure)

# ####### CYCLE 4 #######

nvt_job_10 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=500_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_10",
    tau=100.0,
).make(npt_job_9.output.structure)

nvt_job_11 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=100_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_11",
    tau=100.0,
).make(nvt_job_10.output.structure)

npt_job_12 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=25_000,
    duration=5_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_12",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_11.output.structure)

# ####### CYCLE 5 #######

nvt_job_13 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=500_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_13",
    tau=100.0,
).make(npt_job_12.output.structure)

nvt_job_14 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=10_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_14",
    tau=100.0,
).make(nvt_job_13.output.structure)

npt_job_15 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=5_000,
    calculator=torchsim_calculator,
    duration=5_000,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_15",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_14.output.structure)

# ####### CYCLE 6 #######

nvt_job_16 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=500_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_16",
    tau=100.0,
).make(npt_job_15.output.structure)

nvt_job_17 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=10_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_17",
    tau=100.0,
).make(nvt_job_16.output.structure)

npt_job_18 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=500,
    duration=5_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_18",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_17.output.structure)

# ####### CYCLE 7 #######

nvt_job_19 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=600.0,
    duration=1_000_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_19",
    tau=100.0,
).make(npt_job_18.output.structure)

nvt_job_20 = TorchSimMolecularDynamicsNVTNoseHoover(
    temperature=300.0,
    duration=10_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    logfile="nvt_20",
    tau=100.0,
).make(nvt_job_19.output.structure)

npt_job_21 = TorchSimMolecularDynamicsNPTNoseHoover(
    temperature=300.0,
    external_pressure=1,
    duration=800_000,
    calculator=torchsim_calculator,
    timestep=1.0,
    log_temperature=True,
    log_trajectory=True,
    log_potential_energy=True,
    log_interval=1_000.0,
    log_volume=False,
    logfile="npt_21",
    t_tau=100.0,
    b_tau=1000.0,
).make(nvt_job_20.output.structure)

save_to_disk_job = SaveToDisk(filename="pdms_polymer.xyz").make(npt_job_21.output.structure)

extract_chains_job = ExtractPolymerChains().make(npt_job_21.output.structure)

save_to_disk_job = SaveToDisk(filename="pdms_polymer_chains.xyz").make(
    extract_chains_job.output.structure
)
flow = Flow(
    [
        polymer_job,
        *finite_chain_jobs,
        pack_job,
        opt_job,
        extract_chains_job,
        nvt_job_1,
        nvt_job_2,
        npt_job_3,
        nvt_job_4,
        nvt_job_5,
        npt_job_6,
        nvt_job_7,
        nvt_job_8,
        npt_job_9,
        nvt_job_10,
        nvt_job_11,
        npt_job_12,
        nvt_job_13,
        nvt_job_14,
        npt_job_15,
        nvt_job_16,
        nvt_job_17,
        npt_job_18,
        nvt_job_19,
        nvt_job_20,
        npt_job_21,
        save_to_disk_job,
    ]
)

run_locally(flow)
