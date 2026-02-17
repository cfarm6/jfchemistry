"""Example of using the polymer nodes to run a 21-step MD simulation."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally
from numpy.random import choice

from jfchemistry.calculators.ase import AimNet2Calculator
from jfchemistry.inputs import PolymerInput
from jfchemistry.packing.packmol import PackmolPacking
from jfchemistry.polymers import GenerateFinitePolymerChain

chain_length = 198
number_chains = 20
finite_chain_jobs = []
finite_chain_structures = []
calculator = AimNet2Calculator(model="aimnet2_2025")

# BEGIN WORKFLOW

polymer_job = PolymerInput().make(head="C[*:1]", monomer="[*:1]C[*:2]", tail="C[*:2]")

for _ in range(number_chains):
    rotation_angles = choice([180, 240, -240], size=chain_length, p=[0.6, 0.2, 0.2]).tolist()
    job = GenerateFinitePolymerChain(dihedral_angles=rotation_angles, monomer_dihedral=0.0).make(
        polymer_job.output.structure
    )
    finite_chain_jobs.append(job)
    finite_chain_structures.append(job.output.structure)

pack_job = PackmolPacking(
    packing_mode="box",
    num_molecules=[1] * number_chains,
    density=0.05,
).make(finite_chain_structures)

# opt_job = ASEOptimizer(calculator=calculator).make(pack_job.output.structure)

# ####### CYCLE 1 #######

# nvt_job_1 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=50_000,
#     timestep=1.0,
#     log_interval=500.0,
#     ttime=50.0,
#     calculator=calculator,
#     log_temperature=True,
#     log_potential_energy=True,
#     logfile="nvt_1",
#     log_trajectory=True,
# ).make(opt_job.output.structure)

# nvt_job_2 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_trajectory=True,
#     log_interval=500.0,
#     logfile="nvt_2",
#     ttime=50.0,
# ).make(nvt_job_1.output.structure)

# npt_job_3 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=1.0,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=500.0,
#     log_trajectory=True,
#     log_volume=False,
#     logfile="npt_3",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_2.output.structure)

# # ####### CYCLE 2 #######

# nvt_job_4 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_4",
#     log_trajectory=True,
#     ttime=50.0,
# ).make(npt_job_3.output.structure)

# nvt_job_5 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=100_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_5",
#     ttime=50.0,
# ).make(nvt_job_4.output.structure)

# npt_job_6 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=30_000.0,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_6",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_5.output.structure)

# # ####### CYCLE 3 #######

# nvt_job_7 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_7",
#     ttime=50.0,
# ).make(npt_job_6.output.structure)

# nvt_job_8 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=100_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_8",
#     ttime=50.0,
# ).make(nvt_job_7.output.structure)

# npt_job_9 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=50_000,
#     duration=50_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_9",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_8.output.structure)

# # ####### CYCLE 4 #######

# nvt_job_10 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=500_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_10",
#     ttime=50.0,
# ).make(npt_job_9.output.structure)

# nvt_job_11 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=100_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_11",
#     ttime=50.0,
# ).make(nvt_job_10.output.structure)

# npt_job_12 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=25_000,
#     duration=5_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_12",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_11.output.structure)

# # ####### CYCLE 5 #######

# nvt_job_13 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=500_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_13",
#     ttime=50.0,
# ).make(npt_job_12.output.structure)

# nvt_job_14 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=10_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_14",
#     ttime=50.0,
# ).make(nvt_job_13.output.structure)

# npt_job_15 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=5_000,
#     calculator=calculator,
#     duration=5_000,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_15",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_14.output.structure)

# # ####### CYCLE 6 #######

# nvt_job_16 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=500_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_16",
#     ttime=50.0,
# ).make(npt_job_15.output.structure)

# nvt_job_17 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=10_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_17",
#     ttime=50.0,
# ).make(nvt_job_16.output.structure)

# npt_job_18 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=500,
#     duration=5_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_18",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_17.output.structure)

# # ####### CYCLE 7 #######

# nvt_job_19 = ASEMolecularDynamicsNVTBussi(
#     temperature=600.0,
#     duration=1_000_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_19",
#     ttime=50.0,
# ).make(npt_job_18.output.structure)

# nvt_job_20 = ASEMolecularDynamicsNVTBussi(
#     temperature=300.0,
#     duration=10_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     logfile="nvt_20",
#     ttime=50.0,
# ).make(nvt_job_19.output.structure)

# npt_job_21 = ASEMolecularDynamicsNPTBerendsen(
#     temperature=300.0,
#     external_pressure=1,
#     duration=800_000,
#     calculator=calculator,
#     timestep=1.0,
#     log_temperature=True,
#     log_trajectory=True,
#     log_potential_energy=True,
#     log_interval=1_000.0,
#     log_volume=False,
#     logfile="npt_21",
#     ttime=50.0,
#     ptime=500.0,
# ).make(nvt_job_20.output.structure)

flow = Flow(
    [
        polymer_job,
        *finite_chain_jobs,
        pack_job,
        # opt_job,
        # nvt_job_1,
        # nvt_job_2,
        # npt_job_3,
        # nvt_job_4,
        # nvt_job_5,
        # npt_job_6,
        # nvt_job_7,
        # nvt_job_8,
        # npt_job_9,
        # nvt_job_10,
        # nvt_job_11,
        # npt_job_12,
        # nvt_job_13,
        # nvt_job_14,
        # npt_job_15,
        # nvt_job_16,
        # nvt_job_17,
        # npt_job_18,
        # nvt_job_19,
        # nvt_job_20,
        # npt_job_21,
    ]
)

run_locally(flow)
