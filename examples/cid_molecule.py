"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.single_point import PySCFGPUSinglePoint

# Get all calculator classes
pubchem_cid = Smiles().make("C(CO)Cl")

generate_structure = RDKitGeneration(num_conformers=2).make(pubchem_cid.output.structure)

# deprotonate = CRESTDeprotonation().make(generate_structure.output.structure[0])
# protonate = CRESTProtonation().make(deprotonate.output.structure[0])
# mmmc_conformer = MMMCConformers(optimizer=ASEOptimizer(calculator=TBLiteCalculator())).make(
#     generate_structure.output.structure[0]
# )

# prism_pruner = PrismPrunerFilter(energy_threshold=2.0).make(
#     mmmc_conformer.output.structure, properties=mmmc_conformer.output.properties
# )

# energy_filter = EnergyFilter(threshold=0.1).make(
#     prism_pruner.output.structure, properties=prism_pruner.output.properties
# )

# opt_ase = ASEOptimizer(
#     calculator=TBLiteCalculator(),
# ).make(energy_filter.output.structure)

# opt_torchsim = TorchSimOptimizer(
#     calculator=OrbCalculator(model="orb_v3_direct_20_omat"),
# ).make(opt_ase.output.structure)

sp = PySCFGPUSinglePoint(xc_functional="B3LYP", basis_set="def2-svp", joltqc=True).make(
    generate_structure.output.structure[0]
)

flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        sp,
    ]
)

response = run_locally(flow)
