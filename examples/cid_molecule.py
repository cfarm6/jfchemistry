"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.conformers import CRESTConformers
from jfchemistry.filters.structural.prism_filter import PrismPrunerFilter
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.single_point import TBLiteSinglePointCalculator

pubchem_cid = Smiles().make("C(C=O)Cl")

generate_structure = RDKitGeneration(num_conformers=1).make(pubchem_cid.output.structure)

# optimize_structure = AimNet2Optimizer(optimizer="FIRE").make(generate_structure.output.structure)

# tautomers = CRESTTautomers(threads=16).make(optimize_structure.output.structure)

# single_point = AimNet2SinglePointCalculator().make(tautomers.output.structure)

# filter_energy = EnergyFilter(threshold=12.0).make(
#     single_point.output.structure, single_point.output.properties
# )

conformers = CRESTConformers(
    threads=16, calculation_dynamics_method="gfnff", calculation_energy_method="gfnff"
).make(generate_structure.output.structure)

energyies = TBLiteSinglePointCalculator(method="GFN2-xTB").make(conformers.output.structure)
prism_filter = PrismPrunerFilter(energy_threshold=1.0).make(
    energyies.output.structure, energyies.output.properties
)

flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        energyies,
        # optimize_structure,
        # tautomers,
        # single_point,
        # filter_energy,
        conformers,
        prism_filter,
        # crest_conformers,
        # deprotonation,
        # protonation,
        # tautomers,
    ]
)

response = run_locally(flow)
