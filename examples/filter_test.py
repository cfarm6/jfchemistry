"""Example of using the PubChemCID node to get a molecule from PubChem."""

from jobflow.core.flow import Flow
from jobflow.managers.local import run_locally

from jfchemistry.calculators.ase import TBLiteCalculator
from jfchemistry.filters.structural import PrismPrunerFilter
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import Smiles
from jfchemistry.optimizers import ASEOptimizer
from jfchemistry.single_point import ASESinglePoint

pubchem_cid = Smiles().make("C(C=O)Cl")

generate_structure = RDKitGeneration(num_conformers=3).make(pubchem_cid.output.structure)

energies = ASESinglePoint(
    calculator=TBLiteCalculator(method="GFN2-xTB", verbosity=3),
).make(generate_structure.output.structure)

filtered_energies = PrismPrunerFilter(energy_threshold=2.0, structural_threshold=0.01).make(
    energies.output.structure, energies.output.properties
)

opt = ASEOptimizer(
    optimizer="BFGS",
    calculator=TBLiteCalculator(method="GFN2-xTB", verbosity=0),
).make(filtered_energies.output.structure)

flow = Flow([pubchem_cid, generate_structure, energies, filtered_energies, opt])

run_locally(flow)
