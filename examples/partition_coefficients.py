"""Example workflow for calculating partition coefficients."""

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from jfchemistry.conformers import CRESTConformers
from jfchemistry.generation import RDKitGeneration
from jfchemistry.inputs import PubChemCID
from jfchemistry.modification import CRESTTautomers
from jfchemistry.optimizers import ORCAOptimizer
from jfchemistry.single_point import ORCASinglePointEnergyCalculator

pubchem_cid = PubChemCID().make(10413)

generate_structure = RDKitGeneration(num_conformers=1).make(pubchem_cid.output["structure"])

## ----- GAS PHASE ------
tautomers_gas = CRESTTautomers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    threads=16,
    name="CREST_TAUTOMERS_GAS",
).make(generate_structure.output["structure"])

crest_conformers_gas = CRESTConformers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    calculation_dynamics_method="gfnff",
    calculation_energy_method="gfn2",
    threads=16,
    name="CREST_CONFORMERS_GAS",
).make(tautomers_gas.output["structure"])

r2scan3c_optimizer_gas = ORCAOptimizer(
    cores=16,
    xc_functional="R2SCAN_3C",
    ecp="DEFECP",
    name="R2SCAN_3C_GAS",
).make(crest_conformers_gas.output["structure"])

wr2scan_d4_single_point_gas = ORCASinglePointEnergyCalculator(
    cores=16,
    xc_functional="WR2SCAN",
    basis_set="DEF2_TZVPPD",
    ecp="DEF2ECP",
    name="WR2SCAN_D4_GAS",
).make(r2scan3c_optimizer_gas.output["structure"])

## ----- WATER PHASE ------

tautomers_water = CRESTTautomers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    threads=16,
    solvation=("alpb", "water"),
    name="CREST_TAUTOMERS_WATER",
).make(generate_structure.output["structure"])

crest_conformers_water = CRESTConformers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    calculation_dynamics_method="gfnff",
    calculation_energy_method="gfn2",
    threads=16,
    solvation=("alpb", "water"),
    name="CREST_CONFORMERS_WATER",
).make(tautomers_water.output["structure"])

r2scan3c_optimizer_water = ORCAOptimizer(
    cores=16,
    xc_functional="R2SCAN_3C",
    solvation_model="CPCM",
    solvent="WATER",
    ecp="DEFECP",
    name="R2SCAN_3C_WATER",
).make(crest_conformers_water.output["structure"])

wr2scan_d4_single_point_water = ORCASinglePointEnergyCalculator(
    cores=16,
    xc_functional="WR2SCAN",
    basis_set="DEF2_TZVPPD",
    ecp="DEF2ECP",
    solvation_model="SMD",
    solvent="WATER",
    name="WR2SCAN_D4_WATER",
).make(r2scan3c_optimizer_water.output["structure"])

## ----- Octanol PHASE ------

tautomers_octanol = CRESTTautomers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    threads=16,
    solvation=("alpb", "octanol"),
    name="CREST_TAUTOMERS_OCTANOL",
).make(generate_structure.output["structure"])

crest_conformers_octanol = CRESTConformers(
    executable="/home/carson/Downloads/crest-gnu-12-ubuntu-latest/crest/crest",
    calculation_dynamics_method="gfnff",
    calculation_energy_method="gfn2",
    threads=16,
    solvation=("alpb", "octanol"),
    name="CREST_CONFORMERS_OCTANOL",
).make(tautomers_octanol.output["structure"])

r2scan3c_optimizer_octanol = ORCAOptimizer(
    cores=16,
    xc_functional="R2SCAN_3C",
    solvation_model="CPCM",
    solvent="OCTANOL",
    ecp="DEFECP",
    name="R2SCAN_3C_OCTANOL",
).make(crest_conformers_octanol.output["structure"])

wr2scan_d4_single_point_octanol = ORCASinglePointEnergyCalculator(
    cores=16,
    xc_functional="WR2SCAN",
    basis_set="DEF2_TZVPPD",
    ecp="DEF2ECP",
    solvation_model="SMD",
    solvent="OCTANOL",
    name="WR2SCAN_D4_OCTANOL",
).make(r2scan3c_optimizer_octanol.output["structure"])

## ----- FLOW -------
flow = Flow(
    [
        pubchem_cid,
        generate_structure,
        tautomers_gas,
        crest_conformers_gas,
        r2scan3c_optimizer_gas,
        wr2scan_d4_single_point_gas,
        tautomers_water,
        crest_conformers_water,
        r2scan3c_optimizer_water,
        wr2scan_d4_single_point_water,
        tautomers_octanol,
        crest_conformers_octanol,
        r2scan3c_optimizer_octanol,
        wr2scan_d4_single_point_octanol,
    ]
)

workflow = flow_to_workflow(flow)
launchpad = LaunchPad.from_file("my_launchpad.yaml")

launchpad.add_wf(workflow)
