"""Shared solvent definitions for the partition coefficient workflow."""

from enum import Enum
from typing import get_args

from jfchemistry.calculators.orca_keywords import SolventType as ORCASolventType
from jfchemistry.calculators.tblite_calculator import TBLiteSolventType

## Get the sets of solvents
tblite_solvents: set[str] = {s.lower().replace(" ", "_") for s in get_args(TBLiteSolventType)}

orca_solvents: set[str] = {s.lower().replace(" ", "_") for s in get_args(ORCASolventType)}

##
intersection = tblite_solvents & orca_solvents
PARTITION_COEFFICIENT_SOLVENTS = tuple(sorted(intersection))

if not PARTITION_COEFFICIENT_SOLVENTS:
    msg = "No overlapping solvents between TBLite and ORCA solvent lists."
    raise ValueError(msg)

PartitionCoefficientSolvent = Enum(
    "PartitionCoefficientSolvent",
    {value.lower().replace("-", "_"): value for value in PARTITION_COEFFICIENT_SOLVENTS},
    type=str,
)

PartitionCoefficientSolventType = PartitionCoefficientSolvent
