"""CENSO conformer filtering."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from jobflow.core.maker import Maker


@dataclass
class CENSOScreening(Maker):
    """CENSO screening."""

    name: str = "CENSO Screening"
    # Temperature
    temperature: float = 298.15  # K
    # Evaludate Thermal Energy Contributions
    evaluate_rrho: bool = True
    # Solvation
    solvation: Optional[
        Union[
            tuple[
                Literal["alpb"],
                Literal[
                    "acetone",
                    "acetonitrile",
                    "aniline",
                    "benzaldehyde",
                    "benzene",
                    "ch2cl2",
                    "chcl3",
                    "cs2",
                    "dioxane",
                    "dmf",
                    "dmso",
                    "ether",
                    "ethylacetate",
                    "furane",
                    "hexandecane",
                    "hexane",
                    "methanol",
                    "nitromethane",
                    "octanol",
                    "woctanol",
                    "phenol",
                    "toluene",
                    "thf",
                    "water",
                ],
            ],
            tuple[
                Literal["gbsa"],
                Literal[
                    "acetone",
                    "acetonitrile",
                    "benzene",
                    "CH2Cl2",
                    "CHCl3",
                    "CS2",
                    "DMF",
                    "DMSO",
                    "ether",
                    "H2O",
                    "methanol",
                    "n-hexane",
                    "THF",
                    "toluene",
                ],
            ],
        ]
    ] = None
    # Imaginary Frequency Threshold
    imagthr = -100  # cm^-1
    # Wave number threshold for switching in the rrho approximation
    sthr = 50  # cm^-1
    # Copy molecular orbitals between steps
    copy_mo: bool = True
    # XC functional
    func: Literal[
        "pbeh-3c",
        "b97-3c",
        "r2scan-3c",
        "r2scan-novdw",
        "r2scan-d3",
        "r2scan-d3(0)",
        "r2scan-d4",
        "pbe-novdw",
        "pbe-d3",
        "pbe-d3(0)",
        "pbe-d4",
        "tpss-novdw",
        "tpss-d3",
        "tpss-d4",
        "revtpss-novdw",
        "tpssh-novdw",
        "tpssh-d3",
        "tpssh-d4",
        "b97-d4",
        "kt2-novdw",
        "pbe0-novdw",
        "pbe0-d3",
        "pbe0-d3(0)",
        "pbe0-d4",
        "pw6b95-novdw",
        "pw6b95-d3",
        "pw6b95-d3(0)",
        "pw6b95-d4",
        "b3lyp-novdw",
        "b3lyp-d3",
        "b3lyp-d3(0)",
        "b3lyp-d4",
        "b3lyp-nl",
        "wb97x-v",
        "wb97x-d3",
        "wb97x-d3bj",
        "wb97x-d4",
        "wb97m-v",
        "dsd-blyp-d3",
        "dsd-pbep86-d3",
    ] = "r2scan-3c"
