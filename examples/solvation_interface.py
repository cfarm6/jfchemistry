"""Example: unified implicit-solvent interface across calculators."""

from jfchemistry.calculators.ase.tblite_calculator import TBLiteCalculator
from jfchemistry.calculators.crest.crest_calculator import CRESTCalculator
from jfchemistry.calculators.orca.orca_calculator import ORCACalculator
from jfchemistry.core.solvation import ImplicitSolventConfig


def main() -> None:
    """Instantiate calculators with a shared implicit-solvent schema."""
    # One shared solvent definition
    water_alpb = ImplicitSolventConfig(model="alpb", solvent="Water")
    water_gbsa = ImplicitSolventConfig(model="gbsa", solvent="water")
    water_cpcm = ImplicitSolventConfig(model="cpcm", solvent="Water")

    # TBLite uses ALPB mapping
    tblite = TBLiteCalculator(implicit_solvent=water_alpb)
    print("TBLite:", {"solvation": tblite.solvation, "solvent": tblite.solvent})

    # CREST uses tuple mapping
    crest = CRESTCalculator(implicit_solvent=water_gbsa)
    print("CREST:", {"solvation": crest.solvation})

    # ORCA maps to CPCM/SMD style fields
    orca = ORCACalculator(implicit_solvent=water_cpcm)
    print(
        "ORCA:",
        {
            "solvation": orca.solvation,
            "solvation_model": orca.solvation_model,
            "solvent": orca.solvent,
        },
    )


if __name__ == "__main__":
    main()
