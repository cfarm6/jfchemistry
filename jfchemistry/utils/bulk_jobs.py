"""Handle Conformers and Lists of Structures in Arguments."""

from typing import Any, cast

from jobflow.core.job import Response
from jobflow.core.maker import Maker
from pymatgen.core.structure import IMolecule
from rdkit.Chem import rdchem

from jfchemistry.jfchemistry import RDMolMolecule


def handle_conformers(maker: Maker, structure: RDMolMolecule) -> Response[dict[str, Any]]:
    """Handle distributing jobs for conformers."""
    jobs: list[Response[dict[str, Any]]] = []
    for confId in range(structure.GetNumConformers()):
        s = RDMolMolecule(rdchem.Mol(structure, confId=confId))
        jobs.append(maker.make(s))  # type: ignore

    return Response(
        output={
            "structures": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


def handle_list_of_structures(
    maker: Maker, structures: list[RDMolMolecule]
) -> Response[dict[str, Any]]:
    """Handle distributing jobs for a list of structures."""
    jobs: list[Response[dict[str, Any]]] = []
    for structure in structures:
        if structure.GetNumConformers() > 1:
            jobs.append(handle_conformers(maker, structure))
        else:
            jobs.append(maker.make(structure))  # type: ignore

    return Response(
        output={
            "structures": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


def handle_molecule(
    maker: Maker, structure: RDMolMolecule | list[RDMolMolecule]
) -> Response[dict[str, Any]] | None:
    """Handle distributing jobs for a RDKit molecule."""
    if type(structure) is list:
        return handle_list_of_structures(maker, structure)
    elif cast("RDMolMolecule", structure).GetNumConformers() > 1:
        return handle_conformers(maker, cast("RDMolMolecule", structure))
    else:
        return None


def handle_structures(
    maker: Maker, structures: list[IMolecule] | IMolecule
) -> Response[dict[str, Any]] | None:
    """Handle distributing jobs for a list of Molecules."""
    jobs: list[Response[dict[str, Any]]] = []
    if isinstance(structures, list):
        for structure in structures:
            jobs.append(maker.make(structure))  # type: ignore
    else:
        return None

    return Response(
        output={
            "structures": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )
