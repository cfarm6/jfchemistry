"""The jfchemistry package."""

import pickle
from dataclasses import dataclass
from typing import Any, Optional, cast

from jobflow.core.job import Response, job
from jobflow.core.maker import Maker
from pymatgen.core.structure import SiteCollection
from rdkit.Chem import rdchem


class RDMolMolecule(rdchem.Mol):
    """
    Represents a molecule in the RDKit format.

    Inherits from rdkit.Chem.rdchem.Mol.
    """

    def as_dict(self) -> dict[str, Any]:
        """Convert the molecule to a dictionary."""
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": pickle.dumps(super()),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Convert a dictionary to a molecule."""
        if type(d["data"]) is str:
            return pickle.loads(eval(d["data"]))
        else:
            return pickle.loads(d["data"])


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


def handle_molecules(
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
    maker: Maker, structures: list[SiteCollection] | SiteCollection
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
            "structure": [job.output["structure"] for job in jobs],
            "files": [job.output["files"] for job in jobs],
            "properties": [job.output["properties"] for job in jobs],
        },
        detour=jobs,  # type: ignore
    )


@dataclass
class SingleStructureMaker(Maker):
    """Base Single Structure Maker for jfchemistry.

    This class applies to structures with an assigned 3D geometry.
    """

    name: str = "Single Structure Maker"

    def operation(
        self, structure: SiteCollection
    ) -> tuple[SiteCollection | list[SiteCollection], Optional[dict[str, Any]]]:
        """Operation to perform on the molecule."""
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(
        self,
        structure: SiteCollection | list[SiteCollection],
    ) -> Response[dict[str, Any]]:
        """Handle distributing for lists of molecules or conformers."""
        resp = handle_structures(self, structure)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            structures, properties = self.operation(cast("SiteCollection", structure))
            if type(structures) is list:
                files = [s.to(fmt="xyz") for s in structures]
            else:
                files = [structures.to(fmt="xyz")]
            return Response(
                output={
                    "structure": structures,
                    "files": files,
                    "properties": properties,
                }
            )


@dataclass
class SingleMoleculeMaker(Maker):
    """Base Single RDMolMolecule Maker for jfchemistry.

    This class applies to structures withoutan assigned 3D geometry.
    """

    name: str = "Single RDMolMolecule Maker"

    def operation(
        self, mol: RDMolMolecule
    ) -> tuple[SiteCollection | list[SiteCollection], dict[str, Any]]:
        """Operation to perform on the molecule.

        Args:
            structure: The molecule to perform the operation on.

        Returns
        -------
            A tuple containing the molecule and a dictionary of properties.
        """
        raise NotImplementedError

    @job(files="files", properties="properties")
    def make(
        self,
        molecule: RDMolMolecule | list[RDMolMolecule],
    ) -> Response[dict[str, Any]]:
        """Handle distributing for lists of molecules or conformers."""
        resp = handle_molecules(self, molecule)
        if resp is not None:
            return resp
        else:  # If the structure is not a list, generate a single structure
            molecule, properties = self.operation(cast("RDMolMolecule", molecule))
            if type(molecule) is list:
                files = [m.to(fmt="mol") for m in molecule]
            else:
                files = [molecule.to(fmt="mol")]
            return Response(
                output={
                    "structure": molecule,
                    "files": files,
                    "properties": properties,
                }
            )
