"""Rotate a pymatgen Molecule: user matrix/axis-angle or principal-axes alignment."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from pymatgen.core.structure import Molecule

from jfchemistry.core.makers import PymatGenMaker
from jfchemistry.core.properties import Properties

# Type for 3x3 rotation matrix (list of lists for JSON serialization)
RotationMatrix = list[list[float]]
# Axis: "x" | "y" | "z" or [x, y, z]
AxisSpec = Literal["x", "y", "z"] | list[float]

# Type alias matching JFChemMaker._operation return for override compatibility
_OpResult = tuple[Molecule | list[Molecule], Properties | list[Properties]]

TOLERANCE = 1e-10


def _inertia_tensor_and_principal_axes(
    coords: np.ndarray, masses: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inertia tensor and principal axes (eigenvectors) at center.

    Returns:
        I: 3x3 inertia tensor.
        axes: 3x3 array, columns are eigenvectors (principal axes), ordered by
            increasing eigenvalue (smallest moment -> first column).
    """
    r = coords - center  # (n, 3)
    m = masses.reshape(-1, 1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y)
    Ixz = -np.sum(m * x * z)
    Iyz = -np.sum(m * y * z)
    intertia_tensor = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    _, eigenvectors = np.linalg.eigh(intertia_tensor)
    # eigh returns columns as eigenvectors, ordered by increasing eigenvalue
    return intertia_tensor, np.asarray(eigenvectors)


def _rotation_matrix_principal_axes_to_lab(
    coords: np.ndarray, masses: np.ndarray, axis_order: str
) -> np.ndarray:
    """Rotation matrix R such that new_coords = R.T @ coords_centered aligns principal axes.

    axis_order: "largest_z" (largest moment along z), "smallest_z", "largest_x", "smallest_x", etc.
    """
    center = np.average(coords, axis=0, weights=masses)
    _, axes = _inertia_tensor_and_principal_axes(coords, masses, center)
    # axes columns: 0 = smallest, 1 = middle, 2 = largest moment
    smallest, middle, largest = axes[:, 0], axes[:, 1], axes[:, 2]
    order_map = {
        "largest_z": (middle, smallest, largest),  # x,y,z from principal 1,0,2
        "smallest_z": (middle, largest, smallest),
        "largest_y": (smallest, largest, middle),
        "smallest_y": (largest, smallest, middle),
        "largest_x": (largest, middle, smallest),
        "smallest_x": (smallest, middle, largest),
    }
    if axis_order not in order_map:
        raise ValueError(f"axis_order must be one of {list(order_map.keys())}; got {axis_order!r}")
    x_axis, y_axis, z_axis = order_map[axis_order]
    # R.T sends principal to lab: columns of R = lab axes in principal frame
    R = np.column_stack([x_axis, y_axis, z_axis]).T
    if np.linalg.det(R) < 0:
        R[1] *= -1
    return R


def _apply_rotation_mode(
    atoms,
    pos: np.ndarray,
    masses: np.ndarray,
    center_pt: np.ndarray,
    opts: dict,
) -> np.ndarray:
    """Apply rotation per opts['mode']; return new positions. Mutates atoms for axis_angle."""
    mode = opts.get("mode")
    if mode == "matrix":
        rm = opts.get("rotation_matrix")
        if rm is None:
            raise ValueError("mode 'matrix' requires rotation_matrix")
        R = np.asarray(rm, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("rotation_matrix must be 3x3")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            raise ValueError("rotation_matrix must be a proper rotation matrix (det=1)")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
            raise ValueError("rotation_matrix must be orthogonal (R @ R.T = I)")
        return (R @ (pos - center_pt).T).T + center_pt
    if mode == "axis_angle":
        axis, angle_deg = opts.get("axis"), opts.get("angle_deg")
        if axis is None or angle_deg is None:
            raise ValueError("mode 'axis_angle' requires axis and angle_deg")
        if not np.isfinite(angle_deg):
            raise ValueError("angle_deg must be a finite number")
        if isinstance(axis, (list, tuple)):
            axis_arr = np.asarray(axis, dtype=float)
            if axis_arr.shape != (3,) or np.linalg.norm(axis_arr) < TOLERANCE:
                raise ValueError("axis must be 'x', 'y', 'z', or a non-zero length-3 vector")
        atoms.rotate(angle_deg, axis, center=tuple(center_pt))
        return np.asarray(atoms.get_positions())
    axis_order = opts.get("axis_order", "largest_z")
    R = _rotation_matrix_principal_axes_to_lab(pos, masses, axis_order)
    return (R.T @ (pos - center_pt).T).T + center_pt


@dataclass
class RotateMolecule(PymatGenMaker[Molecule, Molecule]):
    """Rotate a Molecule using a rotation matrix, axis-angle, or principal-axes alignment.

    Only supports pymatgen Molecule (not Structure). Modes:
    - "matrix": apply a user-provided 3x3 rotation matrix.
    - "axis_angle": rotate by angle (degrees) around an axis (e.g. "z" or [0,0,1]).
    - "principal_axes": align principal moments of inertia with lab axes (e.g. largest along z).

    Set rotation parameters as instance attributes; then call make(molecule) or make([mol1, mol2]).
    """

    name: str = "RotateMolecule"

    # Rotation parameters (set at construction or on the instance)
    mode: Literal["matrix", "axis_angle", "principal_axes"] = "principal_axes"
    rotation_matrix: RotationMatrix | None = None
    axis: AxisSpec | None = None
    angle_deg: float | None = None
    axis_order: str = "largest_z"
    center: tuple[float, float, float] | None = None

    def _operation(self, input: Molecule, **kwargs: object) -> _OpResult:
        """Rotate a molecule using this instance's mode and rotation parameters."""
        if input is None or not isinstance(input, Molecule):
            raise TypeError(
                "RotateMolecule only supports Molecule inputs; "
                f"got {type(input).__name__ if input is not None else 'None'}"
            )
        if self.mode not in ("matrix", "axis_angle", "principal_axes"):
            raise ValueError(
                f"mode must be 'matrix', 'axis_angle', or 'principal_axes'; got {self.mode!r}"
            )
        atoms = input.to_ase_atoms()
        pos = np.asarray(atoms.get_positions())
        masses = np.asarray(atoms.get_masses())
        center_pt = (
            np.average(pos, axis=0, weights=masses)
            if self.center is None
            else np.array(self.center, dtype=float)
        )
        if center_pt.shape != (3,):
            raise ValueError("center must be a length-3 sequence (x, y, z)")
        opts = {
            "mode": self.mode,
            "rotation_matrix": self.rotation_matrix,
            "axis": self.axis,
            "angle_deg": self.angle_deg,
            "axis_order": self.axis_order,
        }
        pos_new = _apply_rotation_mode(atoms, pos, masses, center_pt, opts)
        atoms.set_positions(pos_new)
        return cast("_OpResult", (Molecule.from_ase_atoms(atoms), None))
