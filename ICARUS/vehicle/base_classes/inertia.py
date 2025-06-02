from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from ICARUS.core.types import ComplexArray
from ICARUS.core.types import FloatArray
from ICARUS.core.types import Matrix3x3


@dataclass(frozen=True)
class InertiaTensor:
    """Immutable inertia tensor representation with validation and utilities."""

    I_xx: float
    I_yy: float
    I_zz: float
    I_xy: float = 0.0
    I_xz: float = 0.0
    I_yz: float = 0.0

    def __post_init__(self) -> None:
        """Validate inertia tensor properties."""
        # Check positive definiteness of diagonal elements
        if self.I_xx < 0 or self.I_yy < 0 or self.I_zz < 0:
            raise ValueError("Diagonal inertia elements must be non-negative")

        # I1, I2, I3 = self.compute_principal_inertias()
        # print(f"Principal inertias: I1={I1}, I2={I2}, I3={I3}")
        # # Check triangle inequalities (necessary for physical validity)
        # if not (I1 + I2 >= I3 and I1 + I3 >= I2 and I2 + I3 >= I1):
        #     warnings.warn("Inertia tensor may not be physically realizable")

    def compute_principal_inertias(self) -> tuple[float, float, float]:
        """Compute principal moments of inertia."""
        eigvals = np.linalg.eigvals(self.matrix)
        I_1 = eigvals[0]
        I_2 = eigvals[1]
        I_3 = eigvals[2]
        return I_1, I_2, I_3

    @cached_property
    def to_list(self) -> list[float]:
        """Return inertia tensor as a list."""
        return [self.I_xx, self.I_yy, self.I_zz, self.I_xy, self.I_xz, self.I_yz]

    @cached_property
    def matrix(self) -> Matrix3x3:
        """3x3 inertia matrix representation."""
        return np.array(
            [
                [self.I_xx, -self.I_xy, -self.I_xz],
                [-self.I_xy, self.I_yy, -self.I_yz],
                [-self.I_xz, -self.I_yz, self.I_zz],
            ],
            dtype=float,
        )

    @cached_property
    def eigenvalues(self) -> FloatArray | ComplexArray:
        """Principal moments of inertia."""
        return np.linalg.eigvals(self.matrix)

    @cached_property
    def principal_axes(self) -> Matrix3x3:
        """Principal axes as column vectors."""
        _, eigvecs = np.linalg.eigh(self.matrix)
        return eigvecs

    @cached_property
    def trace(self) -> float:
        """Trace of inertia tensor."""
        return self.I_xx + self.I_yy + self.I_zz

    def transform(self, rotation: Matrix3x3 | Rotation) -> InertiaTensor:
        """Transform inertia tensor by rotation."""
        if isinstance(rotation, Rotation):
            R = rotation.as_matrix()
        else:
            R = np.asarray(rotation)

        I_new = R @ self.matrix @ R.T
        return InertiaTensor(
            I_xx=I_new[0, 0],
            I_yy=I_new[1, 1],
            I_zz=I_new[2, 2],
            I_xy=I_new[0, 1],
            I_xz=I_new[0, 2],
            I_yz=I_new[1, 2],
        )

    def __add__(self, other: InertiaTensor) -> InertiaTensor:
        """Add two inertia tensors."""
        return InertiaTensor(
            I_xx=self.I_xx + other.I_xx,
            I_yy=self.I_yy + other.I_yy,
            I_zz=self.I_zz + other.I_zz,
            I_xy=self.I_xy + other.I_xy,
            I_xz=self.I_xz + other.I_xz,
            I_yz=self.I_yz + other.I_yz,
        )

    def __mul__(self, scalar: float) -> InertiaTensor:
        """Scale inertia tensor by scalar."""
        return InertiaTensor(
            I_xx=self.I_xx * scalar,
            I_yy=self.I_yy * scalar,
            I_zz=self.I_zz * scalar,
            I_xy=self.I_xy * scalar,
            I_xz=self.I_xz * scalar,
            I_yz=self.I_yz * scalar,
        )

    @classmethod
    def from_matrix(cls, matrix: Matrix3x3) -> InertiaTensor:
        """Create from 3x3 matrix."""
        matrix = np.asarray(matrix)
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")

        return cls(
            I_xx=matrix[0, 0],
            I_yy=matrix[1, 1],
            I_zz=matrix[2, 2],
            I_xy=matrix[0, 1],
            I_xz=matrix[0, 2],
            I_yz=matrix[1, 2],
        )

    @classmethod
    def sphere(cls, mass: float, radius: float) -> InertiaTensor:
        """Inertia tensor for solid sphere."""
        I_sphere: float = 0.4 * mass * radius**2
        return cls(I_xx=I_sphere, I_yy=I_sphere, I_zz=I_sphere)

    @classmethod
    def cylinder(cls, mass: float, radius: float, height: float, axis: str = "z") -> InertiaTensor:
        """Inertia tensor for solid cylinder."""
        I_perp = mass * (3 * radius**2 + height**2) / 12
        I_axis = 0.5 * mass * radius**2

        if axis.lower() == "z":
            return cls(I_xx=I_perp, I_yy=I_perp, I_zz=I_axis)
        elif axis.lower() == "y":
            return cls(I_xx=I_perp, I_yy=I_axis, I_zz=I_perp)
        elif axis.lower() == "x":
            return cls(I_xx=I_axis, I_yy=I_perp, I_zz=I_perp)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    @classmethod
    def box(cls, mass: float, length: float, width: float, height: float) -> InertiaTensor:
        """Inertia tensor for solid rectangular box."""
        I_xx = mass * (width**2 + height**2) / 12
        I_yy = mass * (length**2 + height**2) / 12
        I_zz = mass * (length**2 + width**2) / 12
        return cls(I_xx=I_xx, I_yy=I_yy, I_zz=I_zz)
