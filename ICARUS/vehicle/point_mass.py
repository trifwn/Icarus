"""
Enhanced PointMass class with improved design, performance, and functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Protocol
from typing import Sequence

import numpy as np
from scipy.integrate import tplquad
from scipy.spatial.transform import Rotation

from ICARUS.core.types import ComplexArray
from ICARUS.core.types import FloatArray

# Type definitions
Vector3D = FloatArray  # Shape (3,)
Matrix3x3 = FloatArray  # Shape (3, 3)
DistributionFunc = Callable[[float, float, float], float]


class MassDistribution(Protocol):
    """Protocol for mass distribution functions."""

    def __call__(self, x: float, y: float, z: float) -> float:
        """Return mass density at point (x, y, z)."""
        ...


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
    def cylinder(
        cls,
        mass: float,
        radius: float,
        height: float,
        axis: str = "z",
    ) -> InertiaTensor:
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
    def box(
        cls,
        mass: float,
        length: float,
        width: float,
        height: float,
    ) -> InertiaTensor:
        """Inertia tensor for solid rectangular box."""
        I_xx = mass * (width**2 + height**2) / 12
        I_yy = mass * (length**2 + height**2) / 12
        I_zz = mass * (length**2 + width**2) / 12
        return cls(I_xx=I_xx, I_yy=I_yy, I_zz=I_zz)


class PointMass:
    """
    Advanced point mass model with robust inertia handling and physics utilities.

    Features:
    - Immutable inertia tensor with validation
    - Efficient caching of computed properties
    - Physics-based factory methods
    - Proper transformation methods
    - Comprehensive comparison and hashing
    """

    def __init__(
        self,
        name: str,
        position: Vector3D | Sequence[float],
        mass: float,
        inertia: InertiaTensor | Matrix3x3 | Vector3D | None = None,
    ) -> None:
        """
        Initialize point mass.

        Args:
            name: Identifier for the mass point
            position: 3D position vector [x, y, z] in meters
            mass: Mass in kg (must be positive)
            inertia: Inertia specification (tensor, matrix, or 6-element vector)
        """
        if mass <= 0:
            raise ValueError("Mass must be positive")

        self._name = str(name)
        self._mass = float(mass)
        self._position = np.asarray(position, dtype=float)

        if self._position.shape != (3,):
            raise ValueError("Position must be 3D vector")

        # Handle inertia input
        if inertia is None:
            self._inertia = InertiaTensor(0.0, 0.0, 0.0)
        elif isinstance(inertia, InertiaTensor):
            self._inertia = inertia
        elif isinstance(inertia, np.ndarray):
            if inertia.shape == (3, 3):
                self._inertia = InertiaTensor.from_matrix(inertia)
            elif inertia.shape == (6,):
                self._inertia = InertiaTensor(*inertia)
            else:
                raise ValueError(
                    "Array inertia must be (3,3) matrix or 6-element vector",
                )
        else:
            try:
                inertia_arr = np.asarray(inertia)
                if inertia_arr.shape == (6,):
                    self._inertia = InertiaTensor(*inertia_arr)
                else:
                    raise ValueError("Invalid inertia format")
            except (ValueError, TypeError):
                raise ValueError(
                    "Inertia must be InertiaTensor, 3x3 matrix, or 6-element vector",
                )

    # Properties with validation
    @property
    def name(self) -> str:
        return self._name

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Mass must be positive")
        self._mass = float(value)
        self._clear_cache()

    @property
    def position(self) -> Vector3D:
        return self._position.copy()

    @position.setter
    def position(self, value: Vector3D) -> None:
        pos = np.asarray(value, dtype=float)
        if pos.shape != (3,):
            raise ValueError("Position must be 3D vector")
        self._position = pos
        self._clear_cache()

    @property
    def x(self) -> float:
        return float(self._position[0])

    @x.setter
    def x(self, value: float) -> None:
        self._position[0] = float(value)
        self._clear_cache()

    @property
    def y(self) -> float:
        return float(self._position[1])

    @y.setter
    def y(self, value: float) -> None:
        self._position[1] = float(value)
        self._clear_cache()

    @property
    def z(self) -> float:
        return float(self._position[2])

    @z.setter
    def z(self, value: float) -> None:
        self._position[2] = float(value)
        self._clear_cache()

    @property
    def inertia(self) -> InertiaTensor:
        return self._inertia

    @inertia.setter
    def inertia(self, value: InertiaTensor | Matrix3x3 | Vector3D) -> None:
        if isinstance(value, InertiaTensor):
            self._inertia = value
        else:
            # Use constructor logic to handle various formats
            temp = PointMass("temp", [0, 0, 0], 1.0, value)
            self._inertia = temp._inertia
        self._clear_cache()

    # Computed properties with caching
    @cached_property
    def position_magnitude(self) -> float:
        """Distance from origin."""
        return float(np.linalg.norm(self._position))

    @cached_property
    def inertia_about_origin(self) -> InertiaTensor:
        """Inertia tensor about origin using parallel axis theorem."""
        r = self._position

        # Parallel axis theorem correction
        correction = InertiaTensor(
            I_xx=self._mass * (r[1] ** 2 + r[2] ** 2),
            I_yy=self._mass * (r[0] ** 2 + r[2] ** 2),
            I_zz=self._mass * (r[0] ** 2 + r[1] ** 2),
            I_xy=self._mass * r[0] * r[1],
            I_xz=self._mass * r[0] * r[2],
            I_yz=self._mass * r[1] * r[2],
        )

        return self._inertia + correction

    def _clear_cache(self) -> None:
        """Clear cached properties when state changes."""
        for attr in ["position_magnitude", "inertia_about_origin"]:
            if hasattr(self, f"_{attr}"):
                delattr(self, f"_{attr}")

    # Factory methods
    @classmethod
    def sphere(
        cls,
        name: str,
        position: Vector3D | Sequence[float],
        mass: float,
        radius: float,
        hollow: bool = False,
    ) -> PointMass:
        """Create spherical mass."""
        if hollow:
            inertia = InertiaTensor.sphere(mass, radius) * (2 / 3)  # Hollow sphere
        else:
            inertia = InertiaTensor.sphere(mass, radius)
        return cls(name, position, mass, inertia)

    @classmethod
    def cylinder(
        cls,
        name: str,
        position: Vector3D,
        mass: float,
        radius: float,
        height: float,
        axis: str = "z",
    ) -> PointMass:
        """Create cylindrical mass."""
        inertia = InertiaTensor.cylinder(mass, radius, height, axis)
        return cls(name, position, mass, inertia)

    @classmethod
    def box(
        cls,
        name: str,
        position: Vector3D | Sequence[float],
        mass: float,
        length: float,
        width: float,
        height: float,
    ) -> PointMass:
        """Create rectangular box mass."""
        inertia = InertiaTensor.box(mass, length, width, height)
        return cls(name, position, mass, inertia)

    @classmethod
    @lru_cache(maxsize=128)
    def from_distribution(
        cls,
        name: str,
        position: Vector3D,
        mass: float,
        distribution: MassDistribution,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-10, 10),
            (-10, 10),
            (-10, 10),
        ),
        # **integration_kwargs: dict[Any, Any],
    ) -> PointMass:
        """
        Create point mass from mass distribution with improved integration.

        Args:
            name: Mass point name
            position: Center position
            mass: Total mass
            distribution: Mass density function f(x,y,z)
            bounds: Integration bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            # **integration_kwargs: Additional arguments for scipy.integrate.tplquad
        """
        # Default integration options
        opts = {"epsabs": 1e-8, "epsrel": 1e-8}
        # opts.update(integration_kwargs)

        # More efficient integration using tplquad
        def compute_moment(
            moment_func: Callable[[float, float, float], float],
        ) -> float:
            result = tplquad(
                lambda z, y, x: distribution(x, y, z) * moment_func(x, y, z),
                bounds[0][0],
                bounds[0][1],  # x bounds
                bounds[1][0],
                bounds[1][1],  # y bounds
                bounds[2][0],
                bounds[2][1],  # z bounds
                **opts,
            )
            return result[0] * mass

        # Compute inertia components
        I_xx = compute_moment(lambda x, y, z: y**2 + z**2)
        I_yy = compute_moment(lambda x, y, z: x**2 + z**2)
        I_zz = compute_moment(lambda x, y, z: x**2 + y**2)
        I_xy = compute_moment(lambda x, y, z: x * y)
        I_xz = compute_moment(lambda x, y, z: x * z)
        I_yz = compute_moment(lambda x, y, z: y * z)

        inertia = InertiaTensor(I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)
        return cls(name, position, mass, inertia)

    # Transformation methods
    def translate(self, displacement: Vector3D) -> PointMass:
        """Create new PointMass translated by displacement."""
        new_pos = self._position + np.asarray(displacement)
        return PointMass(self._name, new_pos, self._mass, self._inertia)

    def rotate_about_origin(self, rotation: Matrix3x3 | Rotation) -> PointMass:
        """Create new PointMass rotated about origin."""
        if isinstance(rotation, Rotation):
            R = rotation.as_matrix()
        else:
            R = np.asarray(rotation)

        new_pos = R @ self._position
        new_inertia = self._inertia.transform(R)
        return PointMass(self._name, new_pos, self._mass, new_inertia)

    def transform(
        self,
        rotation: Matrix3x3 | Rotation,
        translation: Vector3D,
    ) -> PointMass:
        """Apply rotation then translation."""
        return self.rotate_about_origin(rotation).translate(translation)

    # Combination methods
    def __add__(self, other: PointMass) -> PointMass:
        """Combine two point masses into one at their center of mass."""
        total_mass = self._mass + other._mass

        # Center of mass
        com = (self._mass * self._position + other._mass * other._position) / total_mass

        # Combined inertia about center of mass
        inertia1_com = self.translate(com - self._position).inertia_about_origin
        inertia2_com = other.translate(com - other._position).inertia_about_origin
        combined_inertia = inertia1_com + inertia2_com

        return PointMass(
            f"{self._name}+{other._name}",
            com,
            total_mass,
            combined_inertia,
        )

    # Utility methods
    def distance_to(self, other: PointMass | Vector3D) -> float:
        """Distance to another point mass or position."""
        if isinstance(other, PointMass):
            other_pos = other._position
        else:
            other_pos = np.asarray(other)
        return float(np.linalg.norm(self._position - other_pos))

    def copy(self) -> PointMass:
        """Create deep copy."""
        return PointMass(self._name, self._position.copy(), self._mass, self._inertia)

    def __deepcopy__(self, memo: Any) -> PointMass:
        return self.copy()

    # Comparison and hashing
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PointMass):
            return NotImplemented
        return (
            self._name == other._name
            and self._mass == other._mass
            and np.allclose(self._position, other._position)
            and self._inertia == other._inertia
        )

    def __hash__(self) -> int:
        return hash((self._name, self._mass, tuple(self._position), self._inertia))

    # String representation
    def __repr__(self) -> str:
        return (
            f"PointMass(name='{self._name}', position={self._position.tolist()}, "
            f"mass={self._mass}, inertia={self._inertia})"
        )

    def __str__(self) -> str:
        return f"{self._name}: {self._mass:.2f}kg at {self._position}"

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self._name,
            "position": self._position.tolist(),
            "mass": self._mass,
            "inertia": {
                "I_xx": self._inertia.I_xx,
                "I_yy": self._inertia.I_yy,
                "I_zz": self._inertia.I_zz,
                "I_xy": self._inertia.I_xy,
                "I_xz": self._inertia.I_xz,
                "I_yz": self._inertia.I_yz,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointMass:
        """Create from dictionary."""
        inertia_data = data["inertia"]
        inertia = InertiaTensor(**inertia_data)
        return cls(data["name"], data["position"], data["mass"], inertia)


# Example usage and demonstrations
if __name__ == "__main__":
    # Create various point masses
    sphere = PointMass.sphere("ball", [1, 2, 3], 5.0, 0.1)
    box = PointMass.box("cube", [0, 0, 0], 2.0, 1.0, 1.0, 1.0)

    # Combine masses
    combined = sphere + box

    # Transform masses
    rotation = Rotation.from_euler("xyz", [45, 0, 0], degrees=True)
    rotated_sphere = sphere.rotate_about_origin(rotation)

    print(f"Original sphere: {sphere}")
    print(f"Rotated sphere: {rotated_sphere}")
    print(f"Combined mass: {combined}")
    print(f"Distance between masses: {sphere.distance_to(box):.2f}m")
