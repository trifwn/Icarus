from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import jax.numpy as jnp
from jaxtyping import Float

from ICARUS.core.types import Matrix3x3
from ICARUS.core.types import Vector3D

from .inertia import InertiaTensor
from .mass import Mass


class RigidBody(ABC):
    """Abstract base class for any rigid body with physical properties and mass distribution."""

    def __init__(
        self,
        name: str,
        origin: Float,
        orientation: Float,
        masses: list[Mass] | Mass | None = None,
        total_inertia: InertiaTensor | None = None,
    ) -> None:
        """
        Initialize a rigid body.

        Args:
            name: Body identifier
            origin: Position of body reference frame origin
            orientation: Euler angles [pitch, yaw, roll] in degrees
            mass: Total mass (used if mass_points not provided)
            total_inertia: Overall inertia tensor (computed from mass_points if not provided)
        """
        self._name: str = name

        # Define Coordinate System
        self._origin: Float = jnp.array(origin, dtype=float)
        # Convert orientation from degrees to radians
        self._orientation_rad: Float = (
            jnp.array(orientation, dtype=float) * jnp.pi / 180
        )

        # Define Orientation angles
        pitch, roll, yaw = self._orientation_rad
        self._pitch: float = pitch
        self._yaw: float = yaw
        self._roll: float = roll

        # Calculate rotation matrix
        self._update_rotation_matrix()

        # Mass distribution
        if isinstance(masses, Mass):
            masses = [masses]
        self._mass_points: list[Mass] = masses or []

        # Total mass - either from mass_points or provided value
        if self._mass_points:
            self._total_mass = sum(mp.mass for mp in self._mass_points)
        else:
            self._total_mass = 0.0

        # Total inertia - either provided or computed from mass_points
        if total_inertia is not None:
            self._total_inertia = total_inertia
        elif self._mass_points:
            self._total_inertia = self._compute_total_inertia()
        else:
            # Default to zero inertia if no information provided
            self._total_inertia = InertiaTensor(0.0, 0.0, 0.0)

    def _update_rotation_matrix(self) -> None:
        """Update the rotation matrix based on current orientation."""
        self.R_MAT = self._compute_rotation_matrix(
            pitch=self._pitch,
            roll=self._roll,
            yaw=self._yaw,
        )

    @staticmethod
    def _compute_rotation_matrix(
        pitch: float,
        roll: float,
        yaw: float,
    ) -> Matrix3x3:
        """
        Generate a rotation matrix from Euler angles.

        Args:
            pitch (float): Pitch angle in radians
            yaw (float): Yaw angle in radians
            roll (float): Roll angle in radians

        Returns:
            Matrix3x3: 3x3 rotation matrix
        """
        R_PITCH: Float = jnp.array(
            [
                [jnp.cos(pitch), 0, jnp.sin(pitch)],
                [0, 1, 0],
                [-jnp.sin(pitch), 0, jnp.cos(pitch)],
            ],
            dtype=float,
        )

        R_YAW: Float = jnp.array(
            [
                [jnp.cos(yaw), -jnp.sin(yaw), 0],
                [jnp.sin(yaw), jnp.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        R_ROLL: Float = jnp.array(
            [
                [1, 0, 0],
                [0, jnp.cos(roll), -jnp.sin(roll)],
                [0, jnp.sin(roll), jnp.cos(roll)],
            ],
            dtype=float,
        )

        return R_ROLL @ R_YAW @ R_PITCH

    def _compute_total_inertia(self) -> InertiaTensor:
        """Compute total inertia tensor from mass points about body origin."""
        if not self._mass_points:
            return InertiaTensor(0.0, 0.0, 0.0)

        # Combine all mass points into a single equivalent mass
        total_inertia = self._mass_points[0].inertia_about_origin
        for mp in self._mass_points[1:]:
            total_inertia = total_inertia + mp.inertia_about_origin

        return total_inertia

    # Name property
    @property
    def name(self) -> str:
        return self._name.replace(" ", "_")

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    # Mass properties
    @property
    def mass(self) -> float:
        """Total mass of the rigid body."""
        return self._total_mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Set total mass. This will scale existing mass points proportionally."""
        if value <= 0:
            raise ValueError("Mass must be positive")

        if self._mass_points:
            # Scale existing mass points proportionally
            scale_factor = value / self._total_mass
            for mp in self._mass_points:
                mp.mass *= scale_factor
            self._total_mass = value
            self._total_inertia = self._compute_total_inertia()
        else:
            self._total_mass = float(value)

    @property
    def mass_points(self) -> list[Mass]:
        """List of discrete mass points comprising the body."""
        return self._mass_points.copy()

    def add_mass_point(self, mass_point: Mass) -> None:
        """Add a mass point to the body."""
        self._mass_points.append(mass_point)
        self._total_mass += mass_point.mass
        self._total_inertia = self._compute_total_inertia()

    def remove_mass_point(self, name: str) -> Mass | None:
        """Remove a mass point by name and return it."""
        for i, mp in enumerate(self._mass_points):
            if mp.name == name:
                removed = self._mass_points.pop(i)
                self._total_mass -= removed.mass
                self._total_inertia = self._compute_total_inertia()
                return removed
        return None

    # Origin properties
    @property
    def origin(self) -> Float:
        return self._origin

    @origin.setter
    def origin(self, value: Float) -> None:
        movement = jnp.array(value, dtype=float) - self._origin
        self._origin = jnp.array(value, dtype=float)

        # Update mass points positions
        for mp in self._mass_points:
            mp.position = mp.position + movement

        self._on_origin_changed(movement)

    @property
    def x_origin(self) -> float:
        return float(self._origin[0])

    @x_origin.setter
    def x_origin(self, value: float) -> None:
        new_origin = jnp.array([value, self._origin[1], self._origin[2]], dtype=float)
        self.origin = new_origin

    @property
    def y_origin(self) -> float:
        return float(self._origin[1])

    @y_origin.setter
    def y_origin(self, value: float) -> None:
        new_origin = jnp.array([self._origin[0], value, self._origin[2]], dtype=float)
        self.origin = new_origin

    @property
    def z_origin(self) -> float:
        return float(self._origin[2])

    @z_origin.setter
    def z_origin(self, value: float) -> None:
        new_origin = jnp.array([self._origin[0], self._origin[1], value], dtype=float)
        self.origin = new_origin

    # Orientation properties
    @property
    def orientation_rad(self) -> Float:
        return self._orientation_rad

    @property
    def orientation_degrees(self) -> Float:
        """Get orientation in degrees."""
        return self._orientation_rad * 180 / jnp.pi

    @orientation_rad.setter
    def orientation_rad(self, value: Float) -> None:
        old_orientation = self._orientation_rad.copy()
        new_orientation = jnp.array(value, dtype=float)

        self._orientation_rad = jnp.array(value, dtype=float)
        self._pitch, self._roll, self._yaw = self._orientation_rad

        # R_OLD , R_NEW
        self._update_rotation_matrix()
        # Transform mass points
        for mp in self._mass_points:
            # This is a simplified rotation - in practice you might want to rotate about CG
            mp.position = self.transform_point(mp.position)

        self._on_orientation_changed(old_orientation, new_orientation)

    @orientation_degrees.setter
    def orientation_degrees(self, value: Float) -> None:
        """Set orientation in degrees."""
        orientation_rad = jnp.array(value, dtype=float) * jnp.pi / 180
        self.orientation_rad = orientation_rad

    @property
    def pitch_rad(self) -> float:
        return self._pitch

    @property
    def pitch_degrees(self) -> float:
        """Get pitch in degrees."""
        return self._pitch * 180 / jnp.pi

    @pitch_rad.setter
    def pitch_rad(self, value: float) -> None:
        self._pitch = value
        self.orientation_rad = jnp.array([self._pitch, self._roll, self._yaw])

    @pitch_degrees.setter
    def pitch_degrees(self, value: float) -> None:
        """Set pitch in degrees."""
        self.pitch_rad = value * jnp.pi / 180

    @property
    def yaw_rad(self) -> float:
        return self._yaw

    @property
    def yaw_degrees(self) -> float:
        """Get yaw in degrees."""
        return self._yaw * 180 / jnp.pi

    @yaw_rad.setter
    def yaw_rad(self, value: float) -> None:
        self._yaw = value
        self.orientation_rad = jnp.array([self._pitch, self._roll, self._yaw])

    @yaw_degrees.setter
    def yaw_degrees(self, value: float) -> None:
        """Set yaw in degrees."""
        self.yaw_degrees = value * jnp.pi / 180

    @property
    def roll_rad(self) -> float:
        return self._roll

    @property
    def roll_degrees(self) -> float:
        """Get roll in degrees."""
        return self._roll * 180 / jnp.pi

    @roll_rad.setter
    def roll_rad(self, value: float) -> None:
        self._roll = value
        self.orientation_rad = jnp.array([self._pitch, self._roll, self._yaw])

    @roll_degrees.setter
    def roll_degrees(self, value: float) -> None:
        """Set roll in degrees."""
        self.roll_rad = value * jnp.pi / 180

    # Physical properties
    @property
    def CG(self) -> Float:
        """Center of gravity of the body."""
        # Compute center of mass from mass points
        total_moment = jnp.zeros(3)
        for mp in self._mass_points:
            total_moment += mp.mass * mp.position

        return total_moment / self._total_mass

    @property
    def inertia_tensor(self) -> InertiaTensor:
        """Inertia tensor about the center of gravity."""
        if not self._mass_points:
            return self._total_inertia

        cg = self.CG

        # Transform inertia to CG using parallel axis theorem
        total_inertia = InertiaTensor(0.0, 0.0, 0.0)
        for mp in self._mass_points:
            # Position relative to CG
            r_cg = mp.position - cg

            # Inertia about CG using parallel axis theorem
            correction = InertiaTensor(
                I_xx=mp.mass * (r_cg[1] ** 2 + r_cg[2] ** 2),
                I_yy=mp.mass * (r_cg[0] ** 2 + r_cg[2] ** 2),
                I_zz=mp.mass * (r_cg[0] ** 2 + r_cg[1] ** 2),
                I_xy=mp.mass * r_cg[0] * r_cg[1],
                I_xz=mp.mass * r_cg[0] * r_cg[2],
                I_yz=mp.mass * r_cg[1] * r_cg[2],
            )

            total_inertia = total_inertia + mp.inertia + correction

        return total_inertia

    @property
    def inertia(self) -> Float:
        """Inertia tensor as array [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]."""
        return jnp.array(self.inertia_tensor.to_list)

    # Convenience properties for inertia components
    @property
    def Ixx(self) -> float:
        return float(self.inertia_tensor.I_xx)

    @property
    def Iyy(self) -> float:
        return float(self.inertia_tensor.I_yy)

    @property
    def Izz(self) -> float:
        return float(self.inertia_tensor.I_zz)

    @property
    def Ixz(self) -> float:
        return float(self.inertia_tensor.I_xz)

    @property
    def Ixy(self) -> float:
        return float(self.inertia_tensor.I_xy)

    @property
    def Iyz(self) -> float:
        return float(self.inertia_tensor.I_yz)

    @property
    @abstractmethod
    def volume(self) -> float:
        """Volume of the body."""
        pass

    # Abstract methods for handling property changes
    @abstractmethod
    def _on_origin_changed(self, movement: Float) -> None:
        """Called when origin changes. Subclasses should update their geometry."""
        pass

    @abstractmethod
    def _on_orientation_changed(
        self,
        old_orientation: Float,
        new_orientation: Float,
    ) -> None:
        """Called when orientation changes. Subclasses should update their geometry."""
        pass

    # Transform methods
    def transform_point(self, point: Float) -> Float:
        """Transform a point from local to global coordinates."""
        return jnp.matmul(self.R_MAT, point) + self._origin

    def transform_points(self, points: Float) -> Float:
        """Transform multiple points from local to global coordinates."""
        return jnp.matmul(self.R_MAT, points.T).T + self._origin

    def inverse_transform_point(self, point: Float) -> Float:
        """Transform a point from global to local coordinates."""
        return jnp.matmul(jnp.linalg.inv(self.R_MAT), point - self._origin)

    def inverse_transform_points(self, points: Float) -> Float:
        """Transform multiple points from global to local coordinates."""
        return jnp.matmul(jnp.linalg.inv(self.R_MAT), (points - self._origin).T).T

    # Factory methods for common mass distributions
    def add_concentrated_mass(
        self,
        name: str,
        position: Vector3D,
        mass: float,
        inertia: InertiaTensor | Matrix3x3 | None = None,
    ) -> None:
        """Add a concentrated point mass."""
        mass_point = Mass(name, position, mass, inertia=inertia)
        self.add_mass_point(mass_point)

    def add_distributed_mass(
        self,
        name: str,
        geometry_type: str,
        position: Vector3D,
        mass: float,
        **geometry_params: Any,
    ) -> None:
        """Add mass with standard geometric distribution."""
        if geometry_type == "sphere":
            mass_point = Mass.sphere(name, position, mass, **geometry_params)
        elif geometry_type == "cylinder":
            mass_point = Mass.cylinder(name, position, mass, **geometry_params)
        elif geometry_type == "box":
            mass_point = Mass.box(name, position, mass, **geometry_params)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")

        self.add_mass_point(mass_point)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"mass={self.mass:.2f}kg, "
            f"CG={self.CG}, "
            f"mass_points={len(self._mass_points)})"
        )
