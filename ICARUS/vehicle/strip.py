from __future__ import annotations

from typing import Any
from typing import Self

import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import GetAttrKey
from jaxtyping import Float
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.vehicle.base_classes.rigid_body import RigidBody
from ICARUS.vehicle.utils import SymmetryAxes


@jax.tree_util.register_pytree_with_keys_class
class Strip(RigidBody):
    """Class to define a strip of a wing or lifting surface.
    It assumes the strip is defined by the position of two trailing edge points
    and the airfoil. It then calcutes all intermediate points based on the chord
    distribution.
    """

    def __init__(
        self,
        x_c4: float,
        y_c4: float,
        z_c4: float,
        pitch: float,
        roll: float,
        yaw: float,
        chord: float,
        airfoil: Airfoil,
    ) -> None:
        """Initialize the Strip class.

        Args:
            start_leading_edge (FloatArray | list[float]): Starting point of the strip.
            start_chord (float): Starting chord.
            start_airfoil (Airfoil): Starting airfoil.
            end_leading_edge (FloatArray | list[float]): Ending point of the strip.
            end_chord (float): Ending chord.
            end_airfoil (Airfoil, optional): Ending airfoil. Defaults to None. If None, the starting airfoil is used.

        """
        self.x_c4: Float = jnp.array(x_c4)
        self.y_c4: Float = jnp.array(y_c4)
        self.z_c4: Float = jnp.array(z_c4)

        self.chord: Float = jnp.array(chord)
        self.airfoil: Airfoil = airfoil

        rotation = RigidBody._compute_rotation_matrix(
            pitch=pitch * np.pi / 180,
            roll=roll * np.pi / 180,
            yaw=yaw * np.pi / 180,
        )
        base_direction = jnp.array(
            [1.0, 0.0, 0.0],
        )  # Assuming the base direction is along the x-axis
        direction = jnp.dot(rotation, base_direction)

        super().__init__(
            name="Strip",
            origin=jnp.array([self.x_c4, self.y_c4, self.z_c4]),
            orientation=jnp.array([pitch, roll, yaw]),
        )

        leading_edge = (
            jnp.array([self.x_c4, self.y_c4, self.z_c4]) - 0.25 * self.chord * direction
        )

        self.leading_edge: tuple[Float, Float, Float] = (
            leading_edge[0],
            leading_edge[1],
            leading_edge[2],
        )

        self.max_thickness: float = self.airfoil.max_thickness * self.chord

    @property
    def volume(self) -> float:
        """Calculate the volume of the strip."""
        return 0

    def _on_origin_changed(self, movement: FloatArray) -> None:
        pass

    def _on_orientation_changed(
        self,
        old_orientation: FloatArray,
        new_orientation: FloatArray,
    ) -> None:
        pass

    @classmethod
    def from_leading_edge(
        cls,
        leading_edge_x: float,
        leading_edge_y: float,
        leading_edge_z: float,
        pitch: float,
        roll: float,
        yaw: float,
        chord: float,
        airfoil: Airfoil,
    ) -> Strip:
        """Create a Strip instance from the leading edge position, chord, and airfoil."""
        rotation = RigidBody._compute_rotation_matrix(
            pitch=pitch * np.pi / 180,
            roll=roll * np.pi / 180,
            yaw=yaw * np.pi / 180,
        )
        base_direction = jnp.array(
            [1.0, 0.0, 0.0],
        )  # Assuming the base direction is along the x-axis

        direction = jnp.dot(rotation, base_direction)

        quarter_chord = (
            jnp.array([leading_edge_x, leading_edge_y, leading_edge_z])
            + 0.25 * chord * direction
        )

        x_c4, y_c4, z_c4 = quarter_chord

        strip = cls(
            x_c4=x_c4,
            y_c4=y_c4,
            z_c4=z_c4,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            chord=chord,
            airfoil=airfoil,
        )
        return strip

    def translate(self, dx: float, dy: float, dz: float) -> Strip:
        """Translate the strip by the given distances in x, y, and z directions."""
        return Strip(
            x_c4=self.x_c4 + dx,
            y_c4=self.y_c4 + dy,
            z_c4=self.z_c4 + dz,
            pitch=self.pitch_degrees,
            roll=self.roll_degrees,
            yaw=self.yaw_degrees,
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def rotate(self, d_pitch: float, d_roll: float, d_yaw: float) -> Strip:
        """Rotate the strip by the given angles in pitch, roll, and yaw."""
        return Strip(
            x_c4=self.x_c4,
            y_c4=self.y_c4,
            z_c4=self.z_c4,
            pitch=self.pitch_degrees + d_pitch,
            roll=self.roll_degrees + d_roll,
            yaw=self.yaw_degrees + d_yaw,
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def rotate_around_point(
        self,
        point: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> Strip:
        """Rotate the strip around a specified point by given rotation angles."""
        dx = self.x_c4 - point[0]
        dy = self.y_c4 - point[1]
        # dz = self.z_c4 - point[2]

        # Apply rotation around the specified point
        new_x = point[0] + dx * np.cos(rotation[2]) - dy * np.sin(rotation[2])
        new_y = point[1] + dx * np.sin(rotation[2]) + dy * np.cos(rotation[2])
        new_z = self.z_c4

        return Strip(
            x_c4=new_x,
            y_c4=new_y,
            z_c4=new_z,
            pitch=self.pitch_degrees + rotation[0],
            roll=self.roll_degrees + rotation[1],
            yaw=self.yaw_degrees + rotation[2],
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def scale(self, factor: float) -> Strip:
        """Scale the strip by a given factor."""
        return Strip(
            x_c4=self.x_c4,
            y_c4=self.y_c4,
            z_c4=self.z_c4,
            pitch=self.pitch_degrees,
            roll=self.roll_degrees,
            yaw=self.yaw_degrees,
            chord=self.chord * factor,
            airfoil=self.airfoil,
        )

    def return_symmetric(self, axis: SymmetryAxes = SymmetryAxes.Y) -> Strip:
        """Return a symmetric strip based on the specified symmetry axis."""
        if axis == SymmetryAxes.Y:
            return Strip(
                x_c4=self.x_c4,
                y_c4=-self.y_c4,
                z_c4=self.z_c4,
                pitch=self.pitch_degrees,
                roll=self.roll_degrees,
                yaw=self.yaw_degrees,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        elif axis == SymmetryAxes.Z:
            return Strip(
                x_c4=self.x_c4,
                y_c4=self.y_c4,
                z_c4=-self.z_c4,
                pitch=self.pitch_degrees,
                roll=self.roll_degrees,
                yaw=self.yaw_degrees,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        elif axis == SymmetryAxes.X:
            return Strip(
                x_c4=-self.x_c4,
                y_c4=self.y_c4,
                z_c4=self.z_c4,
                pitch=self.pitch_degrees,
                roll=self.roll_degrees,
                yaw=self.yaw_degrees,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        else:
            raise ValueError(f"Invalid symmetry axis: {axis}")

    def plot(
        self,
        ax: Axes | None = None,
        color: str | tuple[Any, ...] | np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Plot the strip."""
        from ICARUS.visualization import parse_Axes

        fig, ax = parse_Axes(ax)

        x, z = self.airfoil.to_selig() * self.chord
        y = np.zeros_like(x)

        # 3x3 rotation matrix for yaw, pitch, and roll
        rotation_mat = RigidBody._compute_rotation_matrix(
            pitch=self.pitch_rad,
            roll=self.roll_rad,
            yaw=self.yaw_rad,
        )

        # Rotate points based on orientation angles
        coords = jnp.array([x, y, z])
        rotated_coords = jnp.dot(rotation_mat, coords)

        x_rot, y_rot, z_rot = rotated_coords

        # Translate to C4 position
        x = x_rot + self.leading_edge[0]
        y = y_rot + self.leading_edge[1]
        z = z_rot + self.leading_edge[2]
        # Plot C4 point
        ax.plot(self.x_c4, self.z_c4, "kx", label="C4")

        le = self.leading_edge
        c4 = np.array((self.x_c4, self.y_c4, self.z_c4))
        te = le - (le - c4) * 4

        chord_x = np.array([le[0], te[0]])
        # chord_y = np.array([le[1], te[1]])
        chord_z = np.array([le[2], te[2]])
        ax.plot(chord_x, chord_z, "k--", label="Chord")

        ax.plot(x, z)
        # Relim axis
        ax.set_aspect("equal")
        fig.show()

    def plot_3D(self, ax: Axes3D | None = None) -> None:
        """Plot the strip in 3D."""
        from ICARUS.visualization import parse_Axes3D

        fig, ax = parse_Axes3D(ax)

        x, z = self.airfoil.to_selig() * self.chord
        y = np.zeros_like(x)

        # 3x3 rotation matrix for yaw, pitch, and roll
        rotation_mat = RigidBody._compute_rotation_matrix(
            pitch=self.pitch_rad,
            roll=self.roll_rad,
            yaw=self.yaw_rad,
        )

        # Rotate points based on orientation angles
        coords = np.array([x, y, z])
        rotated_coords = np.dot(rotation_mat, coords)
        x_rot, y_rot, z_rot = rotated_coords

        # Translate to C4 position
        x = x_rot + self.leading_edge[0]
        y = y_rot + self.leading_edge[1]
        z = z_rot + self.leading_edge[2]
        ax.plot(x, y, z, color="b")

        # Plot C4 point
        ax.plot([self.x_c4], [self.y_c4], [self.z_c4], "kx", label="C4")

        # Plot chord line (3 times the segment from LE to C4)
        le = self.leading_edge
        c4 = np.array((self.x_c4, self.y_c4, self.z_c4))
        te = le - (le - c4) * 4

        chord_x = np.array([le[0], te[0]])
        chord_y = np.array([le[1], te[1]])
        chord_z = np.array([le[2], te[2]])
        ax.plot(chord_x, chord_y, chord_z, "k--", label="Chord")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        fig.show()

    def __repr__(self) -> str:
        """String representation of the Strip object."""
        return (
            f"Strip(x_c4={self.x_c4}, y_c4={self.y_c4}, z_c4={self.z_c4}, "
            f"pitch={self.pitch_degrees}, roll={self.roll_degrees}, yaw={self.yaw_degrees}, "
            f"chord={self.chord}, airfoil={self.airfoil.name})"
        )

    def __str__(self) -> str:
        """String representation of the Strip object."""
        return (
            f"Strip at C4({self.x_c4:.2f}, {self.y_c4:.2f}, {self.z_c4:.2f}) "
            f"with pitch={self.pitch_degrees:.2f}, roll={self.roll_degrees:.2f}, yaw={self.yaw_degrees:.2f}, "
            f"chord={self.chord:.2f} and airfoil={self.airfoil.name}"
        )

    def tree_flatten_with_keys(self):
        return (
            (GetAttrKey("x_c4"), self.x_c4),
            (GetAttrKey("y_c4"), self.y_c4),
            (GetAttrKey("z_c4"), self.z_c4),
            (GetAttrKey("pitch"), self.pitch_degrees),
            (GetAttrKey("roll"), self.roll_degrees),
            (GetAttrKey("yaw"), self.yaw_degrees),
            (GetAttrKey("chord"), self.chord),
            (GetAttrKey("airfoil"), self.airfoil),
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children)
