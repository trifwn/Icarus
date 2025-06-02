from __future__ import annotations

import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.vehicle.utils import SymmetryAxes


from jax.tree_util import register_pytree_node

class Strip:
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
        self.x_c4: float = x_c4
        self.y_c4: float = y_c4
        self.z_c4: float = z_c4

        self.pitch: float = pitch
        self.roll: float = roll
        self.yaw: float = yaw

        self.chord: float = chord
        self.airfoil: Airfoil = airfoil

        direction = np.array([self.pitch, self.roll, self.yaw])
        direction /= np.linalg.norm(direction)
        leading_edge = np.array([self.x_c4, self.y_c4, self.z_c4]) - 0.25 * self.chord * direction

        self.leading_edge: tuple[float, float, float] = (
            float(leading_edge[0]),
            float(leading_edge[1]),
            float(leading_edge[2]),
        )

        self.max_thickness: float = self.airfoil.max_thickness * self.chord

    def translate(self, dx: float, dy: float, dz: float) -> Strip:
        """Translate the strip by the given distances in x, y, and z directions."""
        return Strip(
            x_c4=self.x_c4 + dx,
            y_c4=self.y_c4 + dy,
            z_c4=self.z_c4 + dz,
            pitch=self.pitch,
            roll=self.roll,
            yaw=self.yaw,
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def rotate(self, d_pitch: float, d_roll: float, d_yaw: float) -> Strip:
        """Rotate the strip by the given angles in pitch, roll, and yaw."""
        return Strip(
            x_c4=self.x_c4,
            y_c4=self.y_c4,
            z_c4=self.z_c4,
            pitch=self.pitch + d_pitch,
            roll=self.roll + d_roll,
            yaw=self.yaw + d_yaw,
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def rotate_around_point(self, point: tuple[float, float, float], rotation: tuple[float, float, float]) -> Strip:
        """Rotate the strip around a specified point by given rotation angles."""
        dx = self.x_c4 - point[0]
        dy = self.y_c4 - point[1]
        dz = self.z_c4 - point[2]

        # Apply rotation around the specified point
        new_x = point[0] + dx * np.cos(rotation[2]) - dy * np.sin(rotation[2])
        new_y = point[1] + dx * np.sin(rotation[2]) + dy * np.cos(rotation[2])
        new_z = self.z_c4

        return Strip(
            x_c4=new_x,
            y_c4=new_y,
            z_c4=new_z,
            pitch=self.pitch + rotation[0],
            roll=self.roll + rotation[1],
            yaw=self.yaw + rotation[2],
            chord=self.chord,
            airfoil=self.airfoil,
        )

    def scale(self, factor: float) -> Strip:
        """Scale the strip by a given factor."""
        return Strip(
            x_c4=self.x_c4,
            y_c4=self.y_c4,
            z_c4=self.z_c4,
            pitch=self.pitch,
            roll=self.roll,
            yaw=self.yaw,
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
                pitch=self.pitch,
                roll=self.roll,
                yaw=self.yaw,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        elif axis == SymmetryAxes.Z:
            return Strip(
                x_c4=self.x_c4,
                y_c4=self.y_c4,
                z_c4=-self.z_c4,
                pitch=self.pitch,
                roll=self.roll,
                yaw=self.yaw,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        elif axis == SymmetryAxes.X:
            return Strip(
                x_c4=-self.x_c4,
                y_c4=self.y_c4,
                z_c4=self.z_c4,
                pitch=self.pitch,
                roll=self.roll,
                yaw=self.yaw,
                chord=self.chord,
                airfoil=self.airfoil,
            )
        else:
            raise ValueError(f"Invalid symmetry axis: {axis}")

    def plot(self, *args, **kwargs) -> None:
        """Plot the strip in 3D space."""
        pass

register_pytree_node(
    Strip,
    lambda strip_data: ((
        strip_data.x_c4, 
        strip_data.y_c4, 
        strip_data.z_c4,
        strip_data.pitch, 
        strip_data.roll, 
        strip_data.yaw,
        strip_data.chord, 
        strip_data.airfoil
    ), None),  # tell JAX how to unpack to an iterable
    lambda _, strip_data: Strip(
        *strip_data
    )       # tell JAX how to pack back into a Point
)