from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray


class Strip:
    """Class to define a strip of a wing or lifting surface.
    It assumes the strip is defined by the position of two trailing edge points
    and the airfoil. It then calcutes all intermediate points based on the chord
    distribution.
    """

    def __init__(
        self,
        start_leading_edge: FloatArray | list[float],
        start_chord: float,
        start_twist: float,
        start_airfoil: Airfoil,
        end_leading_edge: FloatArray | list[float],
        end_chord: float,
        end_twist: float,
        end_airfoil: Airfoil | None = None,
        eta: float = 0.0,
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
        self.x0: float = start_leading_edge[0]
        self.y0: float = start_leading_edge[1]
        self.z0: float = start_leading_edge[2]

        self.x1: float = end_leading_edge[0]
        self.y1: float = end_leading_edge[1]
        self.z1: float = end_leading_edge[2]

        self.airfoil_start: Airfoil = start_airfoil
        if end_airfoil is None:
            self.airfoil_end: Airfoil = start_airfoil
        else:
            self.airfoil_end = end_airfoil

        # Morph the airfoils to the new shape
        if self.airfoil_start is not self.airfoil_end:
            self.mean_airfoil = Airfoil.morph_new_from_two_foils(
                self.airfoil_start,
                self.airfoil_end,
                eta,
                self.airfoil_start.n_points,
            )
        else:
            self.mean_airfoil = self.airfoil_start

        self.chords: list[float] = [start_chord, end_chord]
        self.mean_chord: float = (start_chord + end_chord) / 2
        self.twists: list[float] = [start_twist, end_twist]
        self.mean_twist = (start_twist + end_twist) / 2
        self.max_thickness: float = self.mean_airfoil.max_thickness

    def return_symmetric(
        self,
    ) -> Strip:
        """Returns the symmetric initializer of the strip, assuming symmetry in the y axis.
        It also adds a small gap if the strip located along the x axis.

        Returns:
            tuple[list[float], list[float], Airfoil, float, float]: Symmetric Strip initializer

        """
        start_point: list[float] = [self.x1, -self.y1, self.z1]
        if self.y0 == 0:
            end_point: list[float] = [self.x0, 0.01 * self.y1, self.z0]
        else:
            end_point = [self.x0, -self.y0, self.z0]

        symm_strip: Strip = Strip(
            start_leading_edge=start_point,
            start_chord=self.chords[1],
            start_airfoil=self.airfoil_start,
            start_twist=self.twists[0],
            end_leading_edge=end_point,
            end_chord=self.chords[0],
            end_twist=self.twists[1],
            end_airfoil=self.airfoil_end,
        )
        return symm_strip

    def get_root_strip(self) -> FloatArray:
        """Returns the root strip of the wing.

        Returns:
            FloatArray: Array of points defining the root.

        """
        strip: list[FloatArray] = [
            self.x0 + self.chords[0] * np.hstack((self.airfoil_start._x_upper, self.airfoil_start._x_lower)),
            self.y0 + np.repeat(0, self.airfoil_start.n_points),
            self.z0 + self.chords[0] * np.hstack((self.airfoil_start._y_upper, self.airfoil_start._y_lower)),
        ]
        return np.array(strip)

    def get_tip_strip(self) -> FloatArray:
        """Returns the tip strip of the wing.

        Returns:
            FloatArray: Array of points defining the tip.

        """
        strip: list[FloatArray] = [
            self.x1 + self.chords[1] * np.hstack((self.airfoil_end._x_upper, self.airfoil_end._x_lower)),
            self.y1 + np.repeat(0, self.airfoil_start.n_points),
            self.z1 + self.chords[1] * np.hstack((self.airfoil_end._y_upper, self.airfoil_end._y_lower)),
        ]
        return np.array(strip)

    def get_interpolated_section(
        self,
        idx: int,
        n_points_span: int = 10,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Interpolate between start and end strips and return the section at the given index.

        Args:
            idx: index of interpolation
            n_points: number of points to interpolate in the span direction
        Returns:
            tuple[FloatArray, FloatArray, FloatArray]: suction side, camber line and pressure side coordinates of the section at the given index

        """
        x: FloatArray = np.linspace(
            start=self.x0,
            stop=self.x1,
            num=n_points_span,
        )
        y: FloatArray = np.linspace(
            self.y0,
            self.y1,
            n_points_span,
        )
        z: FloatArray = np.linspace(
            self.z0,
            self.z1,
            n_points_span,
        )
        c: FloatArray = np.linspace(
            self.chords[0],
            self.chords[1],
            n_points_span,
        )

        # Relative position of the point wrt to the start and end of the strip
        heta: float = (idx + 1) / (n_points_span + 1)
        self.n_points = len(self.airfoil_start._x_upper)
        airfoil: Airfoil = Airfoil.morph_new_from_two_foils(
            self.airfoil_start,
            self.airfoil_end,
            heta,
            self.airfoil_start.n_points,
        )

        camber_line: FloatArray = np.vstack(
            [
                x[idx] + c[idx] * airfoil._x_lower,
                y[idx] + np.repeat(0, len(airfoil._x_lower)),
                z[idx] + c[idx] * airfoil.camber_line(airfoil._x_lower),
            ],
            dtype=float,
        )

        suction_side: FloatArray = np.vstack(
            [
                x[idx] + c[idx] * airfoil._x_upper,
                y[idx] + np.repeat(0, len(airfoil._x_upper)),
                z[idx] + c[idx] * airfoil.y_upper(airfoil._x_upper),
            ],
            dtype=float,
        )

        pressure_side: FloatArray = np.vstack(
            [
                x[idx] + c[idx] * airfoil._x_lower,
                y[idx] + np.repeat(0, len(airfoil._x_lower)),
                z[idx] + c[idx] * airfoil.y_lower(airfoil._x_lower),
            ],
            dtype=float,
        )

        return suction_side, camber_line, pressure_side

    def __str__(self) -> str:
        return f"Strip: with airfoil {self.airfoil_start.name}"

    def plot(
        self,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        movement: FloatArray | None = None,
        color: tuple[Any, ...] | np.ndarray[Any, Any] | None = None,
    ) -> None:
        pltshow = False

        if isinstance(prev_fig, Figure) and isinstance(prev_ax, Axes3D):
            fig: Figure = prev_fig
            ax: Axes3D = prev_ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")  # type: ignore
            ax.set_title("Strip")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            pltshow = True

        if movement is None:
            movement = np.zeros(3)

        to_plot = ["camber"]

        xs: dict[str, list[float]] = {key: [] for key in to_plot}
        ys: dict[str, list[float]] = {key: [] for key in to_plot}
        zs: dict[str, list[float]] = {key: [] for key in to_plot}

        N_span: int = 10
        for i in range(N_span):
            suction, camber, pressure = self.get_interpolated_section(i, N_span)

            x_camber, y_camber, z_camber = (camber.T + movement).T
            x_suction, y_suction, z_suction = (suction.T + movement).T
            x_pressure, y_pressure, z_pressure = (pressure.T + movement).T

            for key in to_plot:
                if key == "camber":
                    xs[key].append(x_camber)
                    ys[key].append(y_camber)
                    zs[key].append(z_camber)
                elif key == "suction":
                    xs[key].append(x_suction)
                    ys[key].append(y_suction)
                    zs[key].append(z_suction)
                elif key == "pressure":
                    xs[key].append(x_pressure)
                    ys[key].append(y_pressure)
                    zs[key].append(z_pressure)

        for key in to_plot:
            X: FloatArray = np.array(xs[key])
            Y: FloatArray = np.array(ys[key])
            Z: FloatArray = np.array(zs[key])

            if color is not None:
                my_color: Any = np.tile(color, (Z.shape[0], Z.shape[1])).reshape(
                    Z.shape[0],
                    Z.shape[1],
                    4,
                )
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_color)
            else:
                my_color = "red"
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

        if pltshow:
            plt.show()

    def plot_points(
        self,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        movement: FloatArray | None = None,
        color: tuple[Any, ...] | np.ndarray[Any, Any] | None = None,
    ) -> None:
        pltshow = False

        if isinstance(prev_fig, Figure) and isinstance(prev_ax, Axes3D):
            fig: Figure = prev_fig
            ax: Axes3D = prev_ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")  # type: ignore
            ax.set_title("Strip Points")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            pltshow = True

        if movement is None:
            movement = np.zeros(3)

        x_coords: list[float] = []
        y_coords: list[float] = []
        z_coords: list[float] = []

        N: int = 10
        for i in range(N):
            suction, camber, pressure = self.get_interpolated_section(i, N)

            x_camber, y_camber, z_camber = camber
            x_suction, y_suction, z_suction = suction
            x_pressure, y_pressure, z_pressure = pressure

            x_coords.extend(x_suction + movement[0])
            y_coords.extend(y_suction + movement[1])
            z_coords.extend(z_suction + movement[2])

            x_coords.extend(x_camber + movement[0])
            y_coords.extend(y_camber + movement[1])
            z_coords.extend(z_camber + movement[2])

            x_coords.extend(x_pressure + movement[0])
            y_coords.extend(y_pressure + movement[1])
            z_coords.extend(z_pressure + movement[2])

        X: FloatArray = np.array(x_coords)
        Y: FloatArray = np.array(y_coords)
        Z: FloatArray = np.array(z_coords)

        if color is not None:
            ax.scatter(X, Y, Z, c=color)
        else:
            ax.scatter(X, Y, Z)

        if pltshow:
            plt.show()

    def __eq__(self, other: object) -> bool:
        """Compares two strips. They are considered equal if the leading edge points, the chord and the airfoil are the same.

        Args:
            __value (Strip): Strip to compare

        Returns:
            bool: True if the strips are equal, False otherwise.

        """
        if not isinstance(other, Strip):
            return NotImplemented
        if (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.z0 == other.z0
            and self.x1 == other.x1
            and self.y1 == other.y1
            and self.z1 == other.z1
            and self.chords == other.chords
            and self.airfoil_start == other.airfoil_start
            and self.airfoil_end == other.airfoil_end
        ):
            return True
        return False
