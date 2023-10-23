from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray


class Strip:
    """
    Class to define a strip of a wing or lifting surface.
    It assumes the strip is defined by the position of two trailing edge points
    and the airfoil. It then calcutes all intermediate points based on the chord
    distribution.
    """

    def __init__(
        self,
        start_leading_edge: FloatArray | list[float],
        start_chord: float,
        start_airfoil: Airfoil,
        end_leading_edge: FloatArray | list[float],
        end_chord: float,
        end_airfoil: Airfoil | None = None,
    ) -> None:
        """
        Initialize the Strip class.

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

        self.airfoil1: Airfoil = start_airfoil
        if end_airfoil is None:
            self.airfoil2: Airfoil = start_airfoil
        else:
            self.airfoil2 = end_airfoil

        self.chord: list[float] = [start_chord, end_chord]

    def return_symmetric(
        self,
    ) -> "Strip":
        """
        Returns the symmetric initializer of the strip, assuming symmetry in the y axis.
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
            start_chord=self.chord[1],
            start_airfoil=self.airfoil1,
            end_leading_edge=end_point,
            end_chord=self.chord[0],
            end_airfoil=self.airfoil2,
        )
        return symm_strip

    def set_airfoils(self, airfoil: Airfoil, airfoil2: Airfoil | None = None) -> None:
        """
        Used to set or change the Airfoil.

        Args:
            airfoil (Airfoil): Airfoil for the starting section.
            airfoil2 (Airfoil, optional): Airfoil for the ending section. Defaults to None. If None, the starting airfoil is used.
        """
        self.airfoil1 = airfoil
        if airfoil2 is not None:
            self.airfoil2 = airfoil2
        else:
            self.airfoil2 = airfoil

    def get_root_strip(self) -> ndarray[Any, dtype[floating[Any]]]:
        """
        Returns the root strip of the wing.

        Returns:
            ndarray[Any, dtype[floating[Any]]]: Array of points defining the root.
        """
        strip: list[ndarray[Any, dtype[floating[Any]]]] = [
            self.x0 + self.chord[0] * np.hstack((self.airfoil1._x_upper, self.airfoil1._x_lower)),
            self.y0 + np.repeat(0, 2 * self.airfoil1.n_points),
            self.z0 + self.chord[0] * np.hstack((self.airfoil1._y_upper, self.airfoil1._y_lower)),
        ]
        return np.array(strip)

    def get_tip_strip(self) -> ndarray[Any, dtype[floating[Any]]]:
        """
        Returns the tip strip of the wing.

        Returns:
            ndarray[Any, dtype[floating[Any]]]: Array of points defining the tip.
        """
        strip: list[ndarray[Any, dtype[floating[Any]]]] = [
            self.x1 + self.chord[1] * np.hstack((self.airfoil2._x_upper, self.airfoil2._x_lower)),
            self.y1 + np.repeat(0, 2 * self.airfoil1.n_points),
            self.z1 + self.chord[1] * np.hstack((self.airfoil2._y_upper, self.airfoil2._y_lower)),
        ]
        return np.array(strip)

    def get_interpolated_section(
        self,
        idx: int,
        n_points_span: int = 10,
    ) -> ndarray[Any, dtype[floating[Any]]]:
        """Interpolate between start and end strips and return the section at the given index.

        Args:
            idx: index of interpolation
            n_points: number of points to interpolate in the span direction
        Returns:
            strip: 3xn_points array of points
        """
        x: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            start=self.x0,
            stop=self.x1,
            num=n_points_span,
        )
        y: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            self.y0,
            self.y1,
            n_points_span,
        )
        z: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            self.z0,
            self.z1,
            n_points_span,
        )
        c: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            self.chord[0],
            self.chord[1],
            n_points_span,
        )

        # Relative position of the point wrt to the start and end of the strip
        heta: float = (idx + 1) / (n_points_span + 1)

        airfoil: Airfoil = Airfoil.morph_new_from_two_foils(self.airfoil1, self.airfoil2, heta, self.airfoil1.n_points)

        camber_line: ndarray[Any, dtype[Any]] = np.array(
            [
                x[idx] + c[idx] * airfoil._x_lower,
                y[idx] + np.repeat(0, airfoil.n_points),
                z[idx] + c[idx] * airfoil.camber_line(airfoil._x_lower),
            ],
            dtype=float,
        )

        suction_side: ndarray[Any, dtype[Any]] = np.array(
            [
                x[idx] + c[idx] * airfoil._x_upper,
                y[idx] + np.repeat(0, airfoil.n_points),
                z[idx] + c[idx] * airfoil.y_upper(airfoil._x_upper),
            ],
            dtype=float,
        )

        pressure_side: ndarray[Any, dtype[Any]] = np.array(
            [
                x[idx] + c[idx] * airfoil._x_lower,
                y[idx] + np.repeat(0, airfoil.n_points),
                z[idx] + c[idx] * airfoil.y_lower(airfoil._x_lower),
            ],
            dtype=float,
        )

        return camber_line

    def plot(
        self,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        movement: FloatArray | None = None,
        color: tuple[Any, ...] | None = None,
    ) -> None:
        pltshow = False

        if isinstance(prev_fig, Figure) and isinstance(prev_ax, Axes3D):
            fig: Figure = prev_fig
            ax: Axes3D = prev_ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_title("Strip")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            pltshow = True

        if movement is None:
            movement = np.zeros(3)

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        N: int = 10
        for i in range(N):
            x, y, z = self.get_interpolated_section(i, N)
            xs.append(x + movement[0])
            ys.append(y + movement[1])
            zs.append(z + movement[2])
        X: FloatArray = np.array(xs)
        Y: FloatArray = np.array(ys)
        Z: FloatArray = np.array(zs)

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
