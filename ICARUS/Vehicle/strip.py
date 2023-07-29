from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Airfoils.airfoilD import AirfoilD
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
        starting_point: ndarray[int, dtype[floating[Any]]] | list[float],
        ending_point: ndarray[int, dtype[floating[Any]]] | list[float],
        airfoil: AirfoilD,
        starting_chord: float,
        ending_chord: float,
    ) -> None:
        """
        Initialize the Strip class.

        Args:
            starting_point (ndarray[int, dtype[floating[Any]]]): Starting point of the strip
            ending_point (ndarray[int, dtype[floating[Any]]]): Ending point of the strip
            airfoil (AirfoilD): Airfoil of the strip.
            starting_chord (float): Starting chord of the strip
            ending_chord (float): Ending chord of the strip
        """
        self.x0: float = starting_point[0]
        self.y0: float = starting_point[1]
        self.z0: float = starting_point[2]

        self.x1: float = ending_point[0]
        self.y1: float = ending_point[1]
        self.z1: float = ending_point[2]

        self.airfoil: AirfoilD = airfoil
        self.chord: list[float] = [starting_chord, ending_chord]

    def return_symmetric(
        self,
    ) -> tuple[list[float], list[float], AirfoilD, float, float]:
        """
        Returns the symmetric initializer of the strip, assuming symmetry in the y axis.
        It also adds a small gap if the strip located along the x axis.

        Returns:
            tuple[list[float], list[float], AirfoilD, float, float]: Symmetric Strip initializer
        """
        start_point: list[float] = [self.x1, -self.y1, self.z1]
        if self.y0 == 0:
            end_point: list[float] = [self.x0, 0.01 * self.y1, self.z0]
        else:
            end_point = [self.x0, -self.y0, self.z0]
        airf: AirfoilD = self.airfoil
        return start_point, end_point, airf, self.chord[1], self.chord[0]

    def set_airfoil(self, airfoil: AirfoilD) -> None:
        """
        Used to set or change the airfoil.

        Args:
            airfoil (AirfoilD): AirfoilD Class Object.
        """
        self.airfoil = airfoil

    def get_root_strip(self) -> ndarray[Any, dtype[floating[Any]]]:
        """
        Returns the root strip of the wing.

        Returns:
            ndarray[Any, dtype[floating[Any]]]: Array of points defining the root.
        """
        strip: list[ndarray[Any, dtype[floating[Any]]]] = [
            self.x0 + self.chord[0] * np.hstack((self.airfoil._x_upper, self.airfoil._x_lower)),
            self.y0 + np.repeat(0, 2 * self.airfoil.n_points),
            self.z0 + self.chord[0] * np.hstack((self.airfoil._y_upper, self.airfoil._y_lower)),
        ]
        return np.array(strip)

    def get_tip_strip(self) -> ndarray[Any, dtype[floating[Any]]]:
        """
        Returns the tip strip of the wing.

        Returns:
            ndarray[Any, dtype[floating[Any]]]: Array of points defining the tip.
        """
        strip: list[ndarray[Any, dtype[floating[Any]]]] = [
            self.x1 + self.chord[1] * np.hstack((self.airfoil._x_upper, self.airfoil._x_lower)),
            self.y1 + np.repeat(0, 2 * self.airfoil.n_points),
            self.z1 + self.chord[1] * np.hstack((self.airfoil._y_upper, self.airfoil._y_lower)),
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

        strip: ndarray[Any, dtype[Any]] = np.array(
            [
                x[idx] + c[idx] * self.airfoil._x_lower,
                y[idx] + np.repeat(0, self.airfoil.n_points),
                z[idx] + c[idx] * self.airfoil.camber_line_naca4(self.airfoil._x_lower),
            ],
            dtype=float,
        )
        return strip

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
        N: int = 2
        for i in range(N):
            x, y, z = self.get_interpolated_section(i, N)
            xs.append(x + movement[0])
            ys.append(y + movement[1])
            zs.append(z + movement[2])
        X: FloatArray = np.array(xs)
        Y: FloatArray = np.array(ys)
        Z: FloatArray = np.array(zs)

        if isinstance(color, tuple):
            my_color: Any = np.tile(color, (Z.shape[0], Z.shape[1])).reshape(
                Z.shape[0],
                Z.shape[1],
                4,
            )
        else:
            my_color = "red"

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_color)

        if pltshow:
            plt.show()
