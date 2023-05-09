from __future__ import annotations

from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import dtype
from numpy import floating
from numpy import ndarray

from .strip import Strip
from ICARUS.Core.types import NumericArray


class Wing:
    """Class to reprsent a wing."""

    def __init__(
        self,
        name: str,
        airfoil,
        origin: NumericArray,
        orientation: NumericArray,
        is_symmetric: bool,
        span: float,
        sweep_offset: float,
        dih_angle: float,
        chord_fun: Callable[[int, float, float], np.ndarray],
        chord: NumericArray,
        span_fun: Callable[[float, int], np.ndarray],
        N: int,
        M: int,
        mass: float = 1.0,
    ) -> None:
        """Initializes the wing."""

        # Conversions to numpy
        orientation = np.array(orientation, dtype=float)
        origin = np.array(origin, dtype=float)
        chord = np.array(chord, dtype=float)

        self.N: int = N
        self.M: int = M

        self.name: str = name
        self.airfoil = airfoil
        self.origin: np.ndarray = origin
        self.orientation: np.ndarray = orientation
        self.is_symmetric: bool = is_symmetric
        self.span = span
        self.sweep_offset = sweep_offset
        self.dih_angle = dih_angle
        self.chord_fun = chord_fun
        self.chord = chord
        self.span_fun = span_fun
        self.mass: float = mass

        self.gamma: float = dih_angle * np.pi / 180

        # orientation
        self.pitch, self.yaw, self.roll = orientation * np.pi / 180
        R_PITCH = np.array(
            [
                [np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)],
            ],
        )
        R_YAW = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ],
        )
        R_ROLL = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)],
            ],
        )
        self.R_MAT = R_YAW.dot(R_PITCH).dot(R_ROLL)

        # Make Dihedral Angle Distribution
        if is_symmetric:
            self._chord_dist = chord_fun(self.N, *chord)
            self._span_dist = span_fun(span / 2, self.N)
            self._offset_dist = (self._span_dist - span / 2) * (
                sweep_offset / (span / 2)
            )
            self._dihedral_dist = (self._span_dist - span / 2) * np.sin(self.gamma)
        else:
            self._chord_dist = chord_fun(self.N, *chord)
            self._span_dist = span_fun(span, self.N)
            self._offset_dist = self._span_dist * sweep_offset / span
            self._dihedral_dist = self._span_dist * np.sin(self.gamma)

        # Create Grid
        self.create_grid()

        # Create Surfaces
        self.create_strips()

        # Find Chords mean_aerodynamic_chord-standard_mean_chord
        self.mean_chords()

        # Calculate Areas
        self.find_area()

        # Calculate Volumes
        self.find_volume()

        # Find Center of Mass
        self.centerMass()

        # Calculate Moments
        self._inertia(self.mass, self.CG)

    def split_symmetric_wing(self) -> tuple[Wing, Wing] | None:
        """Split Symmetric Wing into two Wings"""
        if self.is_symmetric:
            left = Wing(
                name=f"L{self.name}",
                airfoil=self.airfoil,
                origin=np.array(
                    [
                        self.origin[0] + self.sweep_offset,
                        self.origin[1] - self.span / 2,
                        self.origin[2],
                    ],
                    dtype=float,
                ),
                orientation=self.orientation,
                is_symmetric=False,
                span=self.span / 2,
                sweep_offset=-self.sweep_offset,
                dih_angle=self.dih_angle,
                chord_fun=self.chord_fun,
                chord=self.chord[::-1],
                span_fun=self.span_fun,
                N=self.N,
                M=self.M,
                mass=self.mass / 2,
            )

            right = Wing(
                name=f"R{self.name}",
                airfoil=self.airfoil,
                origin=self.origin,
                orientation=self.orientation,
                is_symmetric=False,
                span=self.span / 2,
                sweep_offset=self.sweep_offset,
                dih_angle=self.dih_angle,
                chord_fun=self.chord_fun,
                chord=self.chord,
                span_fun=self.span_fun,
                N=self.N,
                M=self.M,
                mass=self.mass / 2,
            )
            return left, right
        else:
            print("Cannot Split Body it is not symmetric")
            return None

    def create_strips(self) -> None:
        """Create Strips given the Grid and Airfoil"""
        strips = []
        symStrips = []
        for i in np.arange(0, self.N - 1):
            startPoint = np.array(
                [
                    self._offset_dist[i],
                    self._span_dist[i],
                    self._dihedral_dist[i],
                ],
            )
            startPoint = np.matmul(self.R_MAT, startPoint) + self.origin

            endPoint = np.array(
                [
                    self._offset_dist[i + 1],
                    self._span_dist[i + 1],
                    self._dihedral_dist[i + 1],
                ],
            )
            endPoint = np.matmul(self.R_MAT, endPoint) + self.origin

            if self.is_symmetric:
                surf = Strip(
                    startPoint,
                    endPoint,
                    self.airfoil,
                    self._chord_dist[i],
                    self._chord_dist[i + 1],
                )
                strips.append(surf)
                symStrips.append(Strip(*surf.returnSymmetric()))
            else:
                surf = Strip(
                    startPoint,
                    endPoint,
                    self.airfoil,
                    self._chord_dist[i],
                    self._chord_dist[i + 1],
                )
                strips.append(surf)
        self.strips = strips
        self.all_strips = [*strips, *symStrips]

    def plot_wing(
        self,
        prev_fig=None,
        prev_ax=None,
        prev_movement=None,
    ) -> None:
        """Plot Wing in 3D"""
        show_plot: bool = False

        if prev_fig is None:
            fig: Figure = plt.figure()
            ax: Axes3D = fig.add_subplot(projection="3d")
            ax.set_title(self.name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            show_plot = True
        else:
            fig = prev_fig
            ax = prev_ax

        if prev_movement is None:
            movement: ndarray[Any, dtype[floating]] = np.zeros(3)
        else:
            movement = prev_movement

        # for strip in self.all_strips:
        #     strip.plotStrip(fig, ax, movement)

        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                for item in [self.panels_lower, self.panels_upper]:
                    p1, p3, p4, p2 = item[i, j, :, :]
                    xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2)) + movement[0]

                    ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2)) + movement[1]

                    zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2)) + movement[2]

                    ax.plot_wireframe(xs, ys, zs, linewidth=0.5)

                    if self.is_symmetric:
                        ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)
        if show_plot:
            plt.show()

    def grid_to_panels(self, grid):
        """Convert Grid to Panels"""
        panels = np.empty((self.N - 1, self.M - 1, 4, 3))
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                panels[i, j, 0, :] = grid[i + 1, j]
                panels[i, j, 1, :] = grid[i, j]
                panels[i, j, 2, :] = grid[i, j + 1]
                panels[i, j, 3, :] = grid[i + 1, j + 1]
        return panels

    def create_grid(self):
        """Create Grid for Wing"""
        xs = np.empty((self.M, self.N))
        xs_upper = np.empty((self.M, self.N))
        xs_lower = np.empty((self.M, self.N))

        ys = np.empty((self.M, self.N))
        ys_upper = np.empty((self.M, self.N))
        ys_lower = np.empty((self.M, self.N))

        zs = np.empty((self.M, self.N))
        zs_upper = np.empty((self.M, self.N))
        zs_lower = np.empty((self.M, self.N))

        for i in np.arange(0, self.M):
            xpos = (self._chord_dist) * (i / (self.M - 1))
            xs[i, :] = self._offset_dist + xpos
            xs_lower[i, :] = xs[i, :]
            xs_upper[i, :] = xs[i, :]

            ys[i, :] = self._span_dist
            ys_lower[i, :] = ys[i, :]
            ys_upper[i, :] = ys[i, :]

            for j in np.arange(0, self.N):
                zs_upper[i, j] = self._dihedral_dist[j] + self._chord_dist[
                    j
                ] * self.airfoil.y_upper(i / (self.M - 1))
                zs_lower[i, j] = self._dihedral_dist[j] + self._chord_dist[
                    j
                ] * self.airfoil.y_lower(i / (self.M - 1))
                zs[i, j] = self._dihedral_dist[j] + self._chord_dist[
                    j
                ] * self.airfoil.camber_line(
                    i / (self.M - 1),
                )

            # ROTATE ACCORDING TO R_MAT
            xs[i, :], ys[i, :], zs[i, :] = np.matmul(
                self.R_MAT,
                [xs[i, :], ys[i, :], zs[i, :]],
            )

            xs_lower[i, :], ys_lower[i, :], zs_lower[i, :] = np.matmul(
                self.R_MAT,
                [xs_lower[i, :], ys_lower[i, :], zs_lower[i, :]],
            )

            xs_upper[i, :], ys_upper[i, :], zs_upper[i, :] = np.matmul(
                self.R_MAT,
                [xs_upper[i, :], ys_upper[i, :], zs_upper[i, :]],
            )

        for item in [xs, xs_upper, xs_lower]:
            item += self.origin[0]

        for item in [ys, ys_upper, ys_lower]:
            item += self.origin[1]

        for item in [zs, zs_upper, zs_lower]:
            item += self.origin[2]

        self.grid = np.array((xs, ys, zs)).T
        self.grid_upper = np.array((xs_upper, ys_upper, zs_upper)).T
        self.grid_lower = np.array((xs_lower, ys_lower, zs_lower)).T

        self.panels = self.grid_to_panels(self.grid)
        self.panels_lower = self.grid_to_panels(self.grid_lower)
        self.panels_upper = self.grid_to_panels(self.grid_upper)

    def mean_chords(self):
        "Finds the Mean Aerodynamic Chord (mean_aerodynamic_chord) of the wing."
        num = 0
        denum = 0
        for i in np.arange(0, self.N - 1):
            num += ((self._chord_dist[i] + self._chord_dist[i + 1]) / 2) ** 2 * (
                self._span_dist[i + 1] - self._span_dist[i]
            )
            denum += (
                (self._chord_dist[i] + self._chord_dist[i + 1])
                / 2
                * (self._span_dist[i + 1] - self._span_dist[i])
            )
        self.mean_aerodynamic_chord = num / denum

        # Finds Standard Mean Chord
        num = 0
        denum = 0
        for i in np.arange(0, self.N - 1):
            num += (
                (self._chord_dist[i] + self._chord_dist[i + 1])
                / 2
                * (self._span_dist[i + 1] - self._span_dist[i])
            )
            denum += self._span_dist[i + 1] - self._span_dist[i]
        self.standard_mean_chord = num / denum

    def find_aspect_ratio(self):
        """Finds the Aspect Ratio of the wing."""
        self.aspect_ratio = (self.span**2) / self.Area

    def find_area(self):
        "Finds the area of the wing."

        self.Area = 0
        self.S = 0
        rm1 = np.linalg.inv(self.R_MAT)
        for i in np.arange(0, self.N - 1):
            _, y1, _ = np.matmul(rm1, self.grid_upper[i + 1, 0, :])
            _, y2, _ = np.matmul(rm1, self.grid_upper[i, 0, :])
            self.S += (
                2 * (y1 - y2) * (self._chord_dist[i] + self._chord_dist[i + 1]) / 2
            )

        g_up = self.grid_upper
        g_low = self.grid_lower
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                AB1 = g_up[i + 1, j, :] - g_up[i, j, :]
                AB2 = g_up[i + 1, j + 1, :] - g_up[i, j + 1, :]

                AD1 = g_up[i, j + 1, :] - g_up[i, j, :]
                AD2 = g_up[i + 1, j + 1, :] - g_up[i + 1, j, :]

                Area_up = np.linalg.norm(np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2))

                AB1 = g_low[i + 1, j, :] - g_low[i, j, :]
                AB2 = g_low[i + 1, j + 1, :] - g_low[i, j + 1, :]

                AD1 = g_low[i, j + 1, :] - g_low[i, j, :]
                AD2 = g_low[i + 1, j + 1, :] - g_low[i + 1, j, :]

                Area_low = np.linalg.norm(np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2))

                self.Area += Area_up + Area_low

        # Find Aspect Ratio
        self.find_aspect_ratio()

    def find_volume(self):
        """Finds the volume of the wing. This is done by finding the volume of the wing
        as the sum of a series of panels."""

        self.VolumeDist = np.empty((self.N - 1, self.M - 1))
        self.VolumeDist2 = np.empty((self.N - 1, self.M - 1))

        g_up = self.grid_upper
        g_low = self.grid_lower
        self.AreasB = np.zeros(self.N - 1)
        self.AreasF = np.zeros(self.N - 1)

        # We divide the wing into a set of lower and upper panels that form
        # a tetrahedron. We then find the volume of each tetrahedron and sum.
        # This is equivalent to finding the area of the front and back faces of
        # the tetrahedron taking the average and multiplying by the height.
        # We then have to subtract the volume of the trianglular prism that is
        # formed by the slanted edges of the tetrahedron.
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                # Area of the front face
                AB1 = g_up[i + 1, j, :] - g_up[i, j, :]
                AB2 = g_low[i + 1, j, :] - g_low[i, j, :]

                AD1 = g_up[i, j, :] - g_low[i, j, :]
                AD2 = g_up[i + 1, j, :] - g_low[i + 1, j, :]
                Area_front_v = np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2)
                Area_front = np.linalg.norm(Area_front_v)

                # Area of the back face
                AB1 = g_up[i + 1, j + 1, :] - g_up[i, j + 1, :]
                AB2 = g_low[i + 1, j + 1, :] - g_low[i, j + 1, :]

                AD1 = g_up[i, j + 1, :] - g_low[i, j + 1, :]
                AD2 = g_up[i + 1, j + 1, :] - g_low[i + 1, j + 1, :]
                Area_back_v = np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2)
                Area_back = np.linalg.norm(Area_back_v)

                # Height of the tetrahedron
                dx1 = g_up[i, j + 1, 0] - g_up[i, j, 0]
                dx2 = g_up[i + 1, j + 1, 0] - g_up[i + 1, j, 0]
                dx3 = g_low[i, j + 1, 0] - g_low[i, j, 0]
                dx4 = g_low[i + 1, j + 1, 0] - g_low[i + 1, j, 0]
                dx = (dx1 + dx2 + dx3 + dx4) / 4

                # volume of the tetrahedron
                self.VolumeDist[i, j] = 0.5 * (Area_front + Area_back) * dx

        self.volume = np.sum(self.VolumeDist)
        if self.is_symmetric:
            self.volume = self.volume * 2

    def centerMass(self):
        """Finds the center of mass of the wing.
        This is done by summing the volume of each panel
        and dividing by the total volume."""
        x_cm = 0
        y_cm = 0
        z_cm = 0

        g_up = self.grid_upper
        g_low = self.grid_lower
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                x_upp1 = (g_up[i, j + 1, 0] + g_up[i, j, 0]) / 2
                x_upp2 = (g_up[i + 1, j + 1, 0] + g_up[i + 1, j, 0]) / 2

                x_low1 = (g_low[i, j + 1, 0] + g_low[i, j, 0]) / 2
                x_low2 = (g_low[i + 1, j + 1, 0] + g_low[i + 1, j, 0]) / 2
                x = ((x_upp1 + x_upp2) / 2 + (x_low1 + x_low2) / 2) / 2

                y_upp1 = (g_up[i + 1, j, 1] + g_up[i, j, 1]) / 2
                y_upp2 = (g_up[i + 1, j + 1, 1] + g_up[i, j + 1, 1]) / 2

                y_low1 = (g_low[i + 1, j, 1] + g_low[i, j, 1]) / 2
                y_low2 = (g_low[i + 1, j + 1, 1] + g_low[i, j + 1, 1]) / 2
                y = ((y_upp1 + y_upp2) / 2 + (y_low1 + y_low2) / 2) / 2

                z_upp1 = (g_up[i + 1, j, 2] + g_up[i + 1, j, 2]) / 2
                z_upp2 = (g_up[i + 1, j, 2] + g_up[i + 1, j, 2]) / 2

                z_low1 = (g_low[i + 1, j, 2] + g_low[i + 1, j, 2]) / 2
                z_low2 = (g_low[i + 1, j, 2] + g_low[i + 1, j, 2]) / 2
                z = ((z_upp1 + z_upp2) / 2 + (z_low1 + z_low2) / 2) / 2

                if self.is_symmetric:
                    x_cm += self.VolumeDist[i, j] * 2 * x
                    y_cm += 0
                    z_cm += self.VolumeDist[i, j] * 2 * z
                else:
                    x_cm += self.VolumeDist[i, j] * x
                    y_cm += self.VolumeDist[i, j] * y
                    z_cm += self.VolumeDist[i, j] * z

        self.CG = np.array((x_cm, y_cm, z_cm)) / self.volume

    def _inertia(self, mass, cog):
        """Finds the inertia of the wing."""
        I_xx = 0
        I_yy = 0
        I_zz = 0
        I_xz = 0
        I_xy = 0
        I_yz = 0

        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                x_upp = (self.grid_upper[i, j + 1, 0] + self.grid_upper[i, j, 0]) / 2
                x_low = (self.grid_lower[i, j + 1, 0] + self.grid_lower[i, j, 0]) / 2

                y_upp = (self.grid_upper[i + 1, j, 1] + self.grid_upper[i, j, 1]) / 2
                y_low = (self.grid_lower[i + 1, j, 1] + self.grid_lower[i, j, 1]) / 2

                z_upp1 = (
                    self.grid_upper[i + 1, j, 2] + self.grid_upper[i + 1, j, 2]
                ) / 2
                z_upp2 = (
                    self.grid_upper[i + 1, j, 2] + self.grid_upper[i + 1, j, 2]
                ) / 2
                z_upp = (z_upp1 + z_upp2) / 2

                z_low1 = (
                    self.grid_lower[i + 1, j, 2] + self.grid_lower[i + 1, j, 2]
                ) / 2
                z_low2 = (
                    self.grid_lower[i + 1, j, 2] + self.grid_lower[i + 1, j, 2]
                ) / 2
                z_low = (z_low1 + z_low2) / 2

                xd = ((x_upp + x_low) / 2 - cog[0]) ** 2
                zd = ((z_upp + z_low) / 2 - cog[2]) ** 2
                if self.is_symmetric:
                    yd = (-(y_upp + y_low) / 2 - cog[1]) ** 2
                    yd += ((y_upp + y_low) / 2 - cog[1]) ** 2
                else:
                    yd = ((y_upp + y_low) / 2 - cog[1]) ** 2

                I_xx += self.VolumeDist[i, j] * (yd + zd)
                I_yy += self.VolumeDist[i, j] * (xd + zd)
                I_zz += self.VolumeDist[i, j] * (xd + yd)

                xd = np.sqrt(xd)
                yd = np.sqrt(yd)
                zd = np.sqrt(zd)

                I_xz += self.VolumeDist[i, j] * (xd * zd)
                I_xy += self.VolumeDist[i, j] * (xd * yd)
                I_yz += self.VolumeDist[i, j] * (yd * zd)

        self.inertia = np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)) * (
            mass / self.volume
        )

    def getGrid(self, which="camber"):
        if which == "camber":
            grid = self.grid
        elif which == "upper":
            grid = self.grid_upper
        elif which == "lower":
            grid = self.grid_lower
        else:
            raise ValueError("which must be either camber, upper or lower")
        if self.is_symmetric is True:
            reflection = np.array([1, -1, 1])
            gsym = grid[::-1, :, :] * reflection
            grid = grid[1:, :, :]
            grid = np.concatenate((gsym, grid))
            pass
        return grid


def define_linear_span(
    sp: float,
    Ni: int,
) -> ndarray[Any, dtype[Any, dtype[floating[Any]]]]:
    """Returns a linearly spaced span array."""
    return np.linspace(0, sp, Ni).round(12)


def define_linear_chord(
    Ni: int,
    ch1: float,
    ch2: float,
) -> ndarray[Any, dtype[floating[Any]]]:
    """Returns a linearly spaced chord array."""
    return np.linspace(ch1, ch2, Ni).round(12)
