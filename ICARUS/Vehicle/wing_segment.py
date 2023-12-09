from __future__ import annotations

from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import floating

from .strip import Strip
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray


class Wing_Segment:
    """Class to reprsent a wing segment."""

    def __init__(
        self,
        name: str,
        airfoil: Airfoil | str,
        origin: FloatArray | list[float],
        orientation: FloatArray,
        is_symmetric: bool,
        span: float,
        sweep_offset: float,
        dih_angle: float,
        chord_fun: Callable[[int, float, float], FloatArray],
        chord: FloatArray | list[float],
        span_fun: Callable[[float, int], FloatArray],
        # twist_fun: Callable[[float, int], FloatArray],
        # twist: FloatArray | list[float],
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
        if isinstance(airfoil, str):
            airfoil = Airfoil.naca(airfoil)
        self.airfoil: Airfoil = airfoil
        self.origin: FloatArray = origin
        self.orientation: FloatArray = orientation
        self.is_symmetric: bool = is_symmetric
        self.span: float = span
        self.sweep_offset: float = sweep_offset
        self.dih_angle: float = dih_angle
        self.chord_fun: Callable[[int, float, float], FloatArray] = chord_fun
        self.chord = chord
        self.span_fun: Callable[[float, int], FloatArray] = span_fun
        self.mass: float = mass

        self.gamma: float = dih_angle * np.pi / 180

        # orientation
        self.pitch, self.yaw, self.roll = orientation * np.pi / 180
        R_PITCH: FloatArray = np.array(
            [
                [np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)],
            ],
        )
        R_YAW: FloatArray = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ],
        )
        R_ROLL: FloatArray = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)],
            ],
        )
        self.R_MAT: FloatArray = R_YAW.dot(R_PITCH).dot(R_ROLL)

        # Make Dihedral Angle Distribution
        if is_symmetric:
            self._chord_dist: FloatArray = self.real_chord_fun(
                self.N,
                *chord,
            )
            self._span_dist: FloatArray = span_fun(
                span / 2,
                self.N,
            )
            self._offset_dist = (self._span_dist) * (sweep_offset / (span / 2))
            self._dihedral_dist: FloatArray = (self._span_dist - span / 2) * np.sin(
                self.gamma,
            )
            self._dihedral_dist = -self._dihedral_dist[::-1]
        else:
            self._chord_dist = self.real_chord_fun(self.N, *chord)
            self._span_dist = span_fun(span, self.N)
            self._offset_dist = self._span_dist * sweep_offset / span
            self._dihedral_dist = self._span_dist * np.sin(self.gamma)

        # Initialize Grid Variables for typing purposes
        self.grid: FloatArray = np.empty((self.M, self.N, 3))  # Camber Line
        self.grid_upper: FloatArray = np.empty((self.M, self.N, 3))
        self.grid_lower: FloatArray = np.empty((self.M, self.N, 3))
        # Initialize Panel Variables for typing purposes
        self.panels: FloatArray = np.empty(
            (self.N - 1, self.M - 1, 4, 3),
        )  # Camber Line
        self.panels_upper: FloatArray = np.empty((self.N - 1, self.M - 1, 4, 3))
        self.panels_lower: FloatArray = np.empty((self.N - 1, self.M - 1, 4, 3))
        # Create Grid
        self.create_grid()

        # Create Surfaces
        self.strips: list[Strip] = []
        self.all_strips: list[Strip] = []
        self.create_strips()

        # Find Chords mean_aerodynamic_chord-standard_mean_chord
        self.mean_aerodynamic_chord: float = 0.0
        self.standard_mean_chord: float = 0.0
        self.mean_chords()

        # Calculate Areas
        self.S: float = 0.0
        self.area: float = 0.0
        self.aspect_ratio: float = 0.0
        self.find_area()

        # Calculate Volumes
        self.volume_distribution: FloatArray = np.empty((self.N - 1, self.M - 1))
        self.volume_distribution_2: FloatArray = np.empty((self.N - 1, self.M - 1))
        self.volume: float = 0.0
        self.find_volume()

        # Find Center of Mass
        self.CG: FloatArray = np.empty(3, dtype=float)
        self.find_center_mass()

        # Calculate Moments
        self.inertia: FloatArray = np.empty((6), dtype=float)
        self.calculate_inertia(self.mass, self.CG)

    def change_discretization(self, N: int | None = None, M: int | None = None) -> None:
        if N is not None:
            self.N = N
        if M is not None:
            self.M = M
        self.create_grid()
        self.create_strips()
        self.mean_chords()
        self.find_area()
        self.find_volume()
        self.find_center_mass()
        self.calculate_inertia(self.mass, self.CG)

    def change_mass(self, mass: float) -> None:
        """Change Wing Segment Mass"""
        self.mass = mass
        self.calculate_inertia(self.mass, self.CG)

    @property
    def tip(self) -> FloatArray:
        """Return Tip of Wing. We basically returns the tip strip of the wing."""
        return self.strips[-1].get_tip_strip()

    @property
    def root(self) -> FloatArray:
        """Return Root of Wing. We basically returns the root strip of the wing."""
        return self.strips[0].get_root_strip()

    @property
    def leading_edge(self) -> FloatArray:
        """Return Leading Edge of Wing"""
        return self.grid_upper[0, :, :] + self.origin

    @property
    def trailing_edge(self) -> FloatArray:
        """Return Trailing Edge of Wing"""
        return self.grid_upper[-1, :, :] + self.origin

    def real_chord_fun(
        self,
        Ni: int,
        ch1: float,
        ch2: float,
    ) -> FloatArray:
        const = float(np.max(self.airfoil._x_lower))
        return self.chord_fun(Ni, const * ch1, const * ch2)

    def change_airfoil(self, airfoil: Airfoil) -> None:
        """Change airfoil of Wing"""
        self.airfoil = airfoil
        self.create_grid()
        self.create_strips()
        self.mean_chords()
        self.find_area()
        self.find_volume()
        self.find_center_mass()
        self.calculate_inertia(self.mass, self.CG)

    def split_symmetric_wing(self) -> tuple[Wing_Segment, Wing_Segment]:
        """Split Symmetric Wing into two Wings"""
        if self.is_symmetric:
            left = Wing_Segment(
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

            right = Wing_Segment(
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
            raise ValueError("Cannot Split Body it is not symmetric")

    def create_strips(self) -> None:
        """Create Strips given the Grid and airfoil"""
        strips: list[Strip] = []
        symmetric_strips: list[Strip] = []
        for i in np.arange(0, self.N - 1):
            start_point: FloatArray = np.array(
                [
                    self._offset_dist[i],
                    self._span_dist[i],
                    self._dihedral_dist[i],
                ],
            )
            start_point = np.matmul(self.R_MAT, start_point) + self.origin

            end_point: FloatArray = np.array(
                [
                    self._offset_dist[i + 1],
                    self._span_dist[i + 1],
                    self._dihedral_dist[i + 1],
                ],
            )
            end_point = np.matmul(self.R_MAT, end_point) + self.origin
            surf = Strip(
                start_leading_edge=start_point,
                end_leading_edge=end_point,
                start_airfoil=self.airfoil,
                end_airfoil=self.airfoil,
                start_chord=float(self._chord_dist[i]),
                end_chord=float(self._chord_dist[i + 1]),
            )
            if self.is_symmetric:
                strips.append(surf)
                symmetric_strips.append(surf.return_symmetric())
            else:
                strips.append(surf)
        self.strips = strips
        self.all_strips = [*strips, *symmetric_strips]

    def plot(
        self,
        thin: bool = False,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        prev_movement: FloatArray | None = None,
    ) -> None:
        """Plot Wing in 3D"""
        show_plot: bool = False

        if isinstance(prev_fig, Figure) and isinstance(prev_ax, Axes3D):
            fig: Figure = prev_fig
            ax: Axes3D = prev_ax
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_title(self.name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.axis("scaled")
            ax.view_init(30, 150)
            show_plot = True

        if prev_movement is None:
            movement: FloatArray = np.zeros(3)
        else:
            movement = prev_movement

        # for strip in self.all_strips:
        #     strip.plot(fig, ax, movement)

        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                if thin:
                    items = [self.panels]
                else:
                    items = [self.panels_lower, self.panels_upper]
                for item in items:
                    p1, p3, p4, p2 = item[i, j, :, :]
                    xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2)) + movement[0]

                    ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2)) + movement[1]

                    zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2)) + movement[2]

                    ax.plot_wireframe(xs, ys, zs, linewidth=0.5)

                    if self.is_symmetric:
                        ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)
        if show_plot:
            plt.show()

    def grid_to_panels(self, grid: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Convert Grid to Panels

        Args:
            grid (FloatArray): Grid to convert

        Returns:
            tuple[FloatArray, FloatArray, FloatArray]: Panels, Control Points, Control Normal Vectors
        """
        panels: FloatArray = np.empty((self.N - 1, self.M - 1, 4, 3), dtype=float)
        control_points: FloatArray = np.empty((self.N - 1, self.M - 1, 3), dtype=float)
        control_nj: FloatArray = np.empty((self.N - 1, self.M - 1, 3), dtype=float)
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                panels[i, j, 0, :] = grid[i + 1, j]
                panels[i, j, 1, :] = grid[i, j]
                panels[i, j, 2, :] = grid[i, j + 1]
                panels[i, j, 3, :] = grid[i + 1, j + 1]

                control_points[i, j, 0] = (self.grid[i, j][0] + self.grid[i + 1, j][0]) / 2 + 3 / 4 * (
                    (self.grid[i, j + 1][0] + self.grid[i + 1, j + 1][0]) / 2
                    - (self.grid[i, j][0] + self.grid[i + 1, j][0]) / 2
                )
                control_points[i, j, 1] = (self.grid[i, j][1] + self.grid[i + 1, j][1]) / 2 + 1 / 2 * (
                    (self.grid[i, j + 1][1] + self.grid[i + 1, j + 1][1]) / 2
                    - (self.grid[i, j][1] + self.grid[i + 1, j][1]) / 2
                )
                control_points[i, j, 2] = (self.grid[i, j][2] + self.grid[i + 1, j][2]) / 2 + 1 / 2 * (
                    (self.grid[i, j + 1][2] + self.grid[i + 1, j + 1][2]) / 2
                    - (self.grid[i, j][2] + self.grid[i + 1, j][2]) / 2
                )
                Ak = panels[i, j, 0, :] - panels[i, j, 2, :]
                Bk = panels[i, j, 1, :] - panels[i, j, 3, :]
                cross_prod = np.cross(Ak, Bk)
                control_nj[i, j, :] = cross_prod / np.linalg.norm(cross_prod)
        return panels, control_points, control_nj

    def create_grid(self) -> None:
        """Create Grid for Wing"""
        xs: FloatArray = np.empty((self.M, self.N), dtype=float)
        xs_upper: FloatArray = np.empty((self.M, self.N), dtype=float)
        xs_lower: FloatArray = np.empty((self.M, self.N), dtype=float)

        ys: FloatArray = np.empty((self.M, self.N), dtype=float)
        ys_upper: FloatArray = np.empty((self.M, self.N), dtype=float)
        ys_lower: FloatArray = np.empty((self.M, self.N), dtype=float)

        zs: FloatArray = np.empty((self.M, self.N), dtype=float)
        zs_upper: FloatArray = np.empty((self.M, self.N), dtype=float)
        zs_lower: FloatArray = np.empty((self.M, self.N), dtype=float)

        for i in np.arange(0, self.M):
            xpos: FloatArray = (self._chord_dist) * (i / (self.M - 1))
            xs[i, :] = self._offset_dist + xpos
            xs_lower[i, :] = xs[i, :]
            xs_upper[i, :] = xs[i, :]

            ys[i, :] = self._span_dist * np.cos(self.gamma)
            ys_lower[i, :] = ys[i, :]
            ys_upper[i, :] = ys[i, :]

            for j in np.arange(0, self.N):
                zs_upper[i, j] = self._dihedral_dist[j] + self._chord_dist[j] * self.airfoil.y_upper(i / (self.M - 1))
                zs_lower[i, j] = self._dihedral_dist[j] + self._chord_dist[j] * self.airfoil.y_lower(i / (self.M - 1))
                zs[i, j] = self._dihedral_dist[j] + self._chord_dist[j] * self.airfoil.camber_line(
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

        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)

        (self.panels_lower, self.control_points_lower, self.control_nj_lower) = self.grid_to_panels(self.grid_lower)

        (self.panels_upper, self.control_points_upper, self.control_nj_upper) = self.grid_to_panels(self.grid_upper)

    def mean_chords(self) -> None:
        "Finds the Mean Aerodynamic Chord (mean_aerodynamic_chord) of the wing."
        num: float = 0
        denum: float = 0
        for i in np.arange(0, self.N - 1):
            num += float(
                ((self._chord_dist[i] + self._chord_dist[i + 1]) / 2) ** 2
                * (self._span_dist[i + 1] - self._span_dist[i]),
            )
            denum += float(
                (self._chord_dist[i] + self._chord_dist[i + 1]) / 2 * (self._span_dist[i + 1] - self._span_dist[i]),
            )
        self.mean_aerodynamic_chord = num / denum

        # Finds Standard Mean Chord
        num = 0
        denum = 0
        for i in np.arange(0, self.N - 1):
            num += float(
                (self._chord_dist[i] + self._chord_dist[i + 1]) / 2 * (self._span_dist[i + 1] - self._span_dist[i]),
            )
            denum += float(self._span_dist[i + 1] - self._span_dist[i])
        self.standard_mean_chord = num / denum

    def find_aspect_ratio(self) -> None:
        """Finds the Aspect Ratio of the wing."""
        self.aspect_ratio = (self.span**2) / self.area

    def find_area(self) -> None:
        "Finds the area of the wing."

        rm1: FloatArray = np.linalg.inv(self.R_MAT)
        for i in np.arange(0, self.N - 1):
            _, y1, _ = np.matmul(rm1, self.grid_upper[i + 1, 0, :])
            _, y2, _ = np.matmul(rm1, self.grid_upper[i, 0, :])
            self.S += 2 * (y1 - y2) * (self._chord_dist[i] + self._chord_dist[i + 1]) / 2
        self.S = float(self.S / np.max(self.airfoil._x_lower))

        g_up = self.grid_upper
        g_low = self.grid_lower
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                AB1 = g_up[i + 1, j, :] - g_up[i, j, :]
                AB2 = g_up[i + 1, j + 1, :] - g_up[i, j + 1, :]

                AD1 = g_up[i, j + 1, :] - g_up[i, j, :]
                AD2 = g_up[i + 1, j + 1, :] - g_up[i + 1, j, :]

                Area_up: floating[Any] = np.linalg.norm(
                    np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2),
                )

                AB1 = g_low[i + 1, j, :] - g_low[i, j, :]
                AB2 = g_low[i + 1, j + 1, :] - g_low[i, j + 1, :]

                AD1 = g_low[i, j + 1, :] - g_low[i, j, :]
                AD2 = g_low[i + 1, j + 1, :] - g_low[i + 1, j, :]

                area_low: floating[Any] = np.linalg.norm(
                    np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2),
                )

                self.area += float(Area_up + area_low)

        # Find Aspect Ratio
        self.find_aspect_ratio()

    def find_volume(self) -> None:
        """Finds the volume of the wing. This is done by finding the volume of the wing
        as the sum of a series of panels."""

        g_up = self.grid_upper
        g_low = self.grid_lower
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
                area_front_v: FloatArray = np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2)
                area_front: floating[Any] = np.linalg.norm(area_front_v)

                # Area of the back face
                AB3 = g_up[i + 1, j + 1, :] - g_up[i, j + 1, :]
                AB4 = g_low[i + 1, j + 1, :] - g_low[i, j + 1, :]

                AD3 = g_up[i, j + 1, :] - g_low[i, j + 1, :]
                AD4 = g_up[i + 1, j + 1, :] - g_low[i + 1, j + 1, :]
                area_back_v: FloatArray = np.cross((AB3 + AB4) / 2, (AD3 + AD4) / 2)
                area_back: floating[Any] = np.linalg.norm(area_back_v)

                # Height of the tetrahedron
                dx1: float = g_up[i, j + 1, 0] - g_up[i, j, 0]
                dx2: float = g_up[i + 1, j + 1, 0] - g_up[i + 1, j, 0]
                dx3: float = g_low[i, j + 1, 0] - g_low[i, j, 0]
                dx4: float = g_low[i + 1, j + 1, 0] - g_low[i + 1, j, 0]
                dx: float = (dx1 + dx2 + dx3 + dx4) / 4

                # volume of the tetrahedron
                self.volume_distribution[i, j] = 0.5 * (area_front + area_back) * dx

        self.volume = float(np.sum(self.volume_distribution))
        if self.is_symmetric:
            self.volume = self.volume * 2

    def find_center_mass(self) -> None:
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
                    x_cm += self.volume_distribution[i, j] * 2 * x
                    y_cm += 0
                    z_cm += self.volume_distribution[i, j] * 2 * z
                else:
                    x_cm += self.volume_distribution[i, j] * x
                    y_cm += self.volume_distribution[i, j] * y
                    z_cm += self.volume_distribution[i, j] * z

        self.CG = np.array((x_cm, y_cm, z_cm)) / self.volume

    def calculate_inertia(self, mass: float, cog: FloatArray) -> None:
        """
        Calculates the inertia of the wing about the center of gravity.

        Args:
            mass (float): Mass of the wing. Used to have dimensional inertia
            cog (FloatArray): Center of Gravity of the wing.
        """
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

                z_upp1 = (self.grid_upper[i + 1, j, 2] + self.grid_upper[i + 1, j, 2]) / 2
                z_upp2 = (self.grid_upper[i + 1, j, 2] + self.grid_upper[i + 1, j, 2]) / 2
                z_upp = (z_upp1 + z_upp2) / 2

                z_low1 = (self.grid_lower[i + 1, j, 2] + self.grid_lower[i + 1, j, 2]) / 2
                z_low2 = (self.grid_lower[i + 1, j, 2] + self.grid_lower[i + 1, j, 2]) / 2
                z_low = (z_low1 + z_low2) / 2

                xd = ((x_upp + x_low) / 2 - cog[0]) ** 2
                zd = ((z_upp + z_low) / 2 - cog[2]) ** 2
                if self.is_symmetric:
                    yd = (-(y_upp + y_low) / 2 - cog[1]) ** 2
                    yd += ((y_upp + y_low) / 2 - cog[1]) ** 2
                else:
                    yd = ((y_upp + y_low) / 2 - cog[1]) ** 2

                I_xx += self.volume_distribution[i, j] * (yd + zd)
                I_yy += self.volume_distribution[i, j] * (xd + zd)
                I_zz += self.volume_distribution[i, j] * (xd + yd)

                xd = (x_upp + x_low) / 2 - cog[0]
                zd = (z_upp + z_low) / 2 - cog[2]
                if self.is_symmetric:
                    yd = 0
                else:
                    yd = ((y_upp + y_low) / 2 - cog[1])

                I_xz += self.volume_distribution[i, j] * (xd * zd)
                I_xy += self.volume_distribution[i, j] * (xd * yd)
                I_yz += self.volume_distribution[i, j] * (yd * zd)
        self.inertia = np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)) * (mass / self.volume)

    @property
    def Ixx(self) -> float:
        return float(self.inertia[0])

    @property
    def Iyy(self) -> float:
        return float(self.inertia[1])

    @property
    def Izz(self) -> float:
        return float(self.inertia[2])

    @property
    def Ixz(self) -> float:
        return float(self.inertia[3])

    @property
    def Ixy(self) -> float:
        return float(self.inertia[4])

    @property
    def Iyz(self) -> float:
        return float(self.inertia[5])

    def get_grid(self, which: str = "camber") -> FloatArray:
        """
        Returns the Grid of the Wing.

        Args:
            which (str, optional): upper, lower or camber. Defaults to "camber".

        Raises:
            ValueError: If which is not upper, lower or camber.

        Returns:
            FloatArray: Grid of the Wing.
        """
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

    def __str__(self):
        return f"Wing Segment: {self.name} with {self.N} Panels and {self.M} Panels"


def define_linear_span(
    sp: float,
    Ni: int,
) -> FloatArray:
    """Returns a linearly spaced span array."""
    return np.linspace(0, sp, Ni).round(12)


def define_linear_chord(
    Ni: int,
    ch1: float,
    ch2: float,
) -> FloatArray:
    """Returns a linearly spaced chord array."""
    return np.linspace(ch1, ch2, Ni).round(12)
