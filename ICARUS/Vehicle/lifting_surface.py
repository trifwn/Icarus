from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from .strip import Strip
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.utils import DiscretizationType
from ICARUS.Vehicle.utils import SymmetryAxes


class Lifting_Surface:
    """Class to reprsent a Lifting Surface."""

    def __init__(
        self,
        name: str,
        origin: FloatArray,
        orientation: FloatArray,
        root_airfoil: str | Airfoil,
        spanwise_positions: FloatArray,
        chord_lengths: FloatArray,
        z_offsets: FloatArray,
        x_offsets: FloatArray,
        twists: FloatArray,
        N: int,
        M: int,
        mass: float = 1.0,
        # Optional Parameters
        symmetries: list[SymmetryAxes] | SymmetryAxes = SymmetryAxes.NONE,
        chord_discretization_function: Callable[[int], float] | None = None,
        tip_airfoil: str | Airfoil | None = None,
    ) -> None:
        # Constructor for the Lifting Surface Class
        # The lifting surface is defined by providing the information on a number of points on the wing.
        # On must first give the origin of the wing, the orientation of the wing to define the coordinate system.
        # Relative to the origin, we take a number of points on the wing. For each point we must know:
        #   - The spanwise position of the point
        #   - The chord_length of the wing at that point
        #   - The z-offset of the point
        #   - The x-offset of the point
        #   - The twist of the wing at that point
        #   - The dihedral of the wing at that point
        #   - The airfoil at that point. The airfoil is interpolated between the root and tip airfoil.

        # Check that the number of points is the same for all parameters if not raise an error
        if not (
            len(spanwise_positions)
            == len(chord_lengths)
            == len(z_offsets)
            == len(x_offsets)
            == len(twists)
        ):
            raise ValueError("The number of points must be the same for all parameters")

        self.name: str = name

        # Define Coordinate System
        orientation = np.array(orientation, dtype=float)
        origin = np.array(origin, dtype=float)

        # Define Orientation
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

        # Define Symmetries
        if isinstance(symmetries, SymmetryAxes):
            symmetries = [symmetries]
        self.symmetries: list[SymmetryAxes] = symmetries
        self.is_symmetric_y: bool = True if SymmetryAxes.Y in symmetries else False

        # Define Discretization
        # TODO: Add logic to handle different discretization types
        self.N: int = N
        self.M: int = M
        if chord_discretization_function is None:
            self.chord_spacing: DiscretizationType = DiscretizationType.EQUAL
            # Define Chord Discretization to be the identity function
            self.chord_discretization_function: Callable[[int], float] = lambda x: x / (
                self.M - 1
            )
        else:
            self.chord_discretization_function = chord_discretization_function
            self.chord_spacing = DiscretizationType.UNKNOWN
        self.span_spacing: DiscretizationType = DiscretizationType.UNKNOWN

        # Define Chord
        self.root_chord: float = chord_lengths[0]
        self.tip_chord: float = chord_lengths[-1]
        self.chord = np.array([self.root_chord, self.tip_chord], dtype=float)

        # Get the dihedral and twist distributions
        # These are defined in the local coordinate system at the quarter chord point of each wing strip
        self.twists: FloatArray = twists

        # Define the airfoil
        if isinstance(root_airfoil, str):
            root_airfoil = Airfoil.naca(root_airfoil)
        self.root_airfoil: Airfoil = root_airfoil
        if tip_airfoil is None:
            tip_airfoil = root_airfoil
        elif isinstance(tip_airfoil, str):
            tip_airfoil = Airfoil.naca(tip_airfoil)
        self.tip_airfoil: Airfoil = tip_airfoil

        # Store Origin Parameters
        self._origin: FloatArray = origin
        self._x_origin: float = origin[0]
        self._y_origin: float = origin[1]
        self._z_origin: float = origin[2]

        self.orientation: FloatArray = orientation

        # Define the segment's mass
        self.mass: float = mass

        # Store Span
        span: float = spanwise_positions[-1] - spanwise_positions[0]
        if self.is_symmetric_y:
            self.span = span * 2
        else:
            self.span = span

        # Define Distribution of all internal variables
        self._chord_dist = chord_lengths
        self._span_dist = spanwise_positions
        self._xoffset_dist = x_offsets
        self._zoffset_dist = z_offsets

        ###### Variable Initialization ########
        # Initialize Grid Variables
        self.grid: FloatArray = np.empty((self.M, self.N, 3))  # Camber Line
        self.grid_upper: FloatArray = np.empty((self.M, self.N, 3))
        self.grid_lower: FloatArray = np.empty((self.M, self.N, 3))
        # Initialize Panel Variables
        self.panels: FloatArray = np.empty(
            (self.N - 1, self.M - 1, 4, 3)
        )  # Camber Line
        self.panels_upper: FloatArray = np.empty((self.N - 1, self.M - 1, 4, 3))
        self.panels_lower: FloatArray = np.empty((self.N - 1, self.M - 1, 4, 3))

        # Initialize Strips
        self.strips: list[Strip] = []
        self.all_strips: list[Strip] = []

        # Initialize Mean Chords
        self.mean_aerodynamic_chord: float = 0.0
        self.standard_mean_chord: float = 0.0

        # Initialize Areas
        self.S: float = 0.0
        self.area: float = 0.0

        # Initialize Volumes
        self.volume_distribution: FloatArray = np.empty((self.N - 1, self.M - 1))
        self.volume: float = 0.0

        ####### Calculate Wing Parameters #######
        self.calculate_wing_parameters()
        ####### Calculate Wing Parameters ########

    @classmethod
    def from_span_percentage_functions(
        cls,
        name: str,
        origin: FloatArray,
        orientation: FloatArray,
        root_airfoil: str | Airfoil,
        tip_airfoil: str | Airfoil,
        span: float,
        span_discretization_function: Callable[[int], float],
        chord_discretization_function: Callable[[int], float],
        chord_as_a_function_of_span_percentage: Callable[[float], float],
        x_offset_as_a_function_of_span_percentage: Callable[[float], float],
        dihedral_as_a_function_of_span_percentage: Callable[[float], float],
        twist_as_a_function_of_span_percentage: Callable[[float], float],
        N: int,
        M: int,
        mass: float = 1.0,
        # Optional Parameters
        symmetries: list[SymmetryAxes] | SymmetryAxes = SymmetryAxes.NONE,
    ) -> Lifting_Surface:
        # Define the Lifting Surface from a set of functions instead of a set of points. We must Specify 3 kind of inputs
        # 1) Basic information about the wing:
        #   - The name of the wing
        #   - The origin of the wing
        #   - The orientation of the wing
        #   - The root airfoil
        #   - The tip airfoil
        #   - The mass of the wing
        #   - The number of points N to discretize the span into
        #   - The number of points M to discretize the chord into
        #   - The symmetries of the wing
        #
        # 2) The discretization functions that define how we discretize the wing into points
        #   - The span discretization function (function of the spanwise position) that defines how we discretize the wing into points
        #   - The chord discretization function (function of the spanwise position) that defines how we discretize the chord into points
        #
        # 3) The functions that define the wing geometry. These functions take as input a parameter eta (between 0 and 1) that specifies
        # the span percentage at which the function is evaluated. The function returns the value of the parameter. The functions are:
        #   - The chord function (function of the spanwise position) that defines the chord length of the wing at each eta
        #   - The z-offset function (function of the spanwise position) that defines the z-offset of the wing at each eta
        #   - The x-offset function (function of the spanwise position) that defines the x-offset of the wing at each eta
        #   - The twist function (function of the spanwise position) that defines the twist of the wing at each eta
        #   - The dihedral function (function of the spanwise position) that defines the dihedral of the wing at each eta

        # Create the arrays that will be passed to the constructor
        spanwise_positions: FloatArray = np.empty(N, dtype=float)
        chord_lengths: FloatArray = np.empty(N, dtype=float)
        z_offsets: FloatArray = np.empty(N, dtype=float)
        x_offsets: FloatArray = np.empty(N, dtype=float)
        twists: FloatArray = np.empty(N, dtype=float)

        # Define Airfoils
        if isinstance(root_airfoil, str):
            root_airfoil = Airfoil.naca(root_airfoil)
        if isinstance(tip_airfoil, str):
            tip_airfoil = Airfoil.naca(tip_airfoil)

        # Needed for when we have airfoils that are flapped and therefore have a different chord length
        def real_chord_fun(
            eta: float,
        ) -> float:
            # TODO: Add logic to handle interpolation between root and tip airfoil
            const = float(np.max(root_airfoil._x_lower))
            return const * chord_as_a_function_of_span_percentage(eta)

        # Create the arrays that will be passed to the constructor
        for i in np.arange(0, N):
            eta = span_discretization_function(i)
            spanwise_positions[i] = eta * span
            chord_lengths[i] = real_chord_fun(eta)
            z_offsets[i] = (
                np.tan(dihedral_as_a_function_of_span_percentage(eta)) * span * eta
            )
            x_offsets[i] = x_offset_as_a_function_of_span_percentage(eta)
            twists[i] = twist_as_a_function_of_span_percentage(eta)

        self: Lifting_Surface = Lifting_Surface(
            name=name,
            origin=origin,
            orientation=orientation,
            root_airfoil=root_airfoil,
            tip_airfoil=tip_airfoil,
            spanwise_positions=spanwise_positions,
            chord_lengths=chord_lengths,
            z_offsets=z_offsets,
            x_offsets=x_offsets,
            twists=twists,
            N=N,
            M=M,
            chord_discretization_function=chord_discretization_function,
            mass=mass,
            symmetries=symmetries,
        )
        return self

    def calculate_wing_parameters(self) -> None:
        """Calculate Wing Parameters"""
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

    @property
    def origin(self) -> FloatArray:
        return self._origin
    
    @origin.setter
    def origin(self, value: FloatArray) -> None:
        movement = value - self._origin
        self._origin = value  
        
        # Move Grid
        self.grid += movement[None, None, :]
        self.grid_upper += movement[None, None, :]
        self.grid_lower += movement[None, None, :]

        # Move Panels
        self.panels += movement[None, None, :]
        self.panels_upper += movement[None, None, :]
        self.panels_lower += movement[None, None, :]

    @property
    def x_origin(self) -> float:
        return self._x_origin
    
    @x_origin.setter
    def x_origin(self, value: float) -> None:
        self._x_origin = value
        origin = np.array([value, self._y_origin, self._z_origin], dtype=float)
        self.origin = origin

    @property
    def y_origin(self) -> float:
        return self._y_origin
    
    @y_origin.setter
    def y_origin(self, value: float) -> None:
        self._y_origin = value
        origin = np.array([self._x_origin, value, self._z_origin], dtype=float)
        self.origin = origin
    
    @property
    def z_origin(self) -> float:
        return self._z_origin
    
    @z_origin.setter
    def z_origin(self, value: float) -> None:
        self._z_origin = value
        origin = np.array([self._x_origin, self._y_origin, value], dtype=float)
        self.origin = origin

    @property
    def CG(self) -> FloatArray:
        return self.find_center_mass()

    @property
    def inertia(self) -> FloatArray:
        return self.calculate_inertia(self.mass, self.CG)

    def change_discretization(self, N: int | None = None, M: int | None = None) -> None:
        if N is not None:
            self.N = N
        if M is not None:
            self.M = M
        self.calculate_wing_parameters()

    def change_mass(self, mass: float) -> None:
        """Change Wing Segment Mass"""
        self.mass = mass
        self.inertial = self.calculate_inertia(self.mass, self.CG)

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
        return self.grid_upper[0, :, :] + self._origin

    @property
    def trailing_edge(self) -> FloatArray:
        """Return Trailing Edge of Wing"""
        return self.grid_upper[-1, :, :] + self._origin

    def change_airfoil(self, airfoil: Airfoil) -> None:
        """Change airfoil of Wing"""
        self.root_airfoil = airfoil
        self.calculate_wing_parameters()

    def split_xz_symmetric_wing(self) -> tuple[Lifting_Surface, Lifting_Surface]:
        """Split Symmetric Wing into two Wings"""
        if self.is_symmetric_y:
            left = Lifting_Surface(
                name=f"L{self.name}",
                root_airfoil=self.tip_airfoil,
                origin=np.array(
                    [
                        self._origin[0] + self._zoffset_dist[-1],
                        self._origin[1] - self.span / 2,
                        self._origin[2],
                    ],
                    dtype=float,
                ),
                orientation=self.orientation,
                symmetries=[
                    symmetry
                    for symmetry in self.symmetries
                    if symmetry != SymmetryAxes.Y
                ],
                chord_lengths=self._chord_dist[::-1],
                spanwise_positions=self._span_dist[::-1],
                x_offsets=self._xoffset_dist[::-1],
                z_offsets=self._zoffset_dist[::-1],
                twists=self.twists[::-1],
                tip_airfoil=self.root_airfoil,
                N=self.N,
                M=self.M,
                mass=self.mass / 2,
            )

            right = Lifting_Surface(
                name=f"R{self.name}",
                root_airfoil=self.root_airfoil,
                origin=self._origin,
                orientation=self.orientation,
                symmetries=[
                    symmetry
                    for symmetry in self.symmetries
                    if symmetry != SymmetryAxes.Y
                ],
                chord_lengths=self._chord_dist,
                spanwise_positions=self._span_dist,
                x_offsets=self._xoffset_dist,
                z_offsets=self._zoffset_dist,
                twists=self.twists,
                tip_airfoil=self.tip_airfoil,
                N=self.N,
                M=self.M,
                mass=self.mass / 2,
            )
            return left, right
        else:
            raise ValueError("Cannot Split Body it is not symmetric")

    def create_strips(self) -> None:
        """Create Strips given the Grid and airfoil"""
        strips = []
        symmetric_strips = []

        i_range = np.arange(0, self.N - 1)

        start_points = np.array(
            [
                self._xoffset_dist[i_range],
                self._span_dist[i_range],
                self._zoffset_dist[i_range],
            ],
        )
        start_points = np.matmul(self.R_MAT, start_points) + self._origin[:, None]

        end_points = np.array(
            [
                self._xoffset_dist[i_range + 1],
                self._span_dist[i_range + 1],
                self._zoffset_dist[i_range + 1],
            ],
        )
        end_points = np.matmul(self.R_MAT, end_points) + self._origin[:, None]

        start_chords = np.array(self._chord_dist[i_range])
        end_chords = np.array(self._chord_dist[i_range + 1])

        strips = [
            Strip(
                start_leading_edge=start_points[:, j],
                end_leading_edge=end_points[:, j],
                start_airfoil=self.root_airfoil,
                end_airfoil=self.root_airfoil,
                start_chord=start_chords[j],
                end_chord=end_chords[j],
            )
            for j in range(self.N - 1)
        ]

        if self.is_symmetric_y:
            symmetric_strips = [surf.return_symmetric() for surf in strips]

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
            ax = fig.add_subplot(projection="3d")  # type: ignore
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

                    if self.is_symmetric_y:
                        ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)
        if show_plot:
            plt.show()

    def grid_to_panels(
        self, grid: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        panels = np.empty((self.N - 1, self.M - 1, 4, 3), dtype=float)
        control_points = np.empty((self.N - 1, self.M - 1, 3), dtype=float)
        control_nj = np.empty((self.N - 1, self.M - 1, 3), dtype=float)

        N = self.N
        M = self.M
        panels[:, :, 0, :] = grid[1:, : M - 1]
        panels[:, :, 1, :] = grid[: N - 1, : M - 1]
        panels[:, :, 2, :] = grid[: N - 1, 1:M]
        panels[:, :, 3, :] = grid[1:, 1:M]

        control_points[:, :, 0] = (
            self.grid[: N - 1, : M - 1, 0] + self.grid[1:N, : M - 1, 0]
        ) / 2 + 3 / 4 * (
            (self.grid[: N - 1, 1:M, 0] + self.grid[1:N, 1:M, 0]) / 2
            - (self.grid[: N - 1, : M - 1, 0] + self.grid[1:N, : M - 1, 0]) / 2
        )

        control_points[:, :, 1] = (
            self.grid[: N - 1, : M - 1, 1] + self.grid[1:N, : M - 1, 1]
        ) / 2 + 1 / 2 * (
            (self.grid[: N - 1, 1:M, 1] + self.grid[1:N, 1:M, 1]) / 2
            - (self.grid[: N - 1, : M - 1, 1] + self.grid[1:N, : M - 1, 1]) / 2
        )

        control_points[:, :, 2] = (
            self.grid[: N - 1, : M - 1, 2] + self.grid[1:N, : M - 1, 2]
        ) / 2 + 1 / 2 * (
            (self.grid[: N - 1, 1:M, 2] + self.grid[1:N, 1:M, 2]) / 2
            - (self.grid[: N - 1, : M - 1, 2] + self.grid[1:N, : M - 1, 2]) / 2
        )

        Ak = panels[:, :, 0, :] - panels[:, :, 2, :]
        Bk = panels[:, :, 1, :] - panels[:, :, 3, :]
        cross_prod = np.cross(Ak, Bk)
        control_nj[:, :, :] = (
            cross_prod / np.linalg.norm(cross_prod, axis=2)[:, :, None]
        )

        return panels, control_points, control_nj

    def create_grid(self) -> None:
        chord_eta = [
            self.chord_discretization_function(int(i)) for i in range(0, self.M)
        ]

        xs = np.outer(chord_eta, self._chord_dist) + self._xoffset_dist
        xs_upper = xs.copy()
        xs_lower = xs.copy()

        ys = np.tile(self._span_dist, (self.M, 1))
        ys_upper = ys.copy()
        ys_lower = ys.copy()

        zs_upper = (
            np.outer(self.root_airfoil.y_upper(chord_eta), self._chord_dist)
            + self._zoffset_dist
        )
        zs_lower = (
            np.outer(self.root_airfoil.y_lower(chord_eta), self._chord_dist)
            + self._zoffset_dist
        )
        zs = (
            np.outer(self.root_airfoil.camber_line(chord_eta), self._chord_dist)
            + self._zoffset_dist
        )

        # print(xs.shape, ys.shape, zs.shape)
        # Rotate according to R_MAT
        coordinates = np.matmul(
            self.R_MAT, np.vstack([xs.flatten(), ys.flatten(), zs.flatten()])
        )
        coordinates = coordinates.reshape((3, self.M, self.N))

        coordinates_upper = np.matmul(
            self.R_MAT,
            np.vstack([xs_upper.flatten(), ys_upper.flatten(), zs_upper.flatten()]),
        )
        coordinates_upper = coordinates_upper.reshape((3, self.M, self.N))

        coordinates_lower = np.matmul(
            self.R_MAT,
            np.vstack([xs_lower.flatten(), ys_lower.flatten(), zs_lower.flatten()]),
        )
        coordinates_lower = coordinates_lower.reshape((3, self.M, self.N))

        # Add origin
        coordinates += self._origin[:, None, None]
        coordinates_upper += self._origin[:, None, None]
        coordinates_lower += self._origin[:, None, None]

        # The arrays are now in the form (3, M, N), we need to transpose them to (N, M, 3)
        self.grid = coordinates.transpose(2, 1, 0)
        self.grid_upper = coordinates_upper.transpose(2, 1, 0)
        self.grid_lower = coordinates_lower.transpose(2, 1, 0)
        # error
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(
            self.grid
        )

        (
            self.panels_lower,
            self.control_points_lower,
            self.control_nj_lower,
        ) = self.grid_to_panels(self.grid_lower)

        (
            self.panels_upper,
            self.control_points_upper,
            self.control_nj_upper,
        ) = self.grid_to_panels(self.grid_upper)

    def mean_chords(self) -> None:
        "Finds the Mean Aerodynamic Chord (mean_aerodynamic_chord) of the wing."
        # Vectorized calculation for mean_aerodynamic_chord
        num = np.sum(
            ((self._chord_dist[:-1] + self._chord_dist[1:]) / 2) ** 2
            * (self._span_dist[1:] - self._span_dist[:-1]),
        )
        denum = np.sum(
            (self._chord_dist[:-1] + self._chord_dist[1:])
            / 2
            * (self._span_dist[1:] - self._span_dist[:-1]),
        )
        self.mean_aerodynamic_chord = float(num) / float(denum)

        # Vectorized calculation for standard_mean_chord
        num = np.sum(
            (self._chord_dist[:-1] + self._chord_dist[1:])
            / 2
            * (self._span_dist[1:] - self._span_dist[:-1])
        )
        denum = np.sum(self._span_dist[1:] - self._span_dist[:-1])
        self.standard_mean_chord = float(num) / float(denum)

    def find_area(self) -> None:
        """Finds the area of the wing."""

        rm1 = np.linalg.inv(self.R_MAT)

        # Vectorized calculation for the upper surface
        _, y1_upper, _ = np.matmul(rm1, self.grid_upper[1:, 0, :].T)
        _, y2_upper, _ = np.matmul(rm1, self.grid_upper[:-1, 0, :].T)
        chord_dist_upper = self._chord_dist[:-1] + self._chord_dist[1:]

        self.S = 2 * np.sum((y1_upper - y2_upper) * chord_dist_upper / 2)
        # Normalize by the maximum x value of the root airfoil
        self.S = self.S / float(np.max(self.root_airfoil._x_lower))

        g_up = self.grid_upper
        g_low = self.grid_lower

        # Vectorized calculation for the lower surface
        AB1_up = g_up[1:, :-1, :] - g_up[:-1, :-1, :]
        AB2_up = g_up[1:, 1:, :] - g_up[:-1, 1:, :]
        AD1_up = g_up[:-1, 1:, :] - g_up[:-1, :-1, :]
        AD2_up = g_up[1:, 1:, :] - g_up[1:, :-1, :]
        area_up = np.linalg.norm(
            np.cross((AB1_up + AB2_up) / 2, (AD1_up + AD2_up) / 2), axis=-1
        )

        AB1_low = g_low[1:, :-1, :] - g_low[:-1, :-1, :]
        AB2_low = g_low[1:, 1:, :] - g_low[:-1, 1:, :]
        AD1_low = g_low[:-1, 1:, :] - g_low[:-1, :-1, :]
        AD2_low = g_low[1:, 1:, :] - g_low[1:, :-1, :]
        area_low = np.linalg.norm(
            np.cross((AB1_low + AB2_low) / 2, (AD1_low + AD2_low) / 2), axis=-1
        )

        self.area = float(np.sum(area_up) + np.sum(area_low))

    def find_volume(self) -> None:
        """Finds the volume of the wing using vectorized operations."""
        g_up = self.grid_upper
        g_low = self.grid_lower

        AB1 = g_up[1:, :-1, :] - g_up[:-1, :-1, :]
        AB2 = g_low[1:, :-1, :] - g_low[:-1, :-1, :]
        AD1 = g_up[:-1, :-1, :] - g_low[:-1, :-1, :]
        AD2 = g_up[1:, :-1, :] - g_low[1:, :-1, :]
        area_front_v = np.cross((AB1 + AB2) / 2, (AD1 + AD2) / 2)
        area_front = np.linalg.norm(area_front_v, axis=2)

        AB3 = g_up[1:, 1:, :] - g_up[:-1, 1:, :]
        AB4 = g_low[1:, 1:, :] - g_low[:-1, 1:, :]
        AD3 = g_up[:-1, 1:, :] - g_low[:-1, 1:, :]
        AD4 = g_up[1:, 1:, :] - g_low[1:, 1:, :]
        area_back_v = np.cross((AB3 + AB4) / 2, (AD3 + AD4) / 2)
        area_back = np.linalg.norm(area_back_v, axis=2)

        dx1 = g_up[:-1, 1:, 0] - g_up[:-1, :-1, 0]
        dx2 = g_up[1:, 1:, 0] - g_up[1:, :-1, 0]
        dx3 = g_low[:-1, 1:, 0] - g_low[:-1, :-1, 0]
        dx4 = g_low[1:, 1:, 0] - g_low[1:, :-1, 0]
        dx = (dx1 + dx2 + dx3 + dx4) / 4

        self.volume_distribution[:, :] = 0.5 * (area_front + area_back) * dx

        self.volume = float(np.sum(self.volume_distribution))
        if self.is_symmetric_y:
            self.volume = self.volume * 2

    def find_center_mass(self) -> FloatArray:
        """Finds the center of mass of the wing using vectorized operations."""
        g_up = self.grid_upper
        g_low = self.grid_lower

        x_upp1 = (g_up[:-1, :-1, 0] + g_up[:-1, 1:, 0]) / 2
        x_upp2 = (g_up[1:, :-1, 0] + g_up[1:, 1:, 0]) / 2
        x_low1 = (g_low[:-1, :-1, 0] + g_low[:-1, 1:, 0]) / 2
        x_low2 = (g_low[1:, :-1, 0] + g_low[1:, 1:, 0]) / 2
        x = ((x_upp1 + x_upp2) / 2 + (x_low1 + x_low2) / 2) / 2

        y_upp1 = (g_up[1:, :-1, 1] + g_up[:-1, :-1, 1]) / 2
        y_upp2 = (g_up[1:, 1:, 1] + g_up[:-1, 1:, 1]) / 2
        y_low1 = (g_low[1:, :-1, 1] + g_low[:-1, :-1, 1]) / 2
        y_low2 = (g_low[1:, 1:, 1] + g_low[:-1, 1:, 1]) / 2
        y = ((y_upp1 + y_upp2) / 2 + (y_low1 + y_low2) / 2) / 2

        z_upp1 = (g_up[1:, :-1, 2] + g_up[1:, :-1, 2]) / 2
        z_upp2 = (g_up[1:, :-1, 2] + g_up[1:, :-1, 2]) / 2
        z_low1 = (g_low[1:, :-1, 2] + g_low[1:, :-1, 2]) / 2
        z_low2 = (g_low[1:, :-1, 2] + g_low[1:, :-1, 2]) / 2
        z = ((z_upp1 + z_upp2) / 2 + (z_low1 + z_low2) / 2) / 2

        if self.is_symmetric_y:
            x_cm = np.sum(self.volume_distribution[:, :] * 2 * x)
            y_cm = 0
            z_cm = np.sum(self.volume_distribution[:, :] * 2 * z)
        else:
            x_cm = np.sum(self.volume_distribution[:, :] * x)
            y_cm = np.sum(self.volume_distribution[:, :] * y)
            z_cm = np.sum(self.volume_distribution[:, :] * z)

        return np.array((x_cm, y_cm, z_cm)) / self.volume

    def calculate_inertia(self, mass: float, cog: np.ndarray) -> FloatArray:
        """
        Calculates the inertia of the wing about the center of gravity.

        Args:
            mass (float): Mass of the wing. Used to have dimensional inertia
            cog (np.ndarray): Center of Gravity of the wing.
        """
        x_upp = (self.grid_upper[:-1, :-1, 0] + self.grid_upper[:-1, 1:, 0]) / 2
        x_low = (self.grid_lower[:-1, :-1, 0] + self.grid_lower[:-1, 1:, 0]) / 2

        y_upp = (self.grid_upper[1:, :-1, 1] + self.grid_upper[:-1, :-1, 1]) / 2
        y_low = (self.grid_lower[1:, :-1, 1] + self.grid_lower[:-1, :-1, 1]) / 2

        z_upp = (self.grid_upper[1:, :-1, 2] + self.grid_upper[1:, :-1, 2]) / 2
        z_low = (self.grid_lower[1:, :-1, 2] + self.grid_lower[1:, :-1, 2]) / 2

        xd = ((x_upp + x_low) / 2 - cog[0]) ** 2
        zd = ((z_upp + z_low) / 2 - cog[2]) ** 2

        if self.is_symmetric_y:
            yd = (-(y_upp + y_low) / 2 - cog[1]) ** 2 + (
                (y_upp + y_low) / 2 - cog[1]
            ) ** 2
        else:
            yd = ((y_upp + y_low) / 2 - cog[1]) ** 2

        I_xx = np.sum(self.volume_distribution[:, :] * (yd + zd))
        I_yy = np.sum(self.volume_distribution[:, :] * (xd + zd))
        I_zz = np.sum(self.volume_distribution[:, :] * (xd + yd))

        xd = (x_upp + x_low) / 2 - cog[0]
        zd = (z_upp + z_low) / 2 - cog[2]

        if self.is_symmetric_y:
            yd = 0
        else:
            yd = (y_upp + y_low) / 2 - cog[1]

        I_xz = np.sum(self.volume_distribution[:, :] * (xd * zd))
        I_xy = np.sum(self.volume_distribution[:, :] * (xd * yd))
        I_yz = np.sum(self.volume_distribution[:, :] * (yd * zd))

        return np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)) * (mass / self.volume)

    @property
    def aspect_ratio(self) -> float:
        return (self.span**2) / self.area

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
        if self.is_symmetric_y is True:
            reflection = np.array([1, -1, 1])
            gsym = grid[::-1, :, :] * reflection
            grid = grid[1:, :, :]
            grid = np.concatenate((gsym, grid))
            pass
        return grid

    def __str__(self) -> str:
        return f"Wing Segment: {self.name} with {self.N} Panels and {self.M} Panels"
