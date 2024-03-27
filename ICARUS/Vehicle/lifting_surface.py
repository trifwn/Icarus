from __future__ import annotations

from logging import root
from math import pi
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from .strip import Strip
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
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
        twist_angles: FloatArray,
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
            == len(twist_angles)
        ):
            raise ValueError("The number of points must be the same for all parameters")

        self.name: str = name
        # Define Coordinate System
        orientation = np.array(orientation, dtype=float)
        origin = np.array(origin, dtype=float)

        # Define Orientation
        pitch, yaw, roll = orientation * np.pi / 180
        self._pitch: float = pitch
        self._yaw: float = yaw
        self._roll: float = roll

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
        self.num_panels: int = (self.N - 1) * (self.M - 1)
        self.num_grid_points: int = self.N * self.M

        if chord_discretization_function is None:
            self.chord_spacing: DiscretizationType = DiscretizationType.EQUAL
            # Define Chord Discretization to be the identity function
            self.chord_discretization_function: Callable[[int], float] = lambda x: x / (
                self.M - 1
            )
        else:
            self.chord_discretization_function = chord_discretization_function
            self.chord_spacing = DiscretizationType.USER_DEFINED

        # Define Chord
        self._root_chord: float = chord_lengths[0]
        self._tip_chord: float = chord_lengths[-1]
        self.chords: list[float] = [chord_lengths[0], chord_lengths[-1]]

        # Get the twist distributions
        # These are defined in the local coordinate system at the quarter chord point of each wing strip
        self.twist_angles: FloatArray = twist_angles

        # Define the airfoil
        if isinstance(root_airfoil, str):
            root_airfoil = DB.get_airfoil(root_airfoil)
        self._root_airfoil: Airfoil = root_airfoil
        if tip_airfoil is None:
            tip_airfoil = root_airfoil
        elif isinstance(tip_airfoil, str):
            tip_airfoil = DB.get_airfoil(tip_airfoil)
        self._tip_airfoil: Airfoil = tip_airfoil
        self.airfoils: list[Airfoil] = [root_airfoil, tip_airfoil]

        # Store Origin Parameters
        self._origin: FloatArray = origin
        self._x_origin: float = origin[0]
        self._y_origin: float = origin[1]
        self._z_origin: float = origin[2]

        self._orientation: FloatArray = orientation

        # Define the segment's mass
        self._mass: float = mass

        # Store Span
        span: float = spanwise_positions[-1] - spanwise_positions[0]
        if self.is_symmetric_y:
            self._span = span * 2
        else:
            self._span = span

        # Define Distribution of all internal variables
        self._chord_dist = chord_lengths
        self._span_dist = spanwise_positions
        self._xoffset_dist = x_offsets
        self._zoffset_dist = z_offsets

        ###### Variable Initialization ########
        # Initialize Grid Variables
        self.grid: FloatArray = np.empty((self.num_grid_points, 3))  # Camber Line
        self.grid_upper: FloatArray = np.empty((self.num_grid_points, 3))
        self.grid_lower: FloatArray = np.empty((self.num_grid_points, 3))
        # Initialize Panel Variables
        self.panels: FloatArray = np.empty((self.num_panels, 4, 3))  # Camber Line
        self.panels_upper: FloatArray = np.empty((self.num_panels, 4, 3))
        self.panels_lower: FloatArray = np.empty((self.num_panels, 4, 3))

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
        self.volume_distribution: FloatArray = np.empty(self.num_panels)
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
        # 1) Basic information about the wi:g win thee ofng:
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
        twist_angles: FloatArray = np.empty(N, dtype=float)

        # Define Airfoils
        if isinstance(root_airfoil, str):
            root_airfoil_obj: Airfoil = DB.get_airfoil(root_airfoil)
        elif isinstance(root_airfoil, Airfoil):
            root_airfoil_obj = root_airfoil
        else:
            print(root_airfoil)
            print(type(root_airfoil))
            raise ValueError("Root Airfoil must be a string or an Airfoil")

        if isinstance(tip_airfoil, str):
            tip_airfoil_obj: Airfoil = DB.get_airfoil(tip_airfoil)
        elif isinstance(tip_airfoil, Airfoil):
            tip_airfoil_obj = tip_airfoil
        else:
            print(tip_airfoil)
            print(type(tip_airfoil))
            raise ValueError("Tip Airfoil must be a string or an Airfoil")

        # Needed for when we have airfoils that are flapped and therefore have a different chord length
        def real_chord_fun(
            eta: float,
        ) -> float:
            # TODO: Add logic to handle interpolation between root and tip airfoil
            const = float(np.max(root_airfoil_obj._x_lower))
            return const * chord_as_a_function_of_span_percentage(eta)

        if isinstance(symmetries, SymmetryAxes):
            symmetries = [symmetries]

        if SymmetryAxes.Y in symmetries:
            span = span / 2

        # Create the arrays that will be passed to the constructor
        for i in np.arange(0, N):
            eta = span_discretization_function(i)
            spanwise_positions[i] = eta * span
            chord_lengths[i] = real_chord_fun(eta)
            z_offsets[i] = (
                np.tan(dihedral_as_a_function_of_span_percentage(eta)) * span * eta
            )
            x_offsets[i] = x_offset_as_a_function_of_span_percentage(eta)
            twist_angles[i] = twist_as_a_function_of_span_percentage(eta)

        self: Lifting_Surface = Lifting_Surface(
            name=name,
            origin=origin,
            orientation=orientation,
            root_airfoil=root_airfoil_obj,
            tip_airfoil=tip_airfoil_obj,
            spanwise_positions=spanwise_positions,
            chord_lengths=chord_lengths,
            z_offsets=z_offsets,
            x_offsets=x_offsets,
            twist_angles=twist_angles,
            N=N,
            M=M,
            chord_discretization_function=chord_discretization_function,
            mass=mass,
            symmetries=symmetries,
        )
        return self

    @property
    def origin(self) -> FloatArray:
        return self._origin

    @origin.setter
    def origin(self, value: FloatArray) -> None:
        movement = value - self._origin
        self._origin = value

        grid =  self.grid.reshape(self.N, self.M, 3)
        grid_upper = self.grid_upper.reshape(self.N, self.M, 3)
        grid_lower = self.grid_lower.reshape(self.N, self.M, 3)

        # Move Grid
        grid += movement[None, None, :]
        grid_upper += movement[None, None, :]
        grid_lower += movement[None, None, :]

        self.grid = grid.reshape(-1, 3)
        self.grid_upper = grid_upper.reshape(-1, 3)
        self.grid_lower = grid_lower.reshape(-1, 3)

        # Move Panels
        panels = self.panels.reshape(self.num_panels, 4, 3)
        panels_upper = self.panels_upper.reshape(self.num_panels, 4, 3)
        panels_lower = self.panels_lower.reshape(self.num_panels, 4, 3)

        panels += movement[None, None, :]
        panels_upper += movement[None, None, :]
        panels_lower += movement[None, None, :]

        self.panels = panels.reshape(-1, 4, 3)
        self.panels_upper = panels_upper.reshape(-1, 4, 3)
        self.panels_lower = panels_lower.reshape(-1, 4, 3)

        # Create New Strips
        self.create_strips()

    @property
    def span(self) -> float:
        return self._span

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
    def orientation(self) -> FloatArray:
        return self._orientation

    @orientation.setter
    def orientation(self, value: FloatArray) -> None:
        self._orientation = value
        self._pitch, self._yaw, self._roll = value * np.pi / 180
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
        R_MAT = R_YAW.dot(R_PITCH).dot(R_ROLL)

        # Rotate Grid
        # grid has shape (N, M, 3)
        # R_MAT has shape (3, 3)
        inverseR_MAT = np.linalg.inv(self.R_MAT)

        # Step 1: Flatten the matrix of grid points
        flattened_points = self.grid.reshape(-1, 3)
        # Step 2: DeRotate the points by the old R_MAT
        rotated_points = np.dot(flattened_points, inverseR_MAT.T)
        # Step 3: Rotate the points by the new R_MAT
        rotated_points = np.dot(rotated_points, R_MAT.T)
        # Step 4: Reshape the array back to the original shape
        self.grid = rotated_points.reshape(self.grid.shape)

        # Do the same for the upper and lower surfaces
        flattened_points = self.grid_upper.reshape(-1, 3)
        rotated_points = np.dot(flattened_points, inverseR_MAT.T)
        rotated_points = np.dot(rotated_points, R_MAT.T)
        self.grid_upper = rotated_points.reshape(self.grid_upper.shape)

        flattened_points = self.grid_lower.reshape(-1, 3)
        rotated_points = np.dot(flattened_points, inverseR_MAT.T)
        rotated_points = np.dot(rotated_points, R_MAT.T)
        self.grid_lower = rotated_points.reshape(self.grid_lower.shape)

        # Rotate Panels
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

        self.create_strips()

        # Store the new R_MAT
        self.R_MAT = R_MAT

    @property
    def pitch(self) -> float:
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._pitch = value
        self.orientation = np.array([self._pitch, self._yaw, self._roll])

    @property
    def yaw(self) -> float:
        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._yaw = value
        self.orientation = np.array([self._pitch, self._yaw, self._roll])

    @property
    def roll(self) -> float:
        return self._roll

    @roll.setter
    def roll(self, value: float) -> None:
        self._roll = value
        self.orientation = np.array([self._pitch, self._yaw, self._roll])

    @property
    def CG(self) -> FloatArray:
        return self.calculate_center_mass()

    @property
    def inertia(self) -> FloatArray:
        return self.calculate_inertia(self.mass, self.CG)

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

    @property
    def aspect_ratio(self) -> float:
        return (self.span**2) / self.S

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

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = value

    @property
    def root_airfoil(self) -> Airfoil:
        return self._root_airfoil

    @root_airfoil.setter
    def root_airfoil(self, value: str | Airfoil) -> None:
        if isinstance(value, str):
            value = DB.get_airfoil(value)
        self._root_airfoil = value
        self.calculate_wing_parameters()
        return None

    @property
    def tip_airfoil(self) -> Airfoil:
        return self._tip_airfoil

    @tip_airfoil.setter
    def tip_airfoil(self, value: str | Airfoil) -> None:
        if isinstance(value, str):
            value = DB.get_airfoil(value)
        self._tip_airfoil = value
        self.calculate_wing_parameters()

    @property
    def mean_airfoil(self) -> Airfoil:
        # Calcultate the area weighted mean airfoil by the chord
        dspan = self._span_dist[1:] - self._span_dist[:-1]
        mchord = (self._chord_dist[1:] + self._chord_dist[:-1]) / 2
        area_approx = dspan * mchord
        mean_area = np.mean(dspan * mchord)
        mean_area_pos = np.argmin(np.abs(area_approx - mean_area))
        heta = (self._span_dist[mean_area_pos] - self._span_dist[0]) / (
            self._span_dist[-1] - self._span_dist[0]
        )
        return Airfoil.morph_new_from_two_foils(
            self.root_airfoil,
            self.tip_airfoil,
            heta,
            self.root_airfoil.n_points,
        )

    def calculate_wing_parameters(self) -> None:
        """Calculate Wing Parameters"""
        # Create Grid
        self.create_grid()

        # Create Surfaces
        self.create_strips()

        # Calculate Areas
        self.calculate_area()

        # Find Chords mean_aerodynamic_chord-standard_mean_chord
        self.calculate_mean_chords()

        # Calculate Volumes
        self.calculate_volume()

    def change_discretization(self, N: int | None = None, M: int | None = None) -> None:
        if N is not None:
            self.N = N
        if M is not None:
            self.M = M
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
                twist_angles=self.twist_angles[::-1],
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
                twist_angles=self.twist_angles,
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

        strips: list[Strip] = []
        for j in range(self.N - 1):
            if self.root_airfoil == self.tip_airfoil:
                strip_root_af = self.root_airfoil
                strip_tip_af = self.root_airfoil
            else:
                # Calculate the heta position of the strip
                root_eta = j / (self.N - 1)
                tip_eta = (j + 1) / (self.N - 1)
                strip_root_af = Airfoil.morph_new_from_two_foils(
                    self.root_airfoil,
                    self.tip_airfoil,
                    root_eta,
                    self.root_airfoil.n_points,
                )
                strip_tip_af = Airfoil.morph_new_from_two_foils(
                    self.root_airfoil,
                    self.tip_airfoil,
                    tip_eta,
                    self.tip_airfoil.n_points,
                )
                
            # Based on the shape (area) of the strip, we can calculate the eta position of the strip
            # We can then calculate the mean aerodynamic chord of the strip
            strip_mean_aerodynamic_chord = (
                (start_chords[j] + end_chords[j]) / 2
            )
            strip_half_span = (self._span_dist[j + 1] - self._span_dist[j]) / 2
            eta = 1/2* strip_half_span * (start_chords[j] + strip_mean_aerodynamic_chord) / (
                strip_half_span * (start_chords[j] + end_chords[j])
            )

            strip = Strip(
                    # Define left part of the strip
                    start_leading_edge=start_points[:, j],
                    start_airfoil=strip_root_af,
                    start_chord=start_chords[j],
                    start_twist=self.twist_angles[j],
                    # Define right part of the strip
                    end_leading_edge=end_points[:, j],
                    end_airfoil=strip_tip_af,
                    end_chord=end_chords[j],
                    end_twist=self.twist_angles[j+1],
                    eta=eta,
                )
            strips.append(strip)
        if self.is_symmetric_y:
            symmetric_strips = [strip.return_symmetric() for strip in strips]

        self.strips = strips
        self.all_strips = [*strips, *symmetric_strips]

    def create_grid(self) -> None:
        chord_eta = np.array(
            [self.chord_discretization_function(int(i)) for i in range(0, self.M)]
        )
        # span_eta = np.array(
        #     [self.spanwise_positions[int(i)] / self.span for i in range(0, self.N)]
        # )

        xs = np.outer(chord_eta, self._chord_dist) + self._xoffset_dist
        xs_upper = xs.copy()
        xs_lower = xs.copy()

        ys = np.tile(self._span_dist, (self.M, 1))
        ys_upper = ys.copy()
        ys_lower = ys.copy()

        # Calculate the airfoil at the position of the strip based on eta
        airf_z_up = chord_eta * self.root_airfoil.y_upper(chord_eta) + (
            1 - chord_eta
        ) * self.tip_airfoil.y_upper(chord_eta)

        airf_z_low = chord_eta * self.root_airfoil.y_lower(chord_eta) + (
            1 - chord_eta
        ) * self.tip_airfoil.y_lower(chord_eta)
        airf_camber = chord_eta * self.root_airfoil.camber_line(chord_eta) + (
            1 - chord_eta
        ) * self.tip_airfoil.camber_line(chord_eta)

        zs_upper = np.outer(airf_z_up, self._chord_dist) + self._zoffset_dist
        zs_lower = np.outer(airf_z_low, self._chord_dist) + self._zoffset_dist
        zs = np.outer(airf_camber, self._chord_dist) + self._zoffset_dist

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

        # Rotate according to twist distribution:
        # twist is a list of angles in degrees that are applied at the quarter chord point of each strip
        # We need to rotate the grid points around the quarter chord point of each strip
        # We first need to find the quarter chord point of each strip
        # We then need to rotate the grid points around the quarter chord point of each strip
        c_4 = coordinates[:, 0, :] + (coordinates[:, -1, :] - coordinates[:, 0, :]) / 4

        rotated_coordinates = np.empty((3, self.M, self.N))
        rotated_coordinates_upper = np.empty((3, self.M, self.N))
        rotated_coordinates_lower = np.empty((3, self.M, self.N))

        # For each strip, we rotate the grid points around the quarter chord point of the strip
        for i in range(self.N):
            # Rotate by the twist angle in the xz plane
            R = np.array(
                [
                    [
                        np.cos(self.twist_angles[i]),
                        0,
                        np.sin(self.twist_angles[i]),
                    ],
                    [0, 1, 0],
                    [
                        -np.sin(self.twist_angles[i]),
                        0,
                        np.cos(self.twist_angles[i]),
                    ],
                ],
            )
            rotated_coordinates[:, :, i] = (
                np.matmul(R, (coordinates[:, :, i] - c_4[:, i][:, None]))
                + c_4[:, i][:, None]
            )
            rotated_coordinates_upper[:, :, i] = (
                np.matmul(R, (coordinates_upper[:, :, i] - c_4[:, i][:, None]))
                + c_4[:, i][:, None]
            )
            rotated_coordinates_lower[:, :, i] = (
                np.matmul(R, (coordinates_lower[:, :, i] - c_4[:, i][:, None]))
                + c_4[:, i][:, None]
            )

        # Add origin
        coordinates = rotated_coordinates + self._origin[:, None, None]
        coordinates_upper = rotated_coordinates_upper + self._origin[:, None, None]
        coordinates_lower = rotated_coordinates_lower + self._origin[:, None, None]

        # The arrays are now in the form (3, M, N), we need to transpose them to (N, M, 3)
        self.grid = coordinates.transpose(2, 1, 0).reshape(-1, 3)
        self.grid_upper = coordinates_upper.transpose(2, 1, 0).reshape(-1, 3)
        self.grid_lower = coordinates_lower.transpose(2, 1, 0).reshape(-1, 3)

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

    def grid_to_panels(
        self, grid: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        num_panels = (self.N - 1) * (self.M - 1)
        grid = grid.reshape(self.N, self.M, 3)

        panels = np.empty((num_panels, 4, 3), dtype=float)
        control_points = np.empty((num_panels, 3), dtype=float)
        control_nj = np.empty((num_panels, 3), dtype=float)

        N = self.N
        M = self.M
        panels[:, 0, :] = grid[1:, : M - 1].reshape(-1, 3)
        panels[:, 1, :] = grid[: N - 1, : M - 1].reshape(-1, 3)
        panels[:, 2, :] = grid[: N - 1, 1:M].reshape(-1, 3)
        panels[:, 3, :] = grid[1:, 1:M].reshape(-1, 3)

        control_points[:, 0] = (
            (grid[: N - 1, : M - 1, 0] + grid[1:N, : M - 1, 0]) / 2
            + 3
            / 4
            * (
                (grid[: N - 1, 1:M, 0] + grid[1:N, 1:M, 0]) / 2
                - (grid[: N - 1, : M - 1, 0] + grid[1:N, : M - 1, 0]) / 2
            )
        ).reshape(-1)

        control_points[:, 1] = (
            (grid[: N - 1, : M - 1, 1] + grid[1:N, : M - 1, 1]) / 2
            + 1
            / 2
            * (
                (grid[: N - 1, 1:M, 1] + grid[1:N, 1:M, 1]) / 2
                - (grid[: N - 1, : M - 1, 1] + grid[1:N, : M - 1, 1]) / 2
            )
        ).reshape(-1)

        control_points[:, 2] = (
            (grid[: N - 1, : M - 1, 2] + grid[1:N, : M - 1, 2]) / 2
            + 1
            / 2
            * (
                (grid[: N - 1, 1:M, 2] + grid[1:N, 1:M, 2]) / 2
                - (grid[: N - 1, : M - 1, 2] + grid[1:N, : M - 1, 2]) / 2
            )
        ).reshape(-1)

        Ak = panels[:, 0, :] - panels[:, 2, :]
        Bk = panels[:, 1, :] - panels[:, 3, :]
        cross_prod = np.cross(Ak, Bk)
        control_nj = cross_prod / np.linalg.norm(cross_prod, axis=1)[:, None]

        return panels, control_points, control_nj

    def calculate_mean_chords(self) -> None:
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

    def calculate_area(self) -> None:
        """Finds the area of the wing."""

        rm1 = np.linalg.inv(self.R_MAT)

        grid_upper = self.grid_upper.reshape(self.N, self.M, 3)
        # Vectorized calculation for the upper surface
        _, y1_upper, _ = np.matmul(rm1, grid_upper[1:, 0, :].T)
        _, y2_upper, _ = np.matmul(rm1, grid_upper[:-1, 0, :].T)
        chord_dist_upper = self._chord_dist[:-1] + self._chord_dist[1:]
        # self.S = 2 * np.sum((y1_upper - y2_upper) * chord_dist_upper / 2)

        dspan = self._span_dist[1:] - self._span_dist[:-1]
        ave_chord = (self._chord_dist[1:] + self._chord_dist[:-1])/2
        self.S = float(np.sum(dspan * ave_chord))
        if self.is_symmetric_y:
            self.S = self.S * 2

        # Normalize by the maximum x value of the root airfoil
        # self.S = self.S / float(np.max(self.root_airfoil._x_lower))

        g_up = self.grid_upper.reshape(self.N, self.M, 3)
        g_low = self.grid_lower.reshape(self.N, self.M, 3)

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

    def calculate_volume(self) -> None:
        """Finds the volume of the wing using vectorized operations."""
        g_up = self.grid_upper.reshape(self.N, self.M, 3)
        g_low = self.grid_lower.reshape(self.N, self.M, 3)

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

        self.volume_distribution[:] = (0.5 * (area_front + area_back) * dx).flatten()

        self.volume = float(np.sum(self.volume_distribution))
        if self.is_symmetric_y:
            self.volume = self.volume * 2

    def calculate_center_mass(self) -> FloatArray:
        """Finds the center of mass of the wing using vectorized operations."""
        g_up = self.grid_upper.reshape(self.N, self.M, 3)
        g_low = self.grid_lower.reshape(self.N, self.M, 3)

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
            x_cm = np.sum(
                self.volume_distribution.reshape(self.N - 1, self.M - 1) * 2 * x
            )
            y_cm = 0
            z_cm = np.sum(
                self.volume_distribution.reshape(self.N - 1, self.M - 1) * 2 * z
            )
        else:
            x_cm = np.sum(self.volume_distribution.reshape(self.N - 1, self.M - 1) * x)
            y_cm = np.sum(self.volume_distribution.reshape(self.N - 1, self.M - 1) * y)
            z_cm = np.sum(self.volume_distribution.reshape(self.N - 1, self.M - 1) * z)

        return np.array((x_cm, y_cm, z_cm)) / self.volume

    def calculate_inertia(self, mass: float, cog: FloatArray) -> FloatArray:
        """
        Calculates the inertia of the wing about the center of gravity.

        Args:
            mass (float): Mass of the wing. Used to have dimensional inertia
            cog (FloatArray): Center of Gravity of the wing.
        """
        grid_upper = self.grid_upper.reshape(self.N, self.M, 3)
        grid_lower = self.grid_lower.reshape(self.N, self.M, 3)

        x_upp = (grid_upper[:-1, :-1, 0] + grid_upper[:-1, 1:, 0]) / 2
        x_low = (grid_lower[:-1, :-1, 0] + grid_lower[:-1, 1:, 0]) / 2

        y_upp = (grid_upper[1:, :-1, 1] + grid_upper[:-1, :-1, 1]) / 2
        y_low = (grid_lower[1:, :-1, 1] + grid_lower[:-1, :-1, 1]) / 2

        z_upp = (grid_upper[1:, :-1, 2] + grid_upper[1:, :-1, 2]) / 2
        z_low = (grid_lower[1:, :-1, 2] + grid_lower[1:, :-1, 2]) / 2

        xd = ((x_upp + x_low) / 2 - cog[0]) ** 2
        zd = ((z_upp + z_low) / 2 - cog[2]) ** 2

        if self.is_symmetric_y:
            yd = (-(y_upp + y_low) / 2 - cog[1]) ** 2 + (
                (y_upp + y_low) / 2 - cog[1]
            ) ** 2
        else:
            yd = ((y_upp + y_low) / 2 - cog[1]) ** 2

        I_xx = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (yd + zd)
        )
        I_yy = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (xd + zd)
        )
        I_zz = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (xd + yd)
        )

        xd = (x_upp + x_low) / 2 - cog[0]
        zd = (z_upp + z_low) / 2 - cog[2]

        if self.is_symmetric_y:
            yd = 0
        else:
            yd = (y_upp + y_low) / 2 - cog[1]

        I_xz = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (xd * zd)
        )
        I_xy = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (xd * yd)
        )
        I_yz = np.sum(
            self.volume_distribution.reshape(self.N - 1, self.M - 1) * (yd * zd)
        )

        return np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)) * (mass / self.volume)

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
            grid = grid.reshape(self.N, self.M, 3)
            gsym = grid[::-1, :, :] * reflection
            grid = grid[1:, :, :]
            grid = np.concatenate((gsym, grid))
            pass
        else:
            grid = grid
        return grid

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

        for i in np.arange(0, self.num_panels):
            if thin:
                items = [self.panels]
            else:
                items = [self.panels_lower, self.panels_upper]
            for item in items:
                p1, p3, p4, p2 = item[i, :, :]
                xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2)) + movement[0]

                ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2)) + movement[1]

                zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2)) + movement[2]

                ax.plot_wireframe(xs, ys, zs, linewidth=0.5)

                if self.is_symmetric_y:
                    ax.plot_wireframe(xs, -ys, zs, linewidth=0.5)
        if show_plot:
            plt.show()

    # def plot_surface(
    #     self,
    #     thin: bool = False,
    #     prev_fig: Figure | None = None,
    #     prev_ax: Axes3D | None = None,
    #     prev_movement: FloatArray | None = None,
    # ) -> None:
    #     """
    #     Plots the wing
    #     """

    #     fig: Figure = plt.figure()
    #     ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    #     ax.set_title("Wing")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.axis("equal")
    #     ax.view_init(30, 150)

    #     # Plot the wing grid
    #     for grid, c in zip([self.grid_lower, self.grid_upper], ["b", "r"]):
    #         X = grid[:, 0]
    #         Y = grid[:, 1]
    #         Z = grid[:, 2]

    #         ax.plot_trisurf(X, Y, Z, color=c)
    #     fig.show()

    def __str__(self) -> str:
        return f"Lifting Surface: {self.name} with {self.N} Panels and {self.M} Panels"
