from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.airfoils import Airfoil
from ICARUS.core.serialization import deserialize_function
from ICARUS.core.serialization import serialize_function
from ICARUS.core.types import FloatArray

from . import ControlSurface
from . import ControlType
from . import NoControl
from .base_classes.mass import Mass
from .base_classes.rigid_body import RigidBody
from .strip import Strip
from .utils import DiscretizationType
from .utils import SymmetryAxes
from .utils import equal_spacing_function_factory

if TYPE_CHECKING:
    from . import Wing


class WingSurface(RigidBody):
    """Class to represent a Lifting Surface with integrated mass properties."""

    def __init__(
        self,
        name: str,
        origin: FloatArray,
        orientation: FloatArray,
        root_airfoil: Airfoil,
        spanwise_positions: FloatArray,
        chord_lengths: FloatArray,
        z_offsets: FloatArray,
        x_offsets: FloatArray,
        twist_angles: FloatArray,
        N: int,
        M: int,
        structural_mass: float = 0.0,
        other_masses: list[Mass] | None = None,
        # Optional Parameters
        symmetries: list[SymmetryAxes] | SymmetryAxes = SymmetryAxes.NONE,
        chord_discretization_function: Callable[[int], float] | None = None,
        tip_airfoil: Airfoil | None = None,
        is_lifting: bool = True,
        controls: list[ControlSurface] = [NoControl],
    ) -> None:
        """
        Constructor for the Lifting Surface Class
        The lifting surface is defined by providing the information on a number of points on the wing.
        On must first give the origin of the wing, the orientation of the wing to define the coordinate system.
        Relative to the origin, we take a number of points on the wing. For each point we must know:
          - The spanwise position of the point
          - The chord_length of the wing at that point
          - The z-offset of the point
          - The x-offset of the point
          - The twist of the wing at that point
          - The dihedral of the wing at that point
          - The airfoil at that point. The airfoil is interpolated between the root and tip airfoil.
        """
        # Initialize the base RigidBody with mass distribution
        super().__init__(
            name=name,
            origin=origin,
            orientation=orientation,
            masses=other_masses,
        )
        self.structural_mass: float = structural_mass

        # Check that the number of points is the same for all parameters if not raise an error
        if not (
            len(spanwise_positions)
            == len(chord_lengths)
            == len(z_offsets)
            == len(x_offsets)
            == len(twist_angles)
        ):
            raise ValueError("The number of points must be the same for all parameters")

        self.is_lifting: bool = is_lifting

        # Controls
        self.controls = controls
        control_vars: set[str] = set()
        for control in self.controls:
            if control.name != "none":
                control_vars.add(control.control_var)
        self.control_vars: set[str] = control_vars
        self.num_control_variables = len(control_vars)
        self.control_vector = {control_var: 0.0 for control_var in control_vars}

        # Define Symmetries
        if isinstance(symmetries, SymmetryAxes):
            symmetries = [symmetries]
        self.symmetries: list[SymmetryAxes] = symmetries

        # Define Discretization
        # TODO: Add logic to handle different discretization types
        self.N: int = N
        self.M: int = M
        self.num_panels: int = (self.N - 1) * (self.M - 1)
        self.num_grid_points: int = self.N * self.M

        if chord_discretization_function is None:
            self.chord_spacing = DiscretizationType.LINEAR
            self.chord_discretization_function = equal_spacing_function_factory(
                M,
                stretching=1.0,
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

        # Airfoils
        assert isinstance(root_airfoil, Airfoil), "Root Airfoil must be an Airfoil"
        self._root_airfoil: Airfoil = root_airfoil

        if tip_airfoil is None:
            tip_airfoil = copy(root_airfoil)

        assert isinstance(tip_airfoil, Airfoil), "Tip Airfoil must be an Airfoil"
        self._tip_airfoil: Airfoil = tip_airfoil

        self.airfoils: list[Airfoil] = []

        # Store Span
        span: float = np.abs(spanwise_positions[-1] - spanwise_positions[0])
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
        self.grid: FloatArray = np.empty(
            (self.num_grid_points, 3),
            dtype=float,
        )  # Camber Line
        self.grid_upper: FloatArray = np.empty((self.num_grid_points, 3), dtype=float)
        self.grid_lower: FloatArray = np.empty((self.num_grid_points, 3), dtype=float)
        # Initialize Panel Variables
        self.panels: FloatArray = np.empty(
            (self.num_panels, 4, 3),
            dtype=float,
        )  # Camber Line
        self.panels_upper: FloatArray = np.empty((self.num_panels, 4, 3), dtype=float)
        self.panels_lower: FloatArray = np.empty((self.num_panels, 4, 3), dtype=float)

        # Initialize Strips
        self.strips: list[Strip] = []

        # Initialize Mean Chords
        self.mean_aerodynamic_chord: float = 0.0
        self.standard_mean_chord: float = 0.0

        # Initialize Areas
        self.S: float = 0.0
        self.area: float = 0.0

        # Initialize Volumes
        self.structural_volume_distribution: FloatArray = np.empty(
            self.num_panels,
            dtype=float,
        )
        self.structural_volume: float = 0.0

        ####### Calculate Wing Parameters #######
        self.define()
        ####### Calculate Wing Parameters ########

    @property
    def name(self) -> str:
        return self._name.replace(" ", "_")

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @classmethod
    def from_span_percentage_functions(
        cls,
        name: str,
        origin: FloatArray,
        orientation: FloatArray,
        root_airfoil: Airfoil,
        tip_airfoil: Airfoil,
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
        controls: list[ControlSurface] = [NoControl],
        is_lifting: bool = True,
    ) -> WingSurface:
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
        assert isinstance(root_airfoil, Airfoil), "Root Airfoil must be an Airfoil"
        assert isinstance(tip_airfoil, Airfoil), "Tip Airfoil must be an Airfoil"

        # Needed for when we have airfoils that are flapped and therefore have a different chord length
        def real_chord_fun(
            eta: float,
        ) -> float:
            # TODO: Add logic to handle interpolation between root and tip airfoil
            const = float(np.max(root_airfoil._x_lower))
            return const * chord_as_a_function_of_span_percentage(eta)

        if isinstance(symmetries, SymmetryAxes):
            symmetries = [symmetries]

        if SymmetryAxes.Y in symmetries:
            span = span / 2

        # Create the arrays that will be passed to the constructor
        for i in range(N):
            eta = span_discretization_function(i)
            spanwise_positions[i] = eta * span
            chord_lengths[i] = real_chord_fun(eta)
            z_offsets[i] = (
                np.tan(dihedral_as_a_function_of_span_percentage(eta)) * span * eta
            )
            x_offsets[i] = x_offset_as_a_function_of_span_percentage(eta)
            twist_angles[i] = twist_as_a_function_of_span_percentage(eta)

        self: WingSurface = WingSurface(
            name=name,
            origin=origin,
            orientation=orientation,
            root_airfoil=root_airfoil,
            tip_airfoil=tip_airfoil,
            spanwise_positions=spanwise_positions,
            chord_lengths=chord_lengths,
            z_offsets=z_offsets,
            x_offsets=x_offsets,
            twist_angles=twist_angles,
            N=N,
            M=M,
            chord_discretization_function=chord_discretization_function,
            structural_mass=mass,
            symmetries=symmetries,
            controls=controls,
            is_lifting=is_lifting,
        )
        return self

    def _on_origin_changed(self, movement: FloatArray) -> None:
        """Move the Wing by a given movement vector."""
        grid = self.grid.reshape(self.N, self.M, 3)
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
        self.define_strips()
        # self.calculate_wing_parameters()
        self.calculate_volume()

    @property
    def is_symmetric_y(self) -> bool:
        """Check if the wing is symmetric in Y direction"""
        return SymmetryAxes.Y in self.symmetries

    @property
    def is_symmetric_z(self) -> bool:
        """Check if the wing is symmetric in Z direction"""
        return SymmetryAxes.Z in self.symmetries

    @property
    def is_symmetric_x(self) -> bool:
        """Check if the wing is symmetric in X direction"""
        return SymmetryAxes.X in self.symmetries

    @property
    def span(self) -> float:
        return self._span

    @property
    def volume(self) -> float:
        """Return the volume of the wing."""
        return self.structural_volume

    @property
    def strip_pitches(self) -> FloatArray:
        return self.twist_angles + self.pitch_degrees

    def _on_orientation_changed(
        self,
        old_orientation: FloatArray,
        new_orientation: FloatArray,
    ) -> None:
        """Rotate the Wing by a given rotation matrix."""
        old_pitch, old_roll, old_yaw = old_orientation
        new_pitch, new_roll, new_yaw = new_orientation

        R_OLD = self._compute_rotation_matrix(
            pitch=float(old_pitch),
            roll=float(old_roll),
            yaw=float(old_yaw),
        )
        R_NEW = self._compute_rotation_matrix(
            pitch=float(new_pitch),
            roll=float(new_roll),
            yaw=float(new_yaw),
        )

        # Rotate Grid
        # grid has shape (N, M, 3)
        # R_MAT has shape (3, 3)
        inverseR_OLD = np.linalg.inv(R_OLD)

        # Step 1: Flatten the matrix of grid points
        flattened_points = self.grid.reshape(-1, 3)
        # Step 2: DeRotate the points by the old R_MAT
        rotated_points = np.dot(flattened_points, inverseR_OLD.T)
        # Step 3: Rotate the points by the new R_MAT
        rotated_points = np.dot(rotated_points, R_NEW.T)
        # Step 4: Reshape the array back to the original shape
        self.grid = rotated_points.reshape(self.grid.shape)

        # Do the same for the upper and lower surfaces
        flattened_points = self.grid_upper.reshape(-1, 3)
        rotated_points = np.dot(flattened_points, inverseR_OLD.T)
        rotated_points = np.dot(rotated_points, R_NEW.T)
        self.grid_upper = rotated_points.reshape(self.grid_upper.shape)

        flattened_points = self.grid_lower.reshape(-1, 3)
        rotated_points = np.dot(flattened_points, inverseR_OLD.T)
        rotated_points = np.dot(rotated_points, R_NEW.T)
        self.grid_lower = rotated_points.reshape(self.grid_lower.shape)

        # Rotate Panels
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(
            self.grid,
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

        self.define_strips()

    @property
    def sructural_mass_CG(self) -> FloatArray:
        if self.structural_mass == 0.0:
            return np.zeros(3, dtype=float)
        return self.calculate_geometric_center()

    @property
    def structural_mass_inertia(self) -> FloatArray:
        if self.structural_mass == 0.0:
            return np.zeros((3, 3), dtype=float)
        return self.calculate_geometric_center_inertia(
            self.structural_mass,
            self.sructural_mass_CG,
        )

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
        if self.is_symmetric_y:
            return (self.span**2) / self.S
        return (self.span**2) / self.S

    @property
    def root_airfoil(self) -> Airfoil:
        return self._root_airfoil

    @root_airfoil.setter
    def root_airfoil(self, value: str | Airfoil) -> None:
        if isinstance(value, str):
            from ICARUS.database import Database

            DB = Database.get_instance()
            value = DB.get_airfoil(value)
        self._root_airfoil = value
        self.define()

    @property
    def tip_airfoil(self) -> Airfoil:
        return self._tip_airfoil

    @tip_airfoil.setter
    def tip_airfoil(self, value: str | Airfoil) -> None:
        if isinstance(value, str):
            from ICARUS.database import Database

            DB = Database.get_instance()
            value = DB.get_airfoil(value)
        self._tip_airfoil = value
        self.define()

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

    def define(self) -> None:
        """Calculate Wing Parameters"""
        # Generate airfoils
        self.generate_airfoils()

        # Create Grid
        self.define_grid()

        # Create Surfaces
        self.define_strips()

        # Calculate Areas
        self.calculate_area()

        # Find Chords: mean_aerodynamic_chord , standard_mean_chord
        self.calculate_mean_chords()

        # Calculate Volumes
        self.calculate_volume()

        structural_mass = Mass(
            name=f"{self.name}_structural_mass",
            mass=self.structural_mass,
            position=self.sructural_mass_CG,
            inertia=self.structural_mass_inertia,
        )
        self.remove_mass_point(structural_mass.name)
        self.add_mass_point(structural_mass)

    def change_discretization(self, N: int | None = None, M: int | None = None) -> None:
        if N is not None:
            self.N = N
        if M is not None:
            self.M = M
        self.define()

    def split_xz_symmetric_wing(self) -> Wing:
        """Split Symmetric Wing into two Wings"""
        if self.is_symmetric_y:
            left = WingSurface(
                name=f"{self.name}_left",
                origin=np.array(
                    [
                        self.origin[0],
                        -self.origin[1],
                        self.origin[2],
                    ],
                    dtype=float,
                ),
                orientation=self.orientation_degrees,
                symmetries=[
                    symmetry
                    for symmetry in self.symmetries
                    if symmetry != SymmetryAxes.Y
                ],
                spanwise_positions=-self._span_dist[::-1],
                chord_lengths=self._chord_dist[::-1],
                z_offsets=self._zoffset_dist[::-1],
                x_offsets=self._xoffset_dist[::-1],
                twist_angles=self.twist_angles[::-1],
                root_airfoil=copy(self.tip_airfoil),
                tip_airfoil=copy(self.root_airfoil),
                N=self.N,
                M=self.M,
                structural_mass=self.mass / 2,
                controls=[control.return_symmetric() for control in self.controls],
            )

            right = WingSurface(
                name=f"{self.name}_right",
                root_airfoil=copy(self.root_airfoil),
                origin=self._origin,
                orientation=self.orientation_degrees,
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
                tip_airfoil=copy(self.tip_airfoil),
                N=self.N,
                M=self.M,
                structural_mass=self.mass / 2,
                controls=[copy(control) for control in self.controls],
            )

            from ICARUS.vehicle import Wing

            split_wing = Wing(
                name=self.name,
                wing_segments=[left, right],
            )
            return split_wing
        raise ValueError("Cannot Split Body it is not symmetric")

    def generate_airfoils(self) -> None:
        """Generate Airfoils for the Wing"""
        self.airfoils = []
        for j in range(self.N):
            span = self._span_dist[-1] - self._span_dist[0]
            eta = (self._span_dist[j] - self._span_dist[0]) / (span)

            local_span_position = self._span_dist[j] - self._span_dist[0]
            global_span_position = self._span_dist[j] + self.origin[1]

            if self.root_airfoil == self.tip_airfoil:
                airfoil_j = self.root_airfoil
            else:
                # Calculate the heta position of the strip
                airfoil_j = Airfoil.morph_new_from_two_foils(
                    self.root_airfoil,
                    self.tip_airfoil,
                    eta,
                    self.root_airfoil.n_points,
                )

            # Apply the control vector to the airfoil
            for control in self.controls:
                if control.type != ControlType.AIRFOIL:
                    continue

                if control.control_var not in self.control_vars:
                    continue
                control_val = self.control_vector[control.control_var]

                if control.coordinate_system == "local":
                    is_within = (
                        local_span_position >= control.span_position_start
                    ) and (local_span_position <= control.span_position_end)
                elif control.coordinate_system == "global":
                    is_within = (
                        global_span_position >= control.span_position_start
                    ) and (global_span_position <= control.span_position_end)
                else:
                    raise ValueError(
                        f"Unknown coordinate system {control.coordinate_system}",
                    )

                if is_within:
                    if control.constant_chord != 0:
                        flap_hinge = 1 - control.constant_chord / self._chord_dist[j]
                    else:
                        flap_hinge = control.chord_function(eta)

                    airfoil_j = airfoil_j.flap(
                        flap_hinge_chord_percentage=flap_hinge,
                        chord_extension=control.chord_extension,
                        flap_angle=control_val * control.gain,
                    )
            self.airfoils.append(airfoil_j)

    def define_strips(self) -> None:
        """Create Strips given the Grid and airfoil"""
        strips: list[Strip] = []
        i_range = np.arange(0, self.N)

        start_points = np.array(
            [
                self._xoffset_dist[i_range],
                self._span_dist[i_range],
                self._zoffset_dist[i_range],
            ],
            dtype=float,
        )
        start_points = np.matmul(self.R_MAT, start_points) + self._origin[:, None]

        for j in range(self.N):
            airfoil = self.airfoils[j]
            strip = Strip.from_leading_edge(
                leading_edge_x=start_points[0, j],
                leading_edge_y=start_points[1, j],
                leading_edge_z=start_points[2, j],
                pitch=self.pitch_degrees + self.twist_angles[j],
                roll=self.roll_degrees,
                yaw=self.yaw_degrees,
                chord=self._chord_dist[j],
                airfoil=airfoil,
            )
            strips.append(strip)
        self.strips = strips

    @property
    def all_strips(self) -> list[Strip]:
        if self.is_symmetric_y:
            symmetric_strips = [
                strip.return_symmetric(axis=SymmetryAxes.Y) for strip in self.strips
            ]
            return [*symmetric_strips[::-1], *self.strips]
        return self.strips

    def define_grid(self) -> None:
        chord_eta = np.array(
            [self.chord_discretization_function(int(i)) for i in range(self.M)],
        )
        chord_eta[-1] -= 1e-7
        chord_eta[0] += 1e-7

        xs: FloatArray = np.outer(chord_eta, self._chord_dist) + self._xoffset_dist
        xs_upper: FloatArray = xs.copy()
        xs_lower: FloatArray = xs.copy()

        ys = np.tile(self._span_dist, (self.M, 1))
        ys_upper = ys.copy()
        ys_lower = ys.copy()

        airf_z_up: list[FloatArray] = []
        airf_z_low: list[FloatArray] = []
        airf_camber: list[FloatArray] = []

        for j in range(self.N):
            airf_j: Airfoil = self.airfoils[j]

            z_0 = self._zoffset_dist[j]
            chord = self._chord_dist[j] * airf_j.norm_factor
            airf_z_up_i = airf_j.y_upper(chord_eta) * chord + z_0
            airf_z_low_i = airf_j.y_lower(chord_eta) * chord + z_0
            airf_camber_i = airf_j.camber_line(chord_eta) * chord + z_0

            airf_z_up.append(airf_z_up_i)
            airf_z_low.append(airf_z_low_i)
            airf_camber.append(airf_camber_i)

            # Normalize the xs according to the norm factor of the airfoil
            xs_upper[:, j] = (
                xs_upper[:, j] - self._xoffset_dist[j]
            ) * airf_j.norm_factor + self._xoffset_dist[j]
            xs_lower[:, j] = (
                xs_lower[:, j] - self._xoffset_dist[j]
            ) * airf_j.norm_factor + self._xoffset_dist[j]
            xs[:, j] = (
                xs[:, j] - self._xoffset_dist[j]
            ) * airf_j.norm_factor + self._xoffset_dist[j]

        zs_upper = np.array(airf_z_up).T
        zs_lower = np.array(airf_z_low).T
        zs_camber = np.array(airf_camber).T

        # Rotate according to R_MAT
        coordinates = np.matmul(
            self.R_MAT,
            np.vstack([xs.flatten(), ys.flatten(), zs_camber.flatten()]),
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
            self.grid,
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
        self,
        grid: FloatArray,
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
        """
        Finds the Mean Aerodynamic Chord (mean_aerodynamic_chord) of the wing.
        """
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
            * (self._span_dist[1:] - self._span_dist[:-1]),
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
        # chord_dist_upper = self._chord_dist[:-1] + self._chord_dist[1:]
        # self.S = 2 * np.sum((y1_upper - y2_upper) * chord_dist_upper / 2)

        dspan = self._span_dist[1:] - self._span_dist[:-1]
        ave_chord = (self._chord_dist[1:] + self._chord_dist[:-1]) / 2
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
            np.cross((AB1_up + AB2_up) / 2, (AD1_up + AD2_up) / 2),
            axis=-1,
        )

        AB1_low = g_low[1:, :-1, :] - g_low[:-1, :-1, :]
        AB2_low = g_low[1:, 1:, :] - g_low[:-1, 1:, :]
        AD1_low = g_low[:-1, 1:, :] - g_low[:-1, :-1, :]
        AD2_low = g_low[1:, 1:, :] - g_low[1:, :-1, :]
        area_low = np.linalg.norm(
            np.cross((AB1_low + AB2_low) / 2, (AD1_low + AD2_low) / 2),
            axis=-1,
        )

        self.area = float(np.sum(area_up) + np.sum(area_low))

    # @property
    # def volume(self) -> float:
    #     """Return the volume of the wing."""
    #     if not hasattr(self, "volume_distribution"):
    #         self.volume_distribution = np.zeros((self.N - 1) * (self.M - 1), dtype=float)
    #         self.calculate_volume()
    #     return self.volume

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

        self.structural_volume_distribution[:] = (
            0.5 * (area_front + area_back) * dx
        ).flatten()

        self.structural_volume = float(np.sum(self.structural_volume_distribution))
        if self.is_symmetric_y:
            self.structural_volume = self.structural_volume * 2

    def calculate_geometric_center(self) -> FloatArray:
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
                self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
                * 2
                * x,
            )
            y_cm = 0
            z_cm = np.sum(
                self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
                * 2
                * z,
            )
        else:
            x_cm = np.sum(
                self.structural_volume_distribution.reshape(self.N - 1, self.M - 1) * x,
            )
            y_cm = np.sum(
                self.structural_volume_distribution.reshape(self.N - 1, self.M - 1) * y,
            )
            z_cm = np.sum(
                self.structural_volume_distribution.reshape(self.N - 1, self.M - 1) * z,
            )

        return np.array((x_cm, y_cm, z_cm)) / self.structural_volume

    def calculate_geometric_center_inertia(
        self,
        mass: float,
        cog: FloatArray,
    ) -> FloatArray:
        """Calculates the inertia of the wing about the center of gravity.

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
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (yd + zd),
        )
        I_yy = np.sum(
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (xd + zd),
        )
        I_zz = np.sum(
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (xd + yd),
        )

        xd = (x_upp + x_low) / 2 - cog[0]
        zd = (z_upp + z_low) / 2 - cog[2]

        if self.is_symmetric_y:
            yd = 0
        else:
            yd = (y_upp + y_low) / 2 - cog[1]

        I_xz = np.sum(
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (xd * zd),
        )
        I_xy = np.sum(
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (xd * yd),
        )
        I_yz = np.sum(
            self.structural_volume_distribution.reshape(self.N - 1, self.M - 1)
            * (yd * zd),
        )

        return np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz)) * (
            mass / self.structural_volume
        )

    def get_grid(self, which: str = "camber") -> FloatArray | list[FloatArray]:
        """Returns the Grid of the Wing.

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

        grid = grid.reshape(self.N, self.M, 3)
        if self.is_symmetric_y is True:
            reflection = np.array([1, -1, 1])
            gsym = grid[::-1, :, :] * reflection
            grid = grid[1:, :, :]
            grid = np.concatenate((gsym, grid))
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
            ax = fig.add_subplot(projection="3d")  # noqa
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

    def __control__(self, control_vector: dict[str, float]) -> None:
        # Add the control vector to the wing
        old_control_vector = self.control_vector.copy()
        for key in control_vector:
            self.control_vector[key] = control_vector[key]

        if self.control_vector != old_control_vector:
            self.define()

    def __setstate__(self, state: dict[str, Any]) -> None:
        func_dict = state.get("chord_discretization_function")
        chord_discretization_function = (
            deserialize_function(func_dict) if func_dict else None
        )
        WingSurface.__init__(
            self,
            name=state["name"],
            origin=state["origin"],
            orientation=state["orientation"],
            spanwise_positions=state["spanwise_positions"],
            chord_lengths=state["chord_lengths"],
            x_offsets=state["x_offsets"],
            z_offsets=state["z_offsets"],
            twist_angles=state["twist_angles"],
            root_airfoil=state["root_airfoil"],
            tip_airfoil=state["tip_airfoil"],
            N=state["N"],
            M=state["M"],
            structural_mass=state["mass"],
            symmetries=state["symmetries"],
            chord_discretization_function=chord_discretization_function,
            is_lifting=state["is_lifting"],
            controls=state["controls"],
        )

    def __getstate__(self) -> dict[str, Any]:
        # Convert lambda function to a named function
        state = {
            "name": self.name,
            "origin": self.origin,
            "orientation": self.orientation_degrees,
            "spanwise_positions": np.array(self._span_dist, copy=True),
            "chord_lengths": np.array(self._chord_dist, copy=True),
            "x_offsets": np.array(self._xoffset_dist, copy=True),
            "z_offsets": np.array(self._zoffset_dist, copy=True),
            "twist_angles": np.array(self.twist_angles, copy=True),
            "root_airfoil": self.root_airfoil,
            "tip_airfoil": self.tip_airfoil,
            "N": self.N,
            "M": self.M,
            "mass": self.mass,
            "symmetries": self.symmetries,
            "is_lifting": self.is_lifting,
            "chord_discretization_function": (
                serialize_function(self.chord_discretization_function)
                if self.chord_spacing is not DiscretizationType.LINEAR
                else None
            ),
            "controls": self.controls,
        }
        return state

    def __repr__(self) -> str:
        """Returns a string representation of the wing"""
        return f"(Wing Surface) {self.name}: S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"

    def __str__(self) -> str:
        """Returns a string representation of the wing"""
        return f"(Wing Surface) {self.name}: S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"
