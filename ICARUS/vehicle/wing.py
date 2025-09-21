"""==================
Merged Wing Class
==================
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray

from . import SymmetryAxes
from . import WingSurface

if TYPE_CHECKING:
    from . import Strip


class Wing(WingSurface):
    """Class to represent a Wing of an airplane"""

    def __init__(
        self,
        name: str,
        wing_segments: Sequence[WingSurface],
    ) -> None:
        """Initializes the Wing Object

        Args:
            name (str): Name of the wing e.g. Main Wing
            wing_segments (list[Lifting_Surface]): List of Wing_Segments

        """
        self.name: str = name
        # Create a grid of points to store all Wing Segments
        self.wing_segments: Sequence[WingSurface] = wing_segments
        self.is_lifting = any([segment.is_lifting for segment in wing_segments])

        # Define the Coordinate System of the Merged Wing
        # For the orientation of the wing we default to 0,0,0
        orientation = np.array([0, 0, 0], dtype=float)
        # For the origin of the wing we take the origin of the first wing segment
        origin = wing_segments[0].origin

        # Define Orientation of the Wing
        pitch, roll, yaw = orientation
        self._pitch = pitch
        self._roll = roll
        self._yaw = yaw

        R_PITCH: FloatArray = np.array(
            [
                [np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)],
            ],
            dtype=float,
        )
        R_YAW: FloatArray = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        R_ROLL: FloatArray = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)],
            ],
            dtype=float,
        )
        self.R_MAT: FloatArray = R_YAW.dot(R_PITCH).dot(R_ROLL)

        # Define Symmetries
        self.symmetries: list[SymmetryAxes] = []
        for symmetry_type in SymmetryAxes.__members__.values():
            if all(
                [symmetry_type in segment.symmetries for segment in wing_segments],
            ):
                self.symmetries.append(symmetry_type)

        # Define Choord Lengths
        chords: list[float] = []
        for segment in self.wing_segments:
            chords = [*chords, *segment.chords]
        self.chords: list[float] = chords
        self._root_chord = wing_segments[0]._root_chord
        self._tip_chord = wing_segments[-1]._tip_chord

        # Get the twist distributions
        self.twist_angles = np.hstack(
            [segment.twist_angles for segment in wing_segments],
            dtype=float,
        ).flatten()

        # Define the airfoils
        airfoils: list[Airfoil] = []
        for segment in wing_segments:
            for airfoil in segment.airfoils:
                if airfoil not in airfoils:
                    airfoils.append(airfoil)
        self.airfoils: list[Airfoil] = airfoils

        # Store the Origin Parameters
        self._origin: FloatArray = origin
        self._x_origin: float = origin[0]
        self._y_origin: float = origin[1]
        self._z_origin: float = origin[2]

        self._orientation: FloatArray = orientation

        # Define the mass of the wing
        _chord_dist: list[FloatArray] = []
        _span_dist: list[FloatArray] = []
        _xoffset_dist: list[FloatArray] = []
        _zoffset_dist: list[FloatArray] = []

        for segment in wing_segments:
            seg_span_dist = segment._span_dist + segment.origin[1]
            _chord_dist.extend(segment._chord_dist.tolist())
            _span_dist.extend(seg_span_dist.tolist())
            _xoffset_dist.extend(segment._xoffset_dist.tolist())
            _zoffset_dist.extend(segment._zoffset_dist.tolist())

        self._chord_dist = np.array(_chord_dist, dtype=float)
        self._span_dist = np.array(_span_dist, dtype=float)
        self._xoffset_dist = np.array(_xoffset_dist, dtype=float)
        self._zoffset_dist = np.array(_zoffset_dist, dtype=float)

        # Define Discretization
        self.N = 0  # Should be NaN
        self.M = sum([segment.M for segment in wing_segments])

        self.num_grid_points = 0
        self.num_panels = 0
        for segment in self.wing_segments:
            self.num_grid_points += segment.num_grid_points
            self.num_panels += segment.num_panels

        # Grid Variables
        self.grid: FloatArray = np.empty(
            (self.num_grid_points, 3),
            dtype=float,
        )  # Camber Line
        self.grid_lower: FloatArray = np.empty((self.num_grid_points, 3), dtype=float)
        self.grid_upper: FloatArray = np.empty((self.num_grid_points, 3), dtype=float)

        self.panels: FloatArray = np.empty((self.num_panels, 4, 3), dtype=float)
        self.panels_lower: FloatArray = np.empty((self.num_panels, 4, 3), dtype=float)
        self.panels_upper: FloatArray = np.empty((self.num_panels, 4, 3), dtype=float)

        # Initialize Strips
        self.strips: list[Strip] = []

        # Initialize mean chords
        self.mean_aerodynamic_chord: float = 0.0
        self.standard_mean_chord: float = 0.0

        # Initialize Areas
        self.area: float = 0.0
        self.S: float = 0.0

        # Initialize Volumes
        self.volume_distribution: FloatArray = np.empty(self.num_panels, dtype=float)
        self.volume: float = 0.0

        self.control_vars = set()
        self.controls = []
        for segment in self.wing_segments:
            self.control_vars.update(segment.control_vars)
            self.controls.extend(segment.controls)
        self.control_vector = {control_var: 0.0 for control_var in self.control_vars}

        ####### Calculate Wing Parameters #######
        self.define_wing_parameters()
        ####### Calculate Wing Parameters ########

    def get_separate_segments(self) -> list[WingSurface]:
        """Returns the separate segments of the wing"""
        segments: list[WingSurface] = []
        for segment in self.wing_segments:
            if isinstance(segment, Wing):
                segments.extend(segment.get_separate_segments())
            else:
                segments.append(segment)
        return segments

    def generate_airfoils(self) -> None:
        # Define the airfoils
        airfoils: list[Airfoil] = []
        for segment in self.wing_segments:
            for airfoil in segment.airfoils:
                if airfoil not in airfoils:
                    airfoils.append(airfoil)
        self.airfoils = airfoils

    def define_grid(self) -> None:
        """Creates a grid of points to represent the wing"""
        # Each wing segment has a grid of points that represent the wing segment
        # We need to combine all these points to create a grid for the entire wing
        # Combining the points will create some panels that are not present in the wing

        NM = 0
        for segment in self.wing_segments:
            segment.define_grid()
            NM += segment.num_grid_points

        grid: FloatArray = np.empty((NM, 3), dtype=float)
        grid_lower: FloatArray = np.empty((NM, 3), dtype=float)
        grid_upper: FloatArray = np.empty((NM, 3), dtype=float)

        NM = 0
        # Stack all the grid points of the wing segments
        for segment in self.wing_segments:
            n_points = segment.num_grid_points
            grid[NM : NM + n_points, :] = np.reshape(
                segment.grid,
                (n_points, 3),
            )
            grid_lower[NM : NM + n_points, :] = np.reshape(
                segment.grid_upper,
                (n_points, 3),
            )
            grid_upper[NM : NM + n_points, :] = np.reshape(
                segment.grid_lower,
                (n_points, 3),
            )
            NM += n_points

        self.grid = grid
        self.grid_lower = grid_lower
        self.grid_upper = grid_upper

        NM = 0
        for segment in self.wing_segments:
            NM += segment.num_panels

        panels: FloatArray = np.empty((NM, 4, 3), dtype=float)
        control_points: FloatArray = np.empty((NM, 3), dtype=float)
        control_nj: FloatArray = np.empty((NM, 3), dtype=float)

        panels_lower: FloatArray = np.empty((NM, 4, 3), dtype=float)
        control_points_lower: FloatArray = np.empty((NM, 3), dtype=float)
        control_nj_lower: FloatArray = np.empty((NM, 3), dtype=float)

        panels_upper: FloatArray = np.empty((NM, 4, 3), dtype=float)
        control_points_upper: FloatArray = np.empty((NM, 3), dtype=float)
        control_nj_upper: FloatArray = np.empty((NM, 3), dtype=float)

        NM = 0
        for segment in self.wing_segments:
            inc = segment.num_panels
            panels[NM : NM + inc, :, :] = segment.panels
            control_points[NM : NM + inc, :] = segment.control_points
            control_nj[NM : NM + inc, :] = segment.control_nj

            panels_lower[NM : NM + inc, :, :] = segment.panels_lower
            control_points_lower[NM : NM + inc, :] = segment.control_points_lower
            control_nj_lower[NM : NM + inc, :] = segment.control_nj_lower

            panels_upper[NM : NM + inc, :, :] = segment.panels_upper
            control_points_upper[NM : NM + inc, :] = segment.control_points_upper
            control_nj_upper[NM : NM + inc, :] = segment.control_nj_upper

            NM += inc

        self.panels = panels
        self.control_points = control_points
        self.control_nj = control_nj

        self.panels_lower = panels_lower
        self.control_points_lower = control_points_lower
        self.control_nj_lower = control_nj_lower

        self.panels_upper = panels_upper
        self.control_points_upper = control_points_upper
        self.control_nj_upper = control_nj_upper

    def define_strips(self) -> None:
        """Creates the strips for the wing"""
        self.strips = []
        for segment in self.wing_segments:
            segment.define_strips()
            self.strips.extend(segment.strips)

    def calculate_area(self) -> None:
        """Calculates the area of the wing"""
        self.area = 0.0
        self.S = 0.0
        for segment in self.wing_segments:
            segment.calculate_area()
            self.area += segment.area
            self.S += segment.S

    def calculate_mean_chords(self) -> None:
        mac = 0.0
        smac = 0.0

        for segment in self.wing_segments:
            segment.calculate_mean_chords()
            mac += segment.mean_aerodynamic_chord * segment.area
            smac += segment.standard_mean_chord * segment.area

        self.mean_aerodynamic_chord = mac / self.area
        self.standard_mean_chord = smac / self.area

    def calculate_volume(self) -> None:
        """Calculates the volume of the wing"""
        volume = 0.0
        volume_distribution: list[FloatArray] = []

        for segment in self.wing_segments:
            segment.calculate_volume()
            volume += segment.volume
            volume_distribution.append(
                np.reshape(
                    segment.volume_distribution,
                    segment.num_panels,
                ),
            )

        self.volume = volume
        self.volume_distribution = np.hstack(volume_distribution, dtype=float).flatten()

    def calculate_inertia(self, mass: float, cog: FloatArray) -> FloatArray:
        """Calculates the inertia of the wing"""
        # Divide the mass of the wing among the segments based on their area
        # This is done to calculate the inertia of each segment

        inertia = np.zeros(6)
        for segment in self.wing_segments:
            mass_segment = (segment.area / self.area) * mass
            inertia += segment.calculate_inertia(mass_segment, cog)
        return inertia

    def calculate_center_mass(self) -> FloatArray:
        """Calculates the center of mass of the wing"""
        cog = np.zeros(3, dtype=float)
        x_cm = 0.0
        y_cm = 0.0
        z_cm = 0.0
        if self.mass == 0.0:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        for segment in self.wing_segments:
            cog_segment = segment.calculate_center_mass()
            x_cm += cog_segment[0] * segment.mass
            y_cm += cog_segment[1] * segment.mass
            z_cm += cog_segment[2] * segment.mass
        cog[0] = x_cm / self.mass
        cog[1] = y_cm / self.mass
        cog[2] = z_cm / self.mass
        return cog

    def get_grid(self, which: str = "camber") -> list[FloatArray]:
        """Returns the grid of the wing"""
        grids: list[FloatArray] = []
        for segment in self.wing_segments:
            grid = segment.get_grid(which)
            if isinstance(grid, list):
                grids.extend(grid)
            else:
                grids.append(grid)
        return grids

    @property
    def span(self) -> float:
        span = 0.0
        for segment in self.wing_segments:
            span += segment.span
        return span

    @property
    def mass(self) -> float:
        mass = 0.0
        for segment in self.wing_segments:
            mass += segment.mass
        return mass

    @mass.setter
    def mass(self, new_mass: float) -> None:
        old_mass = self.mass
        surface = self.area

        for segment in self.wing_segments:
            segment.mass = (
                (segment.area / surface) * segment.mass * (new_mass / old_mass)
            )

    @property
    def CG(self) -> FloatArray:
        return self.calculate_center_mass()

    @property
    def orientation(self) -> FloatArray:
        return self._orientation

    @orientation.setter
    def orientation(self, value: FloatArray) -> None:
        self._orientation = value
        self._pitch, self._roll, self._yaw = value

        for segment in self.wing_segments:
            segment.orientation = value + segment.orientation

    @property
    def root_airfoil(self) -> Airfoil:
        return self.wing_segments[0].root_airfoil

    @root_airfoil.setter
    def root_airfoil(self, value: Airfoil) -> None:
        self.wing_segments[0].root_airfoil = value

    @property
    def tip_airfoil(self) -> Airfoil:
        return self.wing_segments[-1].tip_airfoil

    @tip_airfoil.setter
    def tip_airfoil(self, value: Airfoil) -> None:
        self.wing_segments[-1].tip_airfoil = value

    # def change_segment_discretization(self, N: int, M: int) -> None:
    #     """
    #     Changes the discretization of the wing segments
    #     """
    #     for segment in self.wing_segments:
    #         segment.change_discretization(N, M)
    def split_xz_symmetric_wing(self) -> "Wing":
        """Splits the wing into two symmetric wings"""
        wing_segments = []
        for segment in self.wing_segments:
            split_wing = segment.split_xz_symmetric_wing()
            wing_segments.append(split_wing)

        split_wing = Wing(
            name=self.name,
            wing_segments=wing_segments,
        )
        return split_wing

    def __control__(self, control_vector: dict[str, float]) -> None:
        control_dict = {k: control_vector[k] for k in self.control_vars}
        for i in range(len(self.wing_segments)):
            surf_control_vec = {
                key: val
                for key, val in control_dict.items()
                if key in self.wing_segments[i].control_vars
            }
            self.wing_segments[i].__control__(surf_control_vec)
        self.define_wing_parameters()

    def __setstate__(self, state: dict[str, Any]) -> None:
        Wing.__init__(
            self,
            state["name"],
            state["wing_segments"],
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wing_segments": self.wing_segments,
        }

    def __repr__(self) -> str:
        """Returns a string representation of the wing"""
        return f"(Merged Wing): {self.name}: S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"

    def __str__(self) -> str:
        """Returns a string representation of the wing"""
        return f"(Merged Wing): {self.name}: S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"
