"""
==================
Merged Wing Class
==================
"""

import numpy as np

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.strip import Strip
from ICARUS.Vehicle.utils import SymmetryAxes


class MergedWing(Lifting_Surface):
    "Class to represent a Wing of an airplane"

    def __init__(
        self,
        name: str,
        wing_segments: list[Lifting_Surface],
        symmetries: list[SymmetryAxes] | SymmetryAxes = SymmetryAxes.NONE,
    ) -> None:
        """
        Initializes the Wing Object

        Args:
            name (str): Name of the wing e.g. Main Wing
            wing_segments (list[Lifting_Surface]): List of Wing_Segments
        """
        self.name: str = name
        # Create a grid of points to store all Wing Segments
        self.wing_segments: list[Lifting_Surface] = wing_segments

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
        merged_wing_symmetries: list[SymmetryAxes] = []
        if isinstance(symmetries, SymmetryAxes):
            if symmetries == SymmetryAxes.NONE:
                # Check if all wing segments share same symmetries
                for symmetry_type in SymmetryAxes.__members__.values():
                    if all(
                        [
                            symmetry_type in segment.symmetries
                            for segment in wing_segments
                        ]
                    ):
                        merged_wing_symmetries.append(symmetry_type)
            else:
                merged_wing_symmetries = [symmetries]
        self.symmetries: list[SymmetryAxes] = merged_wing_symmetries
        self.is_symmetric_y: bool = True if SymmetryAxes.Y in self.symmetries else False

        # Define Discretization
        self.N = np.sum([segment.N for segment in wing_segments])
        self.M = np.sum([segment.M for segment in wing_segments])

        # Define Choord Lengths
        chords: list[float] = []
        for segment in self.wing_segments:
            chords = [*chords, *segment.chords]
        self.chords: list[float] = chords
        self._root_chord = wing_segments[0]._root_chord
        self._tip_chord = wing_segments[-1]._tip_chord

        # Get the twist distributions
        self.twist_angles = np.array(
            [segment.twist_angles for segment in wing_segments]
        )

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
        self._chord_dist = []
        self._span_dist = []
        self._xoffset_dist = []
        self._zoffset_dist = []

        for segment in wing_segments:
            self._chord_dist.extend(segment._chord_dist.tolist())
            self._span_dist.extend(segment._span_dist.tolist())
            self._xoffset_dist.extend(segment._xoffset_dist.tolist())
            self._zoffset_dist.extend(segment._zoffset_dist.tolist())

        self._chord_dist = np.array(self._chord_dist)
        self._span_dist = np.array(self._span_dist)
        self._xoffset_dist = np.array(self._xoffset_dist)
        self._zoffset_dist = np.array(self._zoffset_dist)

        # Variable Initialization
        NM = 0
        for segment in self.wing_segments:
            NM += (segment.N) * (segment.M)
        self.num_grid_points: int = NM
        # Grid Variables
        self.grid: FloatArray = np.empty((NM, 3))  # Camber Line
        self.grid_lower: FloatArray = np.empty((NM, 3))
        self.grid_upper: FloatArray = np.empty((NM, 3))

        # Panel Variables
        NM = 0
        for segment in self.wing_segments:
            NM += (segment.N - 1) * (segment.M - 1)
        self.num_panels: int = NM

        self.panels: FloatArray = np.empty((NM, 4, 3))
        self.panels_lower: FloatArray = np.empty((NM, 4, 3))
        self.panels_upper: FloatArray = np.empty((NM, 4, 3))

        # Initialize Strips
        self.strips: list[Strip] = []
        self.all_strips: list[Strip] = []

        # Initialize mean chords
        self.mean_aerodynamic_chord: float = 0.0
        self.standard_mean_chord: float = 0.0

        # Initialize Areas
        self.area: float = 0.0
        self.S: float = 0.0

        # Initialize Volumes
        self.volume_distribution: FloatArray = np.empty(self.num_panels)
        self.volume: float = 0.0

        ####### Calculate Wing Parameters #######
        self.calculate_wing_parameters()
        ####### Calculate Wing Parameters ########

    def create_grid(self) -> None:
        """
        Creates a grid of points to represent the wing
        """
        # Each wing segment has a grid of points that represent the wing segment
        # We need to combine all these points to create a grid for the entire wing
        # Combining the points will create some panels that are not present in the wing

        NM = 0
        for segment in self.wing_segments:
            segment.create_grid()
            NM += segment.N * segment.M

        grid: FloatArray = np.empty((NM, 3))
        grid_lower: FloatArray = np.empty((NM, 3))
        grid_upper: FloatArray = np.empty((NM, 3))

        NM = 0
        # Stack all the grid points of the wing segments
        for segment in self.wing_segments:
            grid[NM : NM + segment.M * segment.N, :] = np.reshape(
                segment.grid, (segment.M * segment.N, 3)
            )
            grid_lower[NM : NM + segment.M * segment.N, :] = np.reshape(
                segment.grid_upper,
                (segment.M * segment.N, 3),
            )
            grid_upper[NM : NM + segment.M * segment.N, :] = np.reshape(
                segment.grid_lower,
                (segment.M * segment.N, 3),
            )
            NM += segment.M * segment.N

        self.grid = grid
        self.grid_lower = grid_lower
        self.grid_upper = grid_upper

        NM = 0
        for segment in self.wing_segments:
            NM += (segment.N - 1) * (segment.M - 1)

        panels: FloatArray = np.empty((NM, 4, 3))
        control_points: FloatArray = np.empty((NM, 3))
        control_nj: FloatArray = np.empty((NM, 3))

        panels_lower: FloatArray = np.empty((NM, 4, 3))
        control_points_lower: FloatArray = np.empty((NM, 3))
        control_nj_lower: FloatArray = np.empty((NM, 3))

        panels_upper: FloatArray = np.empty((NM, 4, 3))
        control_points_upper: FloatArray = np.empty((NM, 3))
        control_nj_upper: FloatArray = np.empty((NM, 3))

        NM = 0
        for segment in self.wing_segments:
            inc = (segment.N - 1) * (segment.M - 1)
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

    def create_strips(self) -> None:
        """
        Creates the strips for the wing
        """
        self.strips = []
        self.all_strips = []
        for segment in self.wing_segments:
            segment.create_strips()
            self.strips.extend(segment.strips)
            self.all_strips.extend(segment.all_strips)

    def calculate_area(self) -> None:
        """
        Calculates the area of the wing
        """
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
        """
        Calculates the volume of the wing
        """
        volume = 0.0
        volume_distribution: list[FloatArray] = []

        for segment in self.wing_segments:
            segment.calculate_volume()
            volume += segment.volume
            volume_distribution.append(
                np.reshape(
                    segment.volume_distribution, ((segment.N - 1) * (segment.M - 1))
                )
            )

        self.volume = volume
        self.volume_distribution = np.array(volume_distribution).flatten()

    def calculate_inertia(self, mass: float, cog: FloatArray) -> FloatArray:
        """
        Calculates the inertia of the wing
        """
        # Divide the mass of the wing among the segments based on their area
        # This is done to calculate the inertia of each segment

        inertia = np.zeros((6))
        for segment in self.wing_segments:
            mass_segment = (segment.area / self.area) * mass
            inertia += segment.calculate_inertia(mass_segment, cog)
        return inertia

    def calculate_center_mass(self) -> FloatArray:
        """
        Calculates the center of mass of the wing
        """
        cog = np.zeros(3)
        x_cm = 0.0
        y_cm = 0.0
        z_cm = 0.0

        for segment in self.wing_segments:
            cog_segment = segment.calculate_center_mass()
            x_cm += cog_segment[0] * segment.M
            y_cm += cog_segment[1] * segment.M
            z_cm += cog_segment[2] * segment.M
        cog[0] = x_cm / self.M
        cog[1] = y_cm / self.M
        cog[2] = z_cm / self.M
        return cog
    
    def get_grid(self, which: str = "camber") -> FloatArray:
        """
        Returns the grid of the wing
        """
        grids = []
        for segment in self.wing_segments:
            if which == "camber":
                grids.append(segment.get_grid(which))
            elif which == "lower":
                grids.append(segment.get_grid(which))
            elif which == "upper":
                grids.append(segment.get_grid(which))
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
    
    @property
    def orientation(self) -> FloatArray:
        return self._orientation

    @orientation.setter
    def orientation(self, value: FloatArray) -> None:
        self._orientation = value
        self._pitch, self._roll, self._yaw = value

        for segment in self.wing_segments:
            segment.orientation = value + segment.orientation

    # def change_segment_discretization(self, N: int, M: int) -> None:
    #     """
    #     Changes the discretization of the wing segments
    #     """
    #     for segment in self.wing_segments:
    #         segment.change_discretization(N, M)
