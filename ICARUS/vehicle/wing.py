"""==================
Wing Class
==================
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray

from . import SymmetryAxes
from . import WingSurface
from .base_classes import Mass
from .base_classes import RigidBody

if TYPE_CHECKING:
    from .strip import Strip


class Wing(RigidBody):
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
        # Create a grid of points to store all Wing Segments
        self.wing_segments: Sequence[WingSurface] = wing_segments
        self.is_lifting = any([segment.is_lifting for segment in wing_segments])

        # Define the Coordinate System of the Merged Wing
        # For the orientation of the wing we default to 0,0,0
        orientation = np.array([0, 0, 0], dtype=float)
        # For the origin of the wing we take the origin of the first wing segment
        origin = wing_segments[0].origin

        super().__init__(
            name=name,
            origin=origin,
            orientation=orientation,
        )

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

        # Define the mass of the wing
        _chord_dist: list[FloatArray] = []
        _span_dist: list[FloatArray] = []
        _xoffset_dist: list[FloatArray] = []
        _zoffset_dist: list[FloatArray] = []

        for segment in wing_segments:
            seg_span_dist = segment._span_dist + float(segment.origin[1])
            _chord_dist.extend(segment._chord_dist.tolist())
            _span_dist.extend(seg_span_dist.tolist())
            _xoffset_dist.extend(segment._xoffset_dist.tolist())
            _zoffset_dist.extend(segment._zoffset_dist.tolist())

        self._chord_dist = np.array(_chord_dist, dtype=float)
        self._span_dist = np.array(_span_dist, dtype=float)
        self._xoffset_dist = np.array(_xoffset_dist, dtype=float)
        self._zoffset_dist = np.array(_zoffset_dist, dtype=float)

        # Grid Variables
        num_grid_points = self.num_grid_points
        num_panels = self.num_panels

        self.grid: FloatArray = np.empty((num_grid_points, 3), dtype=float)  # Camber Line
        self.grid_lower: FloatArray = np.empty((num_grid_points, 3), dtype=float)
        self.grid_upper: FloatArray = np.empty((num_grid_points, 3), dtype=float)

        self.panels: FloatArray = np.empty((num_panels, 4, 3), dtype=float)
        self.panels_lower: FloatArray = np.empty((num_panels, 4, 3), dtype=float)
        self.panels_upper: FloatArray = np.empty((num_panels, 4, 3), dtype=float)

        self.control_vars = set()
        self.controls = []
        for segment in self.wing_segments:
            self.control_vars.update(segment.control_vars)
            self.controls.extend(segment.controls)
        self.control_vector = {control_var: 0.0 for control_var in self.control_vars}

        ####### Calculate Wing Parameters #######
        self.define()
        ####### Calculate Wing Parameters ########

    ########## Rigid Body Properties ##########
    def _on_orientation_changed(self, old_orientation: FloatArray, new_orientation: FloatArray) -> None:
        for segment in self.wing_segments:
            segment.orientation_degrees = segment.orientation_degrees + (new_orientation - old_orientation)

    def _on_origin_changed(self, movement: FloatArray) -> None:
        """Updates the origin of the wing segments when the origin of the wing changes"""
        for segment in self.wing_segments:
            segment.origin += movement

    ########## END Rigid Body Properties ##########

    def define(self) -> None:
        """Calculate Wing Parameters"""
        for segment in self.wing_segments:
            segment.define()

        # Create Grid
        self.define_grid()

        for segment in self.wing_segments:
            segment_mass = Mass(
                name=segment.name,
                mass=segment.mass,
                position=segment.origin,
                inertia=segment.inertia,
            )
            self.remove_mass_point(segment_mass.name)
            self.add_mass_point(segment_mass)

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

    def get_separate_segments(self) -> list[WingSurface]:
        """Returns the separate segments of the wing"""
        segments: list[WingSurface] = []
        for segment in self.wing_segments:
            if isinstance(segment, Wing):
                segments.extend(segment.get_separate_segments())
            else:
                segments.append(segment)
        return segments

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
    def num_grid_points(self) -> int:
        num_grid_points = 0
        for segment in self.wing_segments:
            num_grid_points += segment.num_grid_points
        return num_grid_points

    @property
    def num_panels(self) -> int:
        """Returns the number of panels in the wing"""
        num_panels = 0
        for segment in self.wing_segments:
            num_panels += segment.num_panels
        return num_panels

    @property
    def airfoils(self) -> list[Airfoil]:
        # Define the airfoils
        airfoils: list[Airfoil] = []
        for segment in self.wing_segments:
            for airfoil in segment.airfoils:
                if airfoil not in airfoils:
                    airfoils.append(airfoil)
        return airfoils

    @property
    def strips(self) -> list[Strip]:
        strips = []
        for segment in self.wing_segments:
            strips.extend(segment.strips)
        return strips

    @property
    def all_strips(self) -> list[Strip]:
        strips = self.strips
        if self.is_symmetric_y:
            symmetric_strips = [strip.return_symmetric() for strip in strips]
            return [*symmetric_strips[::-1], *strips]
        return strips

    @property
    def area(self) -> float:
        """Returns the area of the wing"""
        area = 0.0
        for segment in self.wing_segments:
            area += segment.area
        return area

    @property
    def S(self) -> float:
        """Returns the area of the wing"""
        S = 0.0
        for segment in self.wing_segments:
            S += segment.S
        return S

    @property
    def mean_aerodynamic_chord(self) -> float:
        """Returns the mean aerodynamic chord of the wing"""
        mac = 0.0
        for segment in self.wing_segments:
            mac += segment.mean_aerodynamic_chord * segment.area
        return mac / self.area

    @property
    def standard_mean_chord(self) -> float:
        """Returns the standard mean chord of the wing"""
        smac = 0.0
        for segment in self.wing_segments:
            smac += segment.standard_mean_chord * segment.area
        return smac / self.area

    @property
    def volume(self) -> float:
        """Calculates the volume of the wing"""
        volume = 0.0
        for segment in self.wing_segments:
            volume += segment.structural_volume
        return volume

    def calculate_inertia(self, mass: float, cog: FloatArray) -> FloatArray:
        """Calculates the inertia of the wing"""
        # Divide the mass of the wing among the segments based on their area
        # This is done to calculate the inertia of each segment

        inertia = np.zeros(6)
        for segment in self.wing_segments:
            mass_segment = (segment.area / self.area) * mass
            inertia += segment.calculate_geometric_center_inertia(mass_segment, cog)
        return inertia

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
        span = 0.0
        for segment in self.wing_segments:
            span += segment.span
        return span

    @property
    def aspect_ratio(self) -> float:
        if self.is_symmetric_y:
            return (self.span**2) / self.S
        return (self.span**2) / self.S

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

    def split_xz_symmetric_wing(self) -> Wing:
        """Splits the wing into two symmetric wings"""
        wing_segments: list[WingSurface] = []
        for segment in self.wing_segments:
            split_wing = segment.split_xz_symmetric_wing()
            wing_segments.extend(split_wing.get_separate_segments())

        split_wing = Wing(
            name=self.name,
            wing_segments=wing_segments,
        )
        return split_wing

    def plot(
        self,
        thin: bool = False,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        prev_movement: FloatArray | None = None,
    ) -> None:
        """Plots the wing"""

        if isinstance(prev_fig, Figure) and isinstance(prev_ax, Axes3D):
            fig: Figure = prev_fig
            ax: Axes3D = prev_ax
            show_plot = False
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

        for segment in self.wing_segments:
            segment.plot(
                thin=thin,
                prev_fig=fig,
                prev_ax=ax,
                prev_movement=prev_movement,
            )

        if show_plot:
            plt.show()

    def __control__(self, control_vector: dict[str, float]) -> None:
        control_dict = {k: control_vector[k] for k in self.control_vars}
        for i in range(len(self.wing_segments)):
            surf_control_vec = {
                key: val
                for key, val in control_dict.items()
                if key in self.wing_segments[i].control_vars
            }
            self.wing_segments[i].__control__(surf_control_vec)
        self.define()

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
