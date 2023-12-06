import os

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import ndarray

from ICARUS.Core.types import FloatArray
from ICARUS.Core.types import FloatOrListArray
from ICARUS.Flight_Dynamics.disturbances import Disturbance
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.surface_connections import Surface_Connection
from ICARUS.Vehicle.wing_segment import Wing_Segment

jsonpickle_pd.register_handlers()


class Airplane:
    def __init__(
        self,
        name: str,
        surfaces: list[Wing_Segment],
        disturbances: list[Disturbance] | None = None,
        orientation: FloatOrListArray | None = None,
    ) -> None:
        """
        Initialize the Airplane class

        Args:
            name (str): Name of the plane
            surfaces (list[Wing]): List of the lifting surfaces of the plane (Wings)
            disturbances (list[Disturbance] | None, optional): Optional List of disturbances. Defaults to None.
            orientation (FloatOrListArray] | None, optional): Plane Orientation. Defaults to None.
        """
        self.name: str = name
        self.CASEDIR: str = name
        self.surfaces: list[Wing_Segment] = surfaces

        if disturbances is None:
            self.disturbances: list[Disturbance] = []
        else:
            self.disturbances = disturbances

        if orientation is None:
            self.orientation: FloatOrListArray = [
                0.0,
                0.0,
                0.0,
            ]
        else:
            self.orientation = orientation

        found_wing: bool = False
        self.S: float = 0
        for surface in surfaces:
            if surface.name.capitalize().startswith("MAIN"):
                self.main_wing: Wing_Segment = surface
                self.S += surface.S
                self.mean_aerodynamic_chord: float = surface.mean_aerodynamic_chord
                self.aspect_ratio: float = surface.aspect_ratio
                self.span: float = surface.span
                found_wing = True

        if not found_wing:
            self.main_wing = surfaces[0]
            self.S = surfaces[0].S
            self.mean_aerodynamic_chord = surfaces[0].mean_aerodynamic_chord
            self.aspect_ratio = surfaces[0].aspect_ratio
            self.span = surfaces[0].span

        self.airfoils: list[str] = self.get_all_airfoils()
        self.masses: list[tuple[float, FloatArray]] = []
        self.moments: list[FloatArray] = []

        self.M: float = 0
        for surface in self.surfaces:
            mass: tuple[float, FloatArray] = (surface.mass, surface.CG)
            mom = surface.inertia

            self.M += surface.mass
            self.moments.append(mom)
            self.masses.append(mass)

        self.CG: FloatArray = self.find_cg()
        self.total_inertia: FloatArray = self.find_inertia(self.CG)

        # Define Computed States
        self.states: list[State] = []

        # Define Connection Dictionary
        self.connections: dict[str, Surface_Connection] = {}
        self.register_connections()

    def get_seperate_surfaces(self) -> list[Wing_Segment]:
        surfaces: list[Wing_Segment] = []
        for surface in self.surfaces:
            if surface.is_symmetric:
                l, r = surface.split_symmetric_wing()
                surfaces.append(l)
                surfaces.append(r)
            else:
                surfaces.append(surface)
        return surfaces

    def register_connections(self) -> None:
        """
        For each surface, detect if it is connected to another surface and register the connection
        There are 2 types of connections:
        1: Surfaces Are Connected Spanwise. So the tip emmisions of one surface are non-existent
        2: Surfaces Are Connected Chordwise. So the trailing edge emmisions of one surface are non-existent

        """
        # Detect if surfaces are connected spanwise
        # To do this, we check if the tip of one surface is the same as the root of another surface
        # If it is, then we register the connection
        return None
        for surface in self.surfaces:
            for other_surface in self.surfaces:
                if surface is not other_surface:
                    if np.allclose(surface.tip, other_surface.root):
                        if surface.name not in self.connections.keys():
                            self.connections[surface.name] = Surface_Connection()
                            self.connections[other_surface.name] = Surface_Connection()

        # Detect if surfaces are connected chordwise
        # To do this, we check if the trailing edge of one surface is the same as the leading edge of another surface
        # If it is, then we register the connection
        for surface in self.surfaces:
            for other_surface in self.surfaces:
                if surface is not other_surface:
                    if np.allclose(surface.trailing_edge, other_surface.leading_edge):
                        if surface.name not in self.connections.keys():
                            self.connections[surface.name] = Surface_Connection()
                            self.connections[other_surface.name] = Surface_Connection()

    def add_point_masses(
        self,
        masses: list[tuple[float, FloatArray]],
    ) -> None:
        """
        Add point masses to the plane. The point masses are defined by a tuple of the mass and the position of the mass.

        Args:
            masses (tuple[float, NDArray[Shape[&#39;3,&#39;], Float]]): (mass, position) eg (3, np.array([0,0,0])
        """
        for mass in masses:
            self.masses.append(mass)
        self.CG = self.find_cg()
        self.total_inertia = self.find_inertia(self.CG)

    def find_cg(self) -> FloatArray:
        """
        Find the center of gravity of the plane

        Returns:
            Array : X,Y,Z coordinates of the center of gravity
        """
        x_cm = 0
        y_cm = 0
        z_cm = 0
        self.M = 0
        for m, r in self.masses:
            self.M += m
            x_cm += m * r[0]
            y_cm += m * r[1]
            z_cm += m * r[2]
        return np.array((x_cm, y_cm, z_cm), dtype=float) / self.M

    def find_inertia(self, point: FloatArray) -> FloatArray:
        """
        Find the inertia of the plane about a point

        Returns:
            Array: The 6 components of the inertia matrix
                   Ixx, Iyy, Izz, Ixz, Ixy, Iyz
        """
        I_xx = 0
        I_yy = 0
        I_zz = 0
        I_xz = 0
        I_xy = 0
        I_yz = 0

        for inertia in self.moments:
            I_xx += inertia[0]
            I_yy += inertia[1]
            I_zz += inertia[2]
            I_xz += inertia[3]
            I_xy += inertia[4]
            I_yz += inertia[5]

        for m, r_bod in self.masses:
            r = point - r_bod
            I_xx += m * (r[1] ** 2 + r[2] ** 2)
            I_yy += m * (r[0] ** 2 + r[2] ** 2)
            I_zz += m * (r[0] ** 2 + r[1] ** 2)
            I_xz += m * (r[0] * r[2])
            I_xy += m * (r[0] * r[1])
            I_yz += m * (r[1] * r[2])

        return np.array((I_xx, I_yy, I_zz, I_xz, I_xy, I_yz))

    def get_all_airfoils(self) -> list[str]:
        """
        Get all the airfoils used in the plane

        Returns:
            list[str]: List of all the airfoils used in the plane
        """
        airfoils: list[str] = []
        for surface in self.surfaces:
            if f"NACA{surface.airfoil.name}" not in airfoils:
                airfoils.append(f"NACA{surface.airfoil.name}")
        return airfoils

    def visualize(
        self,
        prev_fig: Figure | None = None,
        prev_ax: Axes3D | None = None,
        movement: FloatArray | None = None,
        thin: bool = False,
    ) -> None:
        """
        Visualize the plane

        Args:
            prev_fig (Figure | None, optional): Previous Figure. When Called from another object . Defaults to None.
            prev_ax (Axes3D | None, optional): Previous Axes. Same as above . Defaults to None.
            movement (FloatArray | None, optional): Plane Movement from origin. Defaults to None.
        """
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
            ax.view_init(30, 150)
            ax.axis("scaled")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        if isinstance(movement, ndarray):
            mov = movement
        else:
            if movement is None:
                mov = np.zeros(3)
            else:
                try:
                    mov = np.array(movement)
                except ValueError:
                    print("Movement must be a 3 element array")
                    mov = np.zeros(3)

        for surface in self.surfaces:
            surface.plot(thin, fig, ax, mov)
        # Add plot for masses
        for m, r in self.masses:
            ax.scatter(
                r[0] + mov[0],
                r[1] + mov[1],
                r[2] + mov[2],
                marker="o",
                s=int(m * 50.0),
                color="r",
            )
        ax.scatter(
            self.CG[0] + mov[0],
            self.CG[1] + mov[1],
            self.CG[2] + mov[2],
            marker="o",
            s=50,
            color="b",
        )
        plt.show()

    def define_dynamic_pressure(self, u: float, dens: float) -> None:
        """
        Define the simulation parameters for the plane

        Args:
            u (float): Velocity Magnitude of the freestream
            dens (float): Density of the freestream
        """
        self.u_freestream: float = u
        self.dens: float = dens
        self.dynamic_pressure: float = 0.5 * dens * u**2

    def to_json(self) -> str:
        """
        Pickle the plane object to a json string

        Returns:
            str: Json String
        """
        encoded: str = jsonpickle.encode(self)
        return encoded

    def save(self) -> None:
        """
        Save the plane object to a json file
        """
        from ICARUS.Database import DB3D

        fname: str = os.path.join(DB3D, self.CASEDIR, f"{self.name}.json")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    # def __str__(self):
    #     str = f"Plane Object for {self.name}\n"
    #     str += f"Surfaces:\n"
    #     for i,surfaces in enumerate(self.surfaces):
    #         str += f"\t{surfaces.name} NB={i} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
    #     return str

    def __str__(self) -> str:
        string: str = f"Plane Object: {self.name}"
        return string
