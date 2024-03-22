from __future__ import annotations

import os
from typing import Any
from typing import TYPE_CHECKING

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import ndarray


if TYPE_CHECKING:
    from ICARUS.Core.types import FloatArray
    from ICARUS.Core.types import FloatOrListArray
    from ICARUS.Flight_Dynamics.disturbances import Disturbance
    from ICARUS.Flight_Dynamics.state import State
    from ICARUS.Vehicle.lifting_surface import Lifting_Surface
    from ICARUS.Vehicle.surface_connections import Surface_Connection

jsonpickle_pd.register_handlers()


class Airplane:
    def __init__(
        self,
        name: str,
        surfaces: list[Lifting_Surface],
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
        self.surfaces: list[Lifting_Surface] = surfaces

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
                self.main_wing: Lifting_Surface = surface
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
        self.point_masses: list[tuple[float, FloatArray, str]] = []
        self.moments: list[FloatArray] = []

        self.M: float = 0
        for surface in self.surfaces:
            mom = surface.inertia

            self.M += surface.mass
            self.moments.append(mom)

        # Define Computed States
        self.states: list[State] = []

        # Define Connection Dictionary
        self.connections: dict[str, Surface_Connection] = {}
        # self.register_connections()

    def get_position(self, name: str, axis: str) -> float | FloatArray:
        """
        Return the x position of the point mass

        Args:
            name (str): Name of the point mass
            axis (str): Axis to return the position of

        Raises:
            PlaneDoesntContainAttr: _description_

        Returns:
            float: Position of the point mass
        """
        for mass in self.point_masses:
            if mass[2] == name:
                if axis == "x":
                    return float(mass[1][0])
                elif axis == "y":
                    return float(mass[1][1])
                elif axis == "z":
                    return float(mass[1][2])
                elif axis == "xyz":
                    return mass[1]
                else:
                    raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {axis}")

        for surf in self.surfaces:
            if surf.name == name:
                if axis == "x":
                    return float(surf.origin[0])
                elif axis == "y":
                    return float(surf.origin[1])
                elif axis == "z":
                    return float(surf.origin[2])
                elif axis == "xyz":
                    return surf.origin
                else:
                    raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {axis}")

        raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {name}")

    def set_position(self, name: str, axis: str, value: float | FloatArray) -> None:
        """
        Set the position of a point mass

        Args:
            name (str): Name of the point mass
            axis (str): Axis to set the position of
            value (float): New value of the position
        """
        for i, mass in enumerate(self.point_masses):
            m, p, m_name = mass
            if m_name == name:
                # Get the old mass tuple and delete it

                if axis == "x":
                    if not isinstance(value, float):
                        raise ValueError("Value must be a float")
                    p[0] = value

                elif axis == "y":
                    if not isinstance(value, float):
                        raise ValueError("Value must be a float")
                    p[1] = value

                elif axis == "z":
                    if not isinstance(value, float):
                        raise ValueError("Value must be a float")
                    p[2] = value

                elif axis == "xyz":
                    if not isinstance(value, ndarray):
                        raise ValueError("Value must be a ndarray")
                    p = value
                    self.point_masses[i] = (m, p, name)
                else:
                    raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {axis}")
                return

        for surf in self.surfaces:
            if surf.name == name:
                if axis == "x":
                    if isinstance(value, float):
                        surf.x_origin = value
                    else:
                        raise ValueError("Value must be a float")
                elif axis == "y":
                    if isinstance(value, float):
                        surf.y_origin = value
                    else:
                        raise ValueError("Value must be a float")
                elif axis == "z":
                    if isinstance(value, float):
                        surf.z_origin = value
                    else:
                        raise ValueError("Value must be a float")
                elif axis == "xyz":
                    if isinstance(value, ndarray):
                        surf.origin = value
                    else:
                        raise ValueError("Value must be a ndarray")
                else:
                    raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {axis}")
                return
        raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {name}")

    def get_mass(self, name: str) -> float:
        """
        Get the mass of a point mass

        Args:
            name (str): Name of the point mass

        Raises:
            PlaneDoesntContainAttr: _description_

        Returns:
            float: Mass of the point mass
        """
        for m_mass, m_pos, m_name in self.point_masses:
            if m_name == name:
                return m_mass
        raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {name}")

    def change_mass(self, name: str, new_mass: float) -> None:
        """
        Change the mass of a point mass

        Args:
            name (str): Name of the point mass
            mass (float): New mass of the point mass
        """
        for i, mass in enumerate(self.point_masses):
            _, p, m_name = mass
            if m_name == name:
                self.point_masses[i] = (new_mass, p, name)
                return
        raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {name}")

    # Define the __getattribute__ function to get attributes from the surfaces
    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name.endswith("_position_x"):
                name = name.replace("_position_x", "")
                return self.get_position(name, "x")
            elif name.endswith("_position_y"):
                name = name.replace("_position_y", "")
                return self.get_position(name, "y")
            elif name.endswith("_position_z"):
                name = name.replace("_position_z", "")
                return self.get_position(name, "z")
            elif name.endswith("_position"):
                name = name.replace("_position", "")
                return self.get_position(name, "xyz")
            elif name.endswith("_mass"):
                name = name.replace("_mass", "")
                return self.get_mass(name)
            else:
                # How to handle infinite recursion?
                if "surfaces" in self.__dict__.keys():
                    for surface in self.surfaces:
                        if name.startswith(f"{surface.name}_"):
                            return surface.__getattribute__(name.replace(surface.name, ""))
                raise AttributeError(f"Plane doesn't contain attribute {name}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.endswith("_position_x"):
            name = name.replace("_position_x", "")
            self.set_position(name, "x", value)
        elif name.endswith("_position_y"):
            name = name.replace("_position_y", "")
            self.set_position(name, "y", value)
        elif name.endswith("_position_z"):
            name = name.replace("_position_z", "")
            self.set_position(name, "z", value)
        elif name.endswith("_position"):
            name = name.replace("_position", "")
            self.set_position(name, "xyz", value)
        elif name.endswith("_mass"):
            name = name.replace("_mass", "")
            self.change_mass(name, new_mass=value)
        else:
            if hasattr(self, "surfaces"):
                for surface in self.surfaces:
                    if name.startswith(f"{surface.name}_"):
                        surface.__setattr__(name.replace(f"{surface.name}_", ""), value)
                        return
            object.__setattr__(self, name, value)

    @property
    def directory(self) -> str:
        return f"{self.name}"

    @property
    def CG(self) -> FloatArray:
        return self.find_cg()

    @property
    def total_inertia(self) -> FloatArray:
        return self.find_inertia(self.CG)

    @property
    def masses(self) -> list[tuple[float, FloatArray, str]]:
        ret = []
        for surface in self.surfaces:
            mass: tuple[float, FloatArray, str] = (
                surface.mass,
                surface.CG,
                surface.name,
            )
            ret.append(mass)
        for point_mass in self.point_masses:
            ret.append(point_mass)

        return ret

    def get_seperate_surfaces(self) -> list[Lifting_Surface]:
        surfaces: list[Lifting_Surface] = []
        for surface in self.surfaces:
            if surface.is_symmetric_y:
                l, r = surface.split_xz_symmetric_wing()
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
        masses: list[tuple[float, FloatArray, str]],
    ) -> None:
        """
        Add point masses to the plane. The point masses are defined by a tuple of the mass and the position of the mass.

        Args:
            masses (tuple[float, NDArray[Shape[&#39;3,&#39;], Float]]): (mass, position) eg (3, np.array([0,0,0])
        """
        for mass in masses:
            self.point_masses.append(mass)
        # self.CG = self.find_cg()
        # self.total_inertia = self.find_inertia(self.CG)

    def find_cg(self) -> FloatArray:
        """
        Find the center of gravity of the plane

        Returns:
            Array : X,Y,Z coordinates of the center of gravity
        """
        x_cm = 0.
        y_cm = 0.
        z_cm = 0.
        self.M = 0.
        for m, r, desc in self.masses:
            self.M += m
            x_cm += m * r[0]
            y_cm += m * r[1]
            z_cm += m * r[2]
            print(m, r[0], r[1], r[2], desc)
            print(x_cm, y_cm, z_cm, self.M)
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

        for m, r_bod, _ in self.masses:
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
            for airfoil in surface.airfoils:
                if f"{airfoil.name}" not in airfoils:
                    airfoils.append(f"{airfoil.name}")
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
            ax = fig.add_subplot(projection="3d")  # type: ignore
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
        for m, r, desc in self.masses:
            ax.scatter(
                r[0] + mov[0],
                r[1] + mov[1],
                r[2] + mov[2],
                marker="o",
                s=int(m * 50.0),
                color="r",
            )
            # Add label to indicate point mass name
            # Text
            ax.text(r[0] + mov[0], r[1] + mov[1], r[2] + mov[2], '%s' % (desc), size=9, zorder=1, color='k')

        ax.scatter(
            self.CG[0] + mov[0],
            self.CG[1] + mov[1],
            self.CG[2] + mov[2],
            marker="o",
            s=50,
            color="b",
        )
        plt.show()

    def to_json(self) -> str:
        """
        Pickle the plane object to a json string

        Returns:
            str: Json String
        """
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def save(self) -> None:
        """
        Save the plane object to a json file
        """
        from ICARUS.Database import DB3D

        fname: str = os.path.join(DB3D, self.directory, f"{self.name}.json")
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


class PlaneDoesntContainAttr(AttributeError):
    pass
