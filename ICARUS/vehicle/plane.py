from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import ndarray

from ICARUS.vehicle.control_surface import NoControl
from ICARUS.vehicle.merged_wing import MergedWing
from ICARUS.vehicle.strip import Strip

if TYPE_CHECKING:
    from ICARUS.core.types import FloatArray
    from ICARUS.core.types import FloatOrListArray
    from ICARUS.flight_dynamics.state import State
    from ICARUS.vehicle.surface import WingSurface
    from ICARUS.vehicle.surface_connections import Surface_Connection

jsonpickle_pd.register_handlers()


class Airplane:
    def __init__(
        self,
        name: str,
        surfaces: list[WingSurface],
        orientation: FloatOrListArray | None = None,
        point_masses: list[tuple[float, FloatArray, str]] | None = None,
        cg_override: FloatArray | None = None,
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
        self.surfaces: list[WingSurface] = surfaces

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
                self.main_wing: WingSurface = surface
                self.S = surface.S
                self.mean_aerodynamic_chord: float = surface.mean_aerodynamic_chord
                self.aspect_ratio: float = surface.aspect_ratio
                self.span: float = surface.span
                found_wing = True
                break

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

        if point_masses is not None:
            self.add_point_masses(point_masses)

        # Define Computed States
        self.states: list[State] = []

        # Define Connection Dictionary
        self.connections: dict[str, Surface_Connection] = {}
        # self.register_connections()

        # Get the control vector:
        control_vars: set[str] = set()
        for surf in self.surfaces:
            control_vars.update(surf.control_vars)
        self.control_vars: set[str] = control_vars
        self.num_control_variables = len(control_vars)
        self.control_vector: dict[str, float] = {k: 0.0 for k in control_vars}

        self.cg_override: FloatArray | None = cg_override
        if cg_override is not None:
            self.overide_mass = True
            self._CG = cg_override
        else:
            self.overide_mass = False
            self._CG = self.find_cg()

    def __control__(self, control_vector: dict[str, float]) -> None:
        control_dict = {k: control_vector[k] for k in self.control_vars}
        for surf in self.surfaces:
            surf_control_vec = {}
            for name, value in control_dict.items():
                if name in surf.control_vars:
                    surf_control_vec[name] = value
                self.control_vector[name] = value
            surf.__control__(surf_control_vec)
            print(f"Controling {surf.name} with {surf_control_vec}")

    # @property
    # def surfaces(self) -> list[WingSurface]:
    #     surfaces: list[WingSurface] = []
    #     for surface in self._surfaces:
    #         if isinstance(surface, MergedWing):
    #             for s in surface.wing_segments:
    #                 surfaces.append(s)
    #         else:
    #             surfaces.append(surface)
    #     return surfaces

    # @property
    # def strips(self) -> list[Strip]:
    #     strips: list[Strip] = []
    #     for surface in self.surfaces:
    #         for strip in surface.strips:
    #             strips.append(strip)
    #     return strips

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
    # def __getattribute__(self, name: str) -> Any:
    #     try:
    #         return object.__getattribute__(self, name)
    #     except AttributeError:
    #         if name.endswith("_position_x"):
    #             name = name.replace("_position_x", "")
    #             return self.get_position(name, "x")
    #         elif name.endswith("_position_y"):
    #             name = name.replace("_position_y", "")
    #             return self.get_position(name, "y")
    #         elif name.endswith("_position_z"):
    #             name = name.replace("_position_z", "")
    #             return self.get_position(name, "z")
    #         elif name.endswith("_position"):
    #             name = name.replace("_position", "")
    #             return self.get_position(name, "xyz")
    #         elif name.endswith("_mass"):
    #             name = name.replace("_mass", "")
    #             return self.get_mass(name)
    #         else:
    #             # How to handle infinite recursion?
    #             if "surfaces" in self.__dict__.keys():
    #                 for surface in self.surfaces:
    #                     if name.startswith(f"{surface.name}_"):
    #                         return surface.__getattribute__(name.replace(surface.name, ""))
    #             raise AttributeError(f"Plane doesn't contain attribute {name}")

    # def __setattr__(self, name: str, value: Any) -> None:
    #     if name.endswith("_position_x"):
    #         name = name.replace("_position_x", "")
    #         self.set_position(name, "x", value)
    #     elif name.endswith("_position_y"):
    #         name = name.replace("_position_y", "")
    #         self.set_position(name, "y", value)
    #     elif name.endswith("_position_z"):
    #         name = name.replace("_position_z", "")
    #         self.set_position(name, "z", value)
    #     elif name.endswith("_position"):
    #         name = name.replace("_position", "")
    #         self.set_position(name, "xyz", value)
    #     elif name.endswith("_mass"):
    #         name = name.replace("_mass", "")
    #         self.change_mass(name, new_mass=value)
    #     else:
    #         if hasattr(self, "surfaces"):
    #             for surface in self.surfaces:
    #                 if name.startswith(f"{surface.name}_"):
    #                     surface.__setattr__(name.replace(f"{surface.name}_", ""), value)
    #                     return
    #         object.__setattr__(self, name, value)

    @property
    def directory(self) -> str:
        return f"{self.name}"

    @property
    def CG(self) -> FloatArray:
        if self.overide_mass:
            return self._CG
        return self.find_cg()

    @CG.setter
    def CG(self, cg: FloatArray) -> None:
        self._CG = cg
        self.overide_mass = True
        self.cg_override = cg

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

    def get_seperate_surfaces(self) -> list[WingSurface]:
        surfaces: list[WingSurface] = []
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

    def get_surface(self, name: str) -> WingSurface:
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        raise PlaneDoesntContainAttr(f"Plane doesn't contain attribute {name}")

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
            self.M += mass[0]
        # self.CG = self.find_cg()
        # self.total_inertia = self.find_inertia(self.CG)

    def find_cg(self) -> FloatArray:
        """
        Find the center of gravity of the plane

        Returns:
            Array : X,Y,Z coordinates of the center of gravity
        """
        x_cm = 0.0
        y_cm = 0.0
        z_cm = 0.0
        self.M = 0.0
        for m, r, desc in self.masses:
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
        annotate: bool = False,
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
                mov = np.zeros(3, dtype=float)
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
            if annotate:
                ax.text(
                    r[0] + mov[0],
                    r[1] + mov[1],
                    r[2] + mov[2],
                    "%s" % (desc),
                    size=9,
                    zorder=1,
                    color="k",
                )

        ax.scatter(
            self.CG[0] + mov[0],
            self.CG[1] + mov[1],
            self.CG[2] + mov[2],
            marker="o",
            s=50,
            color="b",
        )
        if annotate:
            ax.text(
                self.CG[0] + mov[0],
                self.CG[1] + mov[1],
                self.CG[2] + mov[2],
                "CG",
                size=9,
                zorder=1,
                color="k",
            )
        plt.show()

    def to_json(self) -> str:
        """
        Pickle the plane object to a json string

        Returns:
            str: Json String
        """
        # If the object is a subclass of Airplane, then we can pickle it as an Airplane object
        if self.__class__ == Airplane.__class__:
            encoded = jsonpickle.encode(self)
        else:
            # print("Converting to Airplane")
            # Encode the object as only an Airplane object
            other = Airplane.__copy__(self)
            # print(f"Other is {other}, {type(other)}")
            encoded = jsonpickle.encode(other)
            del other

        return str(encoded)

    def __setstate__(self, state: dict[str, Any]) -> None:
        Airplane.__init__(
            self,
            name=state["name"],
            surfaces=state["surfaces"],
            orientation=state["orientation"],
            point_masses=state['point_masses'],
            cg_override=state['cg_override'],
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "surfaces": self.surfaces,
            "orientation": self.orientation,
            "point_masses": self.point_masses,
            "cg_override": self.CG if self.overide_mass else None,
        }

    def save(self) -> None:
        """
        Save the plane object to a json file
        """
        from ICARUS.database import DB3D

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

    def __copy__(self) -> Airplane:
        return Airplane(
            name=self.name,
            surfaces=self.surfaces,
            orientation=self.orientation,
            point_masses=self.point_masses,
            cg_override=self.CG if self.overide_mass else None,
        )


class PlaneDoesntContainAttr(AttributeError):
    pass
