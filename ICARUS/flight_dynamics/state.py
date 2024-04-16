from __future__ import annotations

import io
import os
from typing import Any
from typing import TYPE_CHECKING

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from pandas import DataFrame
from pandas import Index
from tabulate import tabulate

from .disturbances import Disturbance as dst
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .stability.lateral import lateral_stability_finite_differences
from .stability.longitudal import longitudal_stability_finite_differences
from .trim import trim_state
from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.environment.definition import Environment
from ICARUS.flight_dynamics.stability.state_space import StateSpace

if TYPE_CHECKING:
    from ICARUS.vehicle.plane import Airplane


class State:
    """Class for the state of a vehicle."""

    def __init__(
        self,
        name: str,
        airplane: Airplane,
        environment: Environment,
        u_freestream: float,
    ) -> None:
        # Set Basic State Variables
        self._name: str = name
        self.environment: Environment = environment
        self._u_freestream = u_freestream

        # Get Airplane Properties And State Variables
        self.mean_aerodynamic_chord: float = airplane.mean_aerodynamic_chord
        self.S: float = airplane.S
        self.span: float = airplane.span
        self.dynamic_pressure: float = 0.5 * environment.air_density * u_freestream**2
        self.inertia: FloatArray = airplane.total_inertia
        self.mass: float = airplane.M

        # Get the airplane control variables
        self.control_vars: set[str] = airplane.control_vars
        self.num_control_vars: int = len(self.control_vars)
        self.control_vector_dict: dict[str, float] = {key: 0.0 for key in self.control_vars}

        # Initialize Trim
        self.trim: dict[str, float] = {}
        self.trim_dynamic_pressure = 0

        # Initialize Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.polar = DataFrame()
        self.disturbances: list[dst] = []
        self.pertrubation_results: DataFrame = DataFrame()
        self.sensitivity: Struct = Struct()
        # self.sensitivity_results = Struct()

        # Initialize The Longitudal State Space Matrices
        # Initialize The Lateral State Space Matrices

    def set_control(self, control_vector_dict: dict[str, float]) -> None:
        # if len(control_vector_dict) != self.num_control_vars:
        #     raise ValueError(
        #         f"Control Vector Length Mismatch: {len(control_vector_dict)} != {self.num_control_vars}"
        #     )
        for key in control_vector_dict:
            if key not in self.control_vars:
                raise ValueError(f"Control Variable Not Found: {key}")
            self.control_vector_dict[key] = control_vector_dict[key]
        # self.__control__()

    def __control__(self) -> None:
        raise NotImplementedError("Control Not Implemented")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def u_freestream(self) -> float:
        return self._u_freestream

    @u_freestream.setter
    def u_freestream(self, value: float) -> None:
        self._u_freestream = value
        self.dynamic_pressure = 0.5 * self.environment.air_density * value**2

    def ground_effect(self) -> bool:
        if self.environment.altitude == 0:
            return False
        if self.environment.altitude > 2 * self.span:
            return False
        else:
            return True

    def update_plane(self, airplane: Airplane) -> None:
        self.mean_aerodynamic_chord = airplane.mean_aerodynamic_chord
        self.S = airplane.S
        self.inertia = airplane.total_inertia
        self.mass = airplane.M

        # Reset Trim
        self.trim = {}
        self.trim_dynamic_pressure = 0

        # Reset Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.polar = DataFrame()
        self.disturbances = []
        self.pertrubation_results = DataFrame()
        self.sensitivity = Struct()
        # self.sensitivity_results = Struct()

    def add_polar(
        self,
        polar: DataFrame,
        polar_prefix: str | None = None,
        is_dimensional: bool = True,
        verbose: bool = True,
    ) -> None:
        # Remove prefix from polar columns
        if polar_prefix is not None:
            cols: list[str] = list(polar.columns)
            if "Fz" not in cols and "CL" not in cols:
                for i, col in enumerate(cols):
                    cols[i] = col.replace(f"{polar_prefix} ", "")
                polar.columns = Index(cols, dtype="str")

        if is_dimensional:
            self.polar = self.make_aero_coefficients(polar)
        else:
            self.polar = polar

        # GET TRIM STATE
        self.trim = trim_state(self, verbose=verbose)
        self.trim_dynamic_pressure = 0.5 * self.environment.air_density * self.trim["U"] ** 2.0  # NOW WE UPDATE IT

    def make_aero_coefficients(self, forces: DataFrame) -> DataFrame:
        data: DataFrame = DataFrame()
        S: float = self.S
        MAC: float = self.mean_aerodynamic_chord
        dynamic_pressure: float = self.dynamic_pressure

        data["CL"] = forces["Fz"] / (dynamic_pressure * S)
        data["CD"] = forces["Fx"] / (dynamic_pressure * S)
        data["Cm"] = forces["My"] / (dynamic_pressure * S * MAC)
        data["CL/CD"] = data["CL"] / data["CD"]
        # data["Cn"] = forces["Mz"] / (dynamic_pressure * S * span)
        # data["Cl"] = forces["Mx"] / (dynamic_pressure * S * span)
        data["AoA"] = forces["AoA"]
        return data

    def add_all_pertrubations(
        self,
        scheme: str,
        epsilon: dict[str, float] | None = None,
    ) -> None:
        """Function to add a perturbations to the airplane for
        dynamic analysis
        Inputs:
        - scheme: "Central", "Forward", "Backward"
        - epsilon: Disturbance Magnitudes
        """
        self.scheme: str = scheme
        self.epsilons: dict[str, float] = {}

        self.disturbances = [
            *longitudal_pertrubations(self, scheme, epsilon),
            *lateral_pertrubations(self, scheme, epsilon),
        ]
        self.disturbances.append(dst(None, 0))

    def sensitivity_analysis(self, var: str, space: list[float] | FloatArray) -> None:
        self.sensitivity[var] = []
        for e in space:
            self.sensitivity[var].append(dst(var, e))

    def get_pertrub(self) -> None:
        for disturbance in self.disturbances:
            print(disturbance)

    def set_pertrubation_results(
        self,
        pertrubation_results: DataFrame,
    ) -> None:
        self.pertrubation_results = pertrubation_results
        self.stability_fd()

    def stability_fd(self) -> None:
        longitudal_state_space = longitudal_stability_finite_differences(self)
        lateral_state_space = lateral_stability_finite_differences(self)
        self.state_space = StateSpace(longitudal_state_space, lateral_state_space)

    def plot_eigenvalues(self, plot_lateral: bool = True, plot_longitudal: bool = True) -> tuple[Figure, list[Axes]]:
        """
        Generate a plot of the eigenvalues.
        """
        fig = plt.figure()

        if plot_lateral and plot_longitudal:
            axs: list[Axes] = fig.subplots(1, 2)  # type: ignore
        else:
            axs0 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
            axs = [axs0]
        i = 0
        if plot_longitudal:
            # extract real part
            x: list[float] = [ele.real for ele in self.state_space.longitudal.eigenvalues]
            # extract imaginary part
            y: list[float] = [ele.imag for ele in self.state_space.longitudal.eigenvalues]
            axs[i].scatter(x, y, label="Longitudal", color="r")
            i += 1

        if plot_lateral:
            # extract real part
            x = [ele.real for ele in self.state_space.lateral.eigenvalues]
            # extract imaginary part
            y = [ele.imag for ele in self.state_space.lateral.eigenvalues]
            marker_x = MarkerStyle("x")
            axs[i].scatter(x, y, label="Lateral", color="b", marker=marker_x)

        for j in range(0, i + 1):
            axs[j].set_ylabel("Imaginary")
            axs[j].set_xlabel("Real")
            axs[j].grid()
            axs[j].axvline(0, color="black", lw=2)
            axs[j].legend()
        fig.show()
        return fig, axs

    def __str__(self) -> str:
        ss = io.StringIO()
        ss.write(f"State: {self.name}\n")
        ss.write(f"Trim: {self.trim}\n")
        ss.write(f"\n{45*'--'}\n")

        if hasattr(self, "state_space"):
            ss.write("\nLongitudal State:\n")
            ss.write(
                f"Eigen Values: {[round(item,3) for item in self.state_space.longitudal.eigenvalues]}\n",
            )
            ss.write("Eigen Vectors:\n")
            for item in self.state_space.longitudal.eigenvectors:
                ss.write(f"\t{[round(i,3) for i in item]}\n")
            ss.write("\nThe State Space Matrix:\n")
            ss.write(
                tabulate(self.state_space.longitudal.A, tablefmt="github", floatfmt=".3f"),
            )

            ss.write(f"\n\n{45*'--'}\n")

            ss.write("\nLateral State:\n")
            ss.write(
                f"Eigen Values: {[round(item,3) for item in self.state_space.lateral.eigenvalues]}\n",
            )
            ss.write("Eigen Vectors:\n")
            for item in self.state_space.lateral.eigenvectors:
                ss.write(f"\t{[round(i,3) for i in item]}\n")
            ss.write("\nThe State Space Matrix:\n")
            ss.write(tabulate(self.state_space.lateral.A, tablefmt="github", floatfmt=".3f"))
        return ss.getvalue()

    def to_json(self) -> str:
        """
        Pickle the state object to a json string.

        Returns:
            str: Json String
        """
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def save(self, directory: str) -> None:
        """
        Save the state object to a json file.

        Args:
            directory (str): Directory to save the state to.

        """
        fname: str = os.path.join(directory, f"{self.name}_state.json")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        # TODO ADD A FILE CONTAINING ENUMERATION OF CASES AND CONTROL VECTORS

    @property
    def a_long(self) -> Any:
        return self.state_space.longitudal.A

    @a_long.setter
    def a_long(self, value: FloatArray) -> None:
        self.state_space.longitudal.A = value

    @property
    def astar_long(self) -> Any:
        return self.state_space.longitudal.A_DS

    @astar_long.setter
    def astar_long(self, value: FloatArray) -> None:
        self.state_space.longitudal.A_DS = value

    @property
    def a_lat(self) -> Any:
        return self.state_space.lateral.A

    @a_lat.setter
    def a_lat(self, value: FloatArray) -> None:
        self.state_space.lateral.A = value

    @property
    def astar_lat(self) -> Any:
        return self.state_space.lateral.A_DS

    @astar_lat.setter
    def astar_lat(self, value: FloatArray) -> None:
        self.state_space.lateral.A_DS = value
