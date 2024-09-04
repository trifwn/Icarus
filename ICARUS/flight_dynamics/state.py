from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from typing import Any

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from matplotlib.markers import MarkerStyle
from pandas import DataFrame
from pandas import Index
from tabulate import tabulate
from traitlets import Float

from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.environment.definition import Environment
from ICARUS.flight_dynamics.stability.state_space import StateSpace

from .disturbances import Disturbance as dst
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .stability.lateral import lateral_stability_finite_differences
from .stability.longitudal import longitudal_stability_finite_differences
from .trim import trim_state

if TYPE_CHECKING:
    from ICARUS.vehicle.plane import Airplane


class ControlState:
    def __init__(
        self,
        airplane: Airplane,
    ) -> None:
        # Get the airplane control variables
        self.control_vars: set[str] = airplane.control_vars
        self.num_control_vars: int = len(self.control_vars)
        self.control_vector_dict: dict[str, float] = airplane.control_vector
        self.hash_dict: dict[int, int] = {}

    def update(self, control_vector_dict: dict[str, float]) -> None:
        self.control_vector_dict = control_vector_dict

    @property
    def control_vector(self) -> FloatArray:
        control_vector = np.array(
            [self.control_vector_dict[key] for key in self.control_vars]
        )
        return control_vector

    def __str__(self) -> str:
        return f"Control Variables: {self.control_vars}"

    def __hash__(self) -> int:
        """
        Unique hash for the control state. This is used to generate a unique name for the state.
        It depends on the control variables and their values.


        Raises:
            ValueError: _description_

        Returns:
            int: _description_
        """
        hash_val = hash(frozenset(self.control_vector_dict.items()))
        # Add to the hash dictionary if not already present
        if str(hash_val) not in list(self.hash_dict.keys()):
            self.hash_dict[str(hash_val)] = len(self.hash_dict)

        return hash_val

    def identifier(self) -> int:
        num = self.__hash__()
        return self.hash_dict[str(num)]


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

        # Store reference to the airplane
        self.airplane: Airplane = airplane

        # Get the airplane control variables
        self.control_state: ControlState = ControlState(airplane)

        # Initialize Trim
        self.trim: dict[str, float] = {}
        self.trim_dynamic_pressure = 0

        # Initialize Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.polar: DataFrame = DataFrame()
        self.disturbances: list[dst] = []
        self.pertrubation_results: DataFrame = DataFrame()
        self.sensitivities: Struct = Struct()

    @property
    def name(self) -> str:
        name = self._name
        # Get the airplanes control variables
        identifier = self.control_state.identifier()
        if identifier != 0:
            name += f"_{identifier}"
        return name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def u_freestream(self) -> float:
        return self._u_freestream

    @u_freestream.setter
    def u_freestream(self, value: float) -> None:
        self._u_freestream = value

    @property
    def dynamic_pressure(self) -> float:
        Q = 0.5 * self.environment.air_density * self.u_freestream**2
        return Q

    ##################### Airplane Properties ############################

    @property
    def CG(self) -> FloatArray:
        return self.airplane.CG

    @property
    def mean_aerodynamic_chord(self) -> float:
        return self.airplane.mean_aerodynamic_chord

    @property
    def S(self) -> float:
        return self.airplane.S

    @property
    def span(self) -> float:
        return self.airplane.span

    @property
    def inertia(self) -> FloatArray:
        return self.airplane.total_inertia

    @property
    def mass(self) -> float:
        return self.airplane.M

    ##################### END Airplane Properties ############################

    ########################### CONTROL  ################################

    @property
    def control_vars(self) -> set[str]:
        return self.control_state.control_vars

    @property
    def num_control_vars(self) -> int:
        return self.control_state.num_control_vars

    @property
    def control_vector_dict(self) -> dict[str, float]:
        # Update the control state
        # self.control_state.control_vector_dict
        airplane_val = self.airplane.control_vector
        self.control_state.update(airplane_val)
        return self.control_state.control_vector_dict

    @property
    def control_vector(self) -> FloatArray:
        control_vector_dict = self.control_vector_dict
        control_vector = np.array(
            [control_vector_dict[key] for key in self.control_vars]
        )
        return control_vector

    def set_control(self, control_vector_dict: dict[str, float]) -> None:
        for key in control_vector_dict:
            if key not in self.control_vars:
                raise ValueError(f"Control Variable Not Found: {key}")
        self.control_state.update(control_vector_dict)
        self.airplane.__control__(control_vector_dict)

    ########################### END CONTROL  ################################

    def ground_effect(self) -> bool:
        if self.environment.altitude == 0:
            return False
        if self.environment.altitude > 2 * self.span:
            return False
        else:
            return True

    def update_plane(self, airplane: Airplane) -> None:
        self.airplane = airplane

        # Reset Trim
        self.trim = {}
        self.trim_dynamic_pressure = 0

        # Reset Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.polar = DataFrame()
        self.disturbances = []
        self.pertrubation_results = DataFrame()
        self.sensitivities = Struct()

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
        self.trim_dynamic_pressure = (
            0.5 * self.environment.air_density * self.trim["U"] ** 2.0
        )  # NOW WE UPDATE IT

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
        self.sensitivities[var] = []
        for e in space:
            self.sensitivities[var].append(dst(var, e))

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

    def plot_eigenvalues(
        self,
        plot_lateral: bool = True,
        plot_longitudal: bool = True,
        axs: list[Axes] | None = None,
    ) -> None:
        """
        Generate a plot of the eigenvalues.
        """
        if axs is not None:
            fig: Figure | SubFigure | None = axs[0].figure
            if fig is None:
                fig = plt.figure()
                fig.suptitle(f"Eigenvalues")
            axs_now: list[Axes] = axs
        else:
            fig = plt.figure()
            fig.suptitle(f"Eigenvalues")

            if plot_lateral and plot_longitudal:
                axs_now = fig.subplots(1, 2)  # type: ignore
            else:
                axs0 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
                axs_now = [axs0]

        i = 0
        if plot_longitudal:
            # extract real part
            x: list[float] = [
                ele.real for ele in self.state_space.longitudal.eigenvalues
            ]
            # extract imaginary part
            y: list[float] = [
                ele.imag for ele in self.state_space.longitudal.eigenvalues
            ]
            axs_now[i].scatter(x, y, label="Longitudal", color="r")
            i += 1

        if plot_lateral:
            # extract real part
            x = [ele.real for ele in self.state_space.lateral.eigenvalues]
            # extract imaginary part
            y = [ele.imag for ele in self.state_space.lateral.eigenvalues]
            marker_x = MarkerStyle("x")
            axs_now[i].scatter(x, y, label="Lateral", color="b", marker=marker_x)

        for j in range(0, i + 1):
            axs_now[j].set_ylabel("Imaginary")
            axs_now[j].set_xlabel("Real")
            axs_now[j].axvline(0, color="black", lw=2)
            axs_now[j].grid(True)
            # axs[j].legend()
        if isinstance(fig, SubFigure):
            pass
        else:
            fig.show()

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
                tabulate(
                    self.state_space.longitudal.A, tablefmt="github", floatfmt=".3f"
                ),
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
            ss.write(
                tabulate(self.state_space.lateral.A, tablefmt="github", floatfmt=".3f")
            )
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
