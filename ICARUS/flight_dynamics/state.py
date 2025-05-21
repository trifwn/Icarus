from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from typing import Any

import distinctipy
import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from matplotlib.markers import MarkerStyle
from pandas import DataFrame
from pandas import Series
from tabulate import tabulate

from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.environment.definition import Environment
from ICARUS.flight_dynamics.stability.state_space import StateSpace

from .disturbances import Disturbance as dst
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .stability.lateral import lateral_stability_finite_differences
from .stability.longitudal import longitudal_stability_finite_differences
from .trim import TrimNotPossible
from .trim import TrimOutsidePolars
from .trim import trim_state

if TYPE_CHECKING:
    from ICARUS.vehicle.airplane import Airplane


class ControlState:
    def __init__(
        self,
        airplane: Airplane,
    ) -> None:
        # Get the airplane control variables
        self.control_vars: set[str] = airplane.control_vars
        self.num_control_vars: int = len(self.control_vars)
        self.control_vector_dict: dict[str, float] = airplane.control_vector
        self.hash_dict: dict[str, int] = {}

    def update(self, control_vector_dict: dict[str, float]) -> None:
        self.control_vector_dict = control_vector_dict

    @property
    def control_vector(self) -> FloatArray:
        control_vector = np.array(
            [self.control_vector_dict[key] for key in self.control_vars],
        )
        return control_vector

    def __str__(self) -> str:
        string = "Control State: "
        for key in self.control_vars:
            string += f"{key}: {self.control_vector_dict[key]:.3f} "
        return string

    def __hash__(self) -> int:
        """Unique hash for the control state. This is used to generate a unique name for the state.
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
        self.trim_dynamic_pressure = 0.0

        # Initialize Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.scheme: str = "Central"
        self.epsilons: dict[str, float] = {}
        self.disturbances: list[dst] = []
        self.pertrubation_results: DataFrame = DataFrame()
        self.sensitivities: Struct = Struct()

        # Polars
        self.polar: DataFrame = DataFrame()
        self.polar_prefix: str | None = None

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
    def inertia(self) -> tuple[float, float, float, float, float, float]:
        Ix = float(self.airplane.inertia[0])
        Iy = float(self.airplane.inertia[1])
        Iz = float(self.airplane.inertia[2])
        Ixz = float(self.airplane.inertia[3])
        Ixy = float(self.airplane.inertia[4])
        Iyz = float(self.airplane.inertia[5])
        return Ix, Iy, Iz, Ixz, Ixy, Iyz

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
            [control_vector_dict[key] for key in self.control_vars],
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
            if is_dimensional:
                if "Fz" not in cols:
                    polar["Fz"] = polar[f"{polar_prefix} Fz"]
                    polar["Fx"] = polar[f"{polar_prefix} Fx"]
                    polar["My"] = polar[f"{polar_prefix} My"]
            else:
                if "CL" not in cols:
                    polar["CL"] = polar[f"{polar_prefix} CL"]
                    polar["CD"] = polar[f"{polar_prefix} CD"]
                    polar["Cm"] = polar[f"{polar_prefix} Cm"]
            self.polar_prefix = polar_prefix

        if is_dimensional:
            polar = self.make_aero_coefficients(polar)

        # Merge the df with the old data on the AoA column
        if not self.polar.empty:
            for col in polar.keys():
                if col in self.polar.columns and col != "AoA":
                    self.polar.drop(col, axis=1, inplace=True)

            self.polar = self.polar.merge(
                polar,
                on="AoA",
                how="outer",
            )
        else:
            self.polar = polar

        # GET TRIM STATE
        try:
            self.trim = trim_state(self, verbose=verbose)
            self.trim_dynamic_pressure = 0.5 * self.environment.air_density * self.trim["U"] ** 2.0
        except (TrimNotPossible, TrimOutsidePolars) as e:
            self.trimmable = False
            self.trim = {}
            self.trim_dynamic_pressure = np.nan
            raise e

    def get_polar_prefixes(self) -> list[str]:
        cols_pol = [col for col in self.polar.columns if "CL" in col]
        return list({col[:-3] for col in cols_pol if col != "CL"})

    def print_trim(self) -> None:
        print(self.trim_to_string())

    def trim_to_string(self) -> str:
        ss = io.StringIO()
        if not self.trim:
            ss.write("State Not Trimmed")
        else:
            ss.write(f"Trim State (Calculated with {self.polar_prefix})\n")
            for key, value in self.trim.items():
                ss.write(f"\t{key}: {value:.3f}\n")

        return ss.getvalue()

    def change_polar_prefix(self, polar_prefix: str) -> None:
        self.polar_prefix = polar_prefix
        self.polar["CL"] = self.polar[f"{polar_prefix} CL"]
        self.polar["CD"] = self.polar[f"{polar_prefix} CD"]
        self.polar["Cm"] = self.polar[f"{polar_prefix} Cm"]
        try:
            self.trim = trim_state(self, verbose=False)
            self.trim_dynamic_pressure = 0.5 * self.environment.air_density * self.trim["U"] ** 2.0
        except TrimNotPossible:
            self.trim_dynamic_pressure = np.nan
            self.trim = {}
            self.trimmable = False

    def change_pertrubation_prefix(self, pertrubation_prefix: str) -> None:
        self.pertrubation_results["Fx"] = self.pertrubation_results[f"{pertrubation_prefix} Fx"]
        self.pertrubation_results["Fy"] = self.pertrubation_results[f"{pertrubation_prefix} Fy"]
        self.pertrubation_results["Fz"] = self.pertrubation_results[f"{pertrubation_prefix} Fz"]
        self.pertrubation_results["Mx"] = self.pertrubation_results[f"{pertrubation_prefix} Mx"]
        self.pertrubation_results["My"] = self.pertrubation_results[f"{pertrubation_prefix} My"]
        self.pertrubation_results["Mz"] = self.pertrubation_results[f"{pertrubation_prefix} Mz"]
        self.stability_fd()

    def make_aero_coefficients(self, forces: DataFrame) -> DataFrame:
        data: DataFrame = DataFrame()
        S: float = self.S
        MAC: float = self.mean_aerodynamic_chord
        dynamic_pressure: float = self.dynamic_pressure

        for key in forces.columns:
            if key.endswith("Fz"):
                fz = forces[key]
                data[f"{key[:-2]}CL"] = fz / (dynamic_pressure * S)
            if key.endswith("Fx"):
                fy = forces[key]
                data[f"{key[:-2]}CD"] = fy / (dynamic_pressure * S)
            if key.endswith("My"):
                mx = forces[key]
                data[f"{key[:-2]}Cm"] = mx / (dynamic_pressure * S * MAC)
            if key.endswith("AoA"):
                data["AoA"] = forces[key]
        return data

    def add_all_pertrubations(
        self,
        scheme: str | None = None,
        epsilon: dict[str, float] | None = None,
    ) -> None:
        """Function to add a perturbations to the airplane for
        dynamic analysis
        Inputs:
        - scheme: "Central", "Forward", "Backward"
        - epsilon: Disturbance Magnitudes
        """
        if scheme is None:
            scheme = "Central"
        self.scheme = scheme
        self.epsilons = {}

        self.disturbances = [
            *longitudal_pertrubations(self, scheme, epsilon),
            *lateral_pertrubations(self, scheme, epsilon),
        ]
        self.disturbances.append(dst(None, 0))

    def sensitivity_analysis(self, var: str, space: list[float] | FloatArray) -> None:
        self.sensitivities[var] = []
        for e in space:
            self.sensitivities[var].append(dst(var, float(e)))

    def get_pertrub(self) -> None:
        for disturbance in self.disturbances:
            print(disturbance)

    def set_pertrubation_results(
        self,
        pertrubation_results: DataFrame,
        polar_prefix: str | None = None,
    ) -> None:
        # Remove prefix from polar columns
        if polar_prefix is not None:
            cols: list[str] = list(pertrubation_results.columns)
            if "Fx" not in cols:
                pertrubation_results["Fx"] = pertrubation_results[f"{polar_prefix} Fx"]
                pertrubation_results["Fy"] = pertrubation_results[f"{polar_prefix} Fy"]
                pertrubation_results["Fz"] = pertrubation_results[f"{polar_prefix} Fz"]
                pertrubation_results["Mx"] = pertrubation_results[f"{polar_prefix} Mx"]
                pertrubation_results["My"] = pertrubation_results[f"{polar_prefix} My"]
                pertrubation_results["Mz"] = pertrubation_results[f"{polar_prefix} Mz"]

        if not pertrubation_results.empty:
            for col in pertrubation_results.columns:
                if col in self.polar.columns and col != "Epsilon" and col != "Type":
                    self.polar.drop(col, axis=1, inplace=True)

            self.pertrubation_results = pertrubation_results
        else:
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
        title: str | None = None,
    ) -> tuple[list[Axes], Figure | SubFigure]:
        """Generate a plot of the eigenvalues."""
        if axs is not None:
            fig: Figure | SubFigure | None = axs[0].figure
            if fig is None:
                fig = plt.figure()
                fig.suptitle(f"Eigenvalues for {self.airplane.name} at state {self.name}")
            axs_now: list[Axes] = axs
        else:
            fig = plt.figure()
            if title is not None:
                fig.suptitle(title)
            else:
                fig.suptitle(f"Eigenvalues for {self.airplane.name} at state {self.name}")

            if plot_lateral and plot_longitudal:
                axs_now = fig.subplots(1, 2)  # type: ignore
            else:
                axs0 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
                axs_now = [axs0]

        i = 0
        if plot_longitudal:
            axs_now[i].set_title("Longitudal Eigenvalues")
            # extract real part and imaginary part
            x: list[float] = [ele.real for ele in self.state_space.longitudal.eigenvalues]
            y: list[float] = [ele.imag for ele in self.state_space.longitudal.eigenvalues]

            # Get the xs where x>0
            x_pos = [x[j] for j in range(len(x)) if x[j] > 0]
            y_pos = [y[j] for j in range(len(x)) if x[j] > 0]

            # Get the xs where x<0
            x_neg = [x[j] for j in range(len(x)) if x[j] < 0]
            y_neg = [y[j] for j in range(len(x)) if x[j] < 0]

            axs_now[i].scatter(x_pos, y_pos, label="Longitudal", color="r", marker=MarkerStyle("x"))
            axs_now[i].scatter(x_neg, y_neg, label="Longitudal", color="r", marker=MarkerStyle("o"))
            i += 1

        if plot_lateral:
            axs_now[i].set_title("Lateral Eigenvalues")
            # extract real and imaginary parts
            x = [ele.real for ele in self.state_space.lateral.eigenvalues]
            y = [ele.imag for ele in self.state_space.lateral.eigenvalues]

            # Get the xs where x>0
            x_pos = [x[j] for j in range(len(x)) if x[j] > 0]
            y_pos = [y[j] for j in range(len(x)) if x[j] > 0]

            # Get the xs where x<0
            x_neg = [x[j] for j in range(len(x)) if x[j] < 0]
            y_neg = [y[j] for j in range(len(x)) if x[j] < 0]

            axs_now[i].scatter(x_pos, y_pos, label="Lateral", color="b", marker=MarkerStyle("x"))
            axs_now[i].scatter(x_neg, y_neg, label="Lateral", color="b", marker=MarkerStyle("o"))
        for j in range(i + 1):
            axs_now[j].set_ylabel("Imaginary")
            axs_now[j].set_xlabel("Real")
            axs_now[j].axvline(0, color="black", lw=2)
            axs_now[j].grid(True)
            # axs[j].legend()
        if isinstance(fig, SubFigure):
            pass
        else:
            fig.show()
        return axs_now, fig

    def plot_polars(
        self,
        prefixes: list[str] | None = None,
        plots: list[list[str]] = [
            ["AoA", "CL"],
            ["AoA", "CD"],
            ["AoA", "Cm"],
            ["AoA", "CL/CD"],
        ],
        dimensional: bool = False,
        title: str | None = None,
    ) -> tuple[Figure, np.ndarray[Any, Any]]:
        """Function to plot stored polars

        Args:
            prefixes (list[str], optional): List of solvers to plot. Defaults to ["All"].
            plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
            dimensional (bool, optional): If true we convert to forces and moments. Defaults to False.
            title (str, optional): Figure Title. Defaults to "Aero Coefficients".

        Returns:
            tuple[ndarray, Figure]: Array of Axes and Figure
        """
        if title is None:
            title = f"{self.airplane.name}: {self.name}"
            if dimensional:
                title += " Aerodynamic Forces"
            else:
                title += " Aerodynamic Coefficients"
        number_of_plots = len(plots) + 1
        # Divide the plots equally
        sqrt_num = number_of_plots**0.5
        i: int = int(np.ceil(sqrt_num))
        j: int = int(np.floor(sqrt_num))

        fig: Figure = plt.figure(figsize=(10, 10))
        axs: np.ndarray[Axes] = fig.subplots(i, j)  # type: ignore
        fig.suptitle(f"{title}", fontsize=16)

        for plot, ax in zip(plots, axs.flatten()[: len(plots)]):
            ax.set_xlabel(plot[0])
            ax.set_ylabel(plot[1])
            ax.set_title(f"{plot[1]} vs {plot[0]}")
            ax.grid()
            ax.axhline(y=0, color="k")
            ax.axvline(x=0, color="k")

        prefixes_available = self.get_polar_prefixes()
        if prefixes is None:
            prefixes_to_plot = [prefix for prefix in prefixes_available if not prefix.endswith("ONERA")]
        else:
            prefixes_to_plot = [prefix for prefix in prefixes if prefix in prefixes_available]

        colors_ = distinctipy.get_colors(len(prefixes_to_plot))
        axs = axs.flatten()
        polar: DataFrame = self.polar.copy()

        S: float = self.S
        MAC: float = self.mean_aerodynamic_chord
        dynamic_pressure: float = self.dynamic_pressure

        for j, prefix in enumerate(prefixes_to_plot):
            if dimensional:
                polar[f"{prefix} CL"] = polar[f"{prefix} CL"] * dynamic_pressure * S
                polar[f"{prefix} CD"] = polar[f"{prefix} CD"] * dynamic_pressure * S
                polar[f"{prefix} Cm"] = polar[f"{prefix} Cm"] * dynamic_pressure * S * MAC

            for plot, ax in zip(plots, axs[: len(plots)]):
                if plot[0] == "CL/CD" or plot[1] == "CL/CD":
                    polar[f"{prefix} CL/CD"] = polar[f"{prefix} CL"] / polar[f"{prefix} CD"]
                if plot[0] == "CD/CL" or plot[1] == "CD/CL":
                    polar[f"{prefix} CD/CL"] = polar[f"{prefix} CD"] / polar[f"{prefix} CL"]

                key0 = f"{prefix} {plot[0]}"
                key1 = f"{prefix} {plot[1]}"

                if plot[0] == "AoA":
                    key0 = "AoA"
                if plot[1] == "AoA":
                    key1 = "AoA"

                x: Series[float] = polar[f"{key0}"]
                y: Series[float] = polar[f"{key1}"]
                c = colors_[j]
                ax.plot(
                    x,
                    y,
                    ls="--",
                    color=c,
                    label=f" {prefix}",
                    marker=MarkerStyle("o"),
                    markersize=5,
                    linewidth=1.5,
                )

        # Remove empty plots
        for ax in axs.flatten()[len(plots) :]:
            ax.remove()

        handles, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", ncol=2)
        # Adjust the plots
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.1)
        fig.show()
        return fig, axs

    def __str__(self) -> str:
        ss = io.StringIO()
        ss.write(f"State: {self.name}\n")
        ss.write(f"{self.trim_to_string()}\n")
        ss.write(f"\n{45 * '--'}\n")

        def convert_to_str(num: complex) -> str:
            # CHeck is it complex
            try:
                return f"{num.real:.3f} {num.imag:+.3f}j"
            except AttributeError:
                return f"{num:.3f}"

        if hasattr(self, "state_space"):
            ss.write("\nLongitudal State:\n")
            ss.write(
                f"Eigen Values: {[convert_to_str(item) for item in self.state_space.longitudal.eigenvalues]}\n",
            )
            ss.write("Eigen Vectors:\n")
            for item in self.state_space.longitudal.eigenvectors:
                ss.write(f"\t{[convert_to_str(i) for i in item]}\n")
            ss.write("\nThe State Space Matrix:\n")
            ss.write(
                tabulate(
                    self.state_space.longitudal.A,
                    tablefmt="github",
                    floatfmt=".3f",
                ),
            )

            ss.write(f"\n\n{45 * '--'}\n")

            ss.write("\nLateral State:\n")
            ss.write(
                f"Eigen Values: {[convert_to_str(item) for item in self.state_space.lateral.eigenvalues]}\n",
            )
            ss.write("Eigen Vectors:\n")
            for item in self.state_space.lateral.eigenvectors:
                ss.write(f"\t{[convert_to_str(i) for i in item]}\n")
            ss.write("\nThe State Space Matrix:\n")
            ss.write(
                tabulate(self.state_space.lateral.A, tablefmt="github", floatfmt=".3f"),
            )
        return ss.getvalue()

    def to_json(self) -> str:
        """Pickle the state object to a json string.

        Returns:
            str: Json String

        """
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def save(self, directory: str) -> None:
        """Save the state object to a json file.

        Args:
            directory (str): Directory to save the state to.

        """
        # Check if directory exists
        os.makedirs(directory, exist_ok=True)
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
