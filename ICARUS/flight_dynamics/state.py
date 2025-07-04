from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from typing import Any

import jsonpickle
import numpy as np
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from pandas import DataFrame
from pandas import Series
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray
from ICARUS.environment import Environment
from ICARUS.visualization import markers
from ICARUS.visualization import polar_plot
from ICARUS.visualization import pre_existing_figure

from .disturbances import Disturbance
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .stability import StateSpace
from .stability.lateral import lateral_stability_finite_differences
from .stability.longitudal import longitudal_stability_finite_differences
from .trim import TrimNotPossible
from .trim import TrimOutsidePolars
from .trim import trim_state

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane


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
        return np.array(
            [self.control_vector_dict[key] for key in self.control_vars],
        )

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
        self.disturbances: list[Disturbance] = []
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
        return self.control_state.control_vector_dict.copy()

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
            self.trim = trim_state(self)
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
            self.trim = trim_state(self)
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
        self.disturbances.append(Disturbance(None, 0))

    def sensitivity_analysis(self, var: str, space: list[float] | FloatArray) -> None:
        self.sensitivities[var] = []
        for e in space:
            self.sensitivities[var].append(Disturbance(var, float(e)))

    def print_pertrubations(self) -> None:
        console = Console()
        table = Table(title="Disturbance Summary", show_lines=True)

        table.add_column("Name", style="bold cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Amplitude", justify="right", style="yellow")
        table.add_column("Axis", style="green")

        for d in self.disturbances:
            table.add_row(
                d.name,
                f"Rotational {d.type}" if d.is_rotational else f"Translational {d.type}",
                f"{d.amplitude:.3f}" if d.amplitude is not None else "N/A",
                f"{d.axis if d.axis is not None else 'N/A'}",
            )
        # Center the table
        centered_table = Align.center(table)
        console.print(centered_table)

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

    @pre_existing_figure(subplots=(1, 2), default_title="AVL Eigenvalues")
    def plot_eigenvalues(
        self,
        axs: list[Axes],
        plot_lateral: bool = True,
        plot_longitudal: bool = True,
    ) -> None:
        """Generate a plot of the eigenvalues."""
        # Initialize counter for axes indexing
        i = 0
        # Longitudal plot
        if plot_longitudal:
            ax = axs[i]
            ax.set_title("Longitudal Eigenvalues")
            x = [ev.real for ev in self.state_space.longitudal.eigenvalues]
            y = [ev.imag for ev in self.state_space.longitudal.eigenvalues]
            x_pos = [xj for xj in x if xj > 0]
            y_pos = [yj for xj, yj in zip(x, y) if xj > 0]
            x_neg = [xj for xj in x if xj < 0]
            y_neg = [yj for xj, yj in zip(x, y) if xj < 0]
            ax.scatter(x_pos, y_pos, label="Longitudal", color="r", marker=MarkerStyle("x"))
            ax.scatter(x_neg, y_neg, label="Longitudal", color="r", marker=MarkerStyle("o"))
            i += 1

        # Lateral plot
        if plot_lateral:
            ax = axs[i]
            ax.set_title("Lateral Eigenvalues")
            x = [ev.real for ev in self.state_space.lateral.eigenvalues]
            y = [ev.imag for ev in self.state_space.lateral.eigenvalues]
            x_pos = [xj for xj in x if xj > 0]
            y_pos = [yj for xj, yj in zip(x, y) if xj > 0]
            x_neg = [xj for xj in x if xj < 0]
            y_neg = [yj for xj, yj in zip(x, y) if xj < 0]
            ax.scatter(x_pos, y_pos, label="Lateral", color="b", marker=MarkerStyle("x"))
            ax.scatter(x_neg, y_neg, label="Lateral", color="b", marker=MarkerStyle("o"))

        # Common styling
        for ax in axs[: i + 1]:
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            ax.axvline(0, color="black", lw=2)
            ax.grid(True)

    @polar_plot(default_title="Polar Plots")
    def plot_polars(
        self,
        axs: list[Axes],
        prefixes: list[str] | None = None,
        plots: list[list[str]] = [
            ["AoA", "CL"],
            ["AoA", "CD"],
            ["AoA", "Cm"],
            ["AoA", "CL/CD"],
        ],
        dimensional: bool = False,
        color: list[float] | None = None,
        label: str | None = None,
        title: str | None = None,
    ) -> None:
        """Function to plot stored polars

        Args:
            prefixes (list[str], optional): List of solvers to plot. Defaults to ["All"].
            plots (list[list[str]], optional): List of plots to plot. Defaults to [["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"], ["CL", "CD"]].
            dimensional (bool, optional): If true we convert to forces and moments. Defaults to False.
            title (str, optional): Figure Title. Defaults to "Aero Coefficients".

        Returns:
            tuple[ndarray, Figure]: Array of Axes and Figure
        """
        if title is not None and axs[0].figure:
            axs[0].figure.suptitle(title)

        for plot, ax in zip(plots, axs[: len(plots)]):
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
                ax.plot(
                    x,
                    y,
                    ls="--",
                    color=color if color is not None else None,
                    label=f"{self.name} {prefix}" if label is None else f"{label} {prefix}",
                    marker=markers[j],
                    markersize=5,
                    linewidth=1.5,
                )

    def __str__(self) -> str:
        """Rich string representation of the State."""
        console = Console(file=io.StringIO(), force_terminal=False, width=120)

        # Main Panel for the state
        trim_info = self.trim_to_string()
        panel_content = f"State: [bold]{self.name}[/bold]\n{trim_info}"
        console.print(Panel(panel_content, title="[bold green]State Information[/bold green]", border_style="blue"))

        def convert_to_str(num: complex) -> str:
            try:
                return f"{num.real:.3f} {num.imag:+.3f}j"
            except AttributeError:
                return f"{num:.3f}"

        if hasattr(self, "state_space"):
            # Longitudal
            long_eigen_values = ", ".join([convert_to_str(item) for item in self.state_space.longitudal.eigenvalues])
            long_eigen_vectors = "\n".join(
                "  " + ", ".join([convert_to_str(i) for i in item]) for item in self.state_space.longitudal.eigenvectors
            )
            long_panel_content = (
                f"[bold]Eigen Values:[/bold] {long_eigen_values}\n[bold]Eigen Vectors:[/bold]\n{long_eigen_vectors}"
            )
            long_panel = Panel(long_panel_content, title="[cyan]Longitudal State[/cyan]", border_style="cyan")
            console.print(long_panel)

            long_table = Table(title="State Space Matrix", show_header=False)
            for row in self.state_space.longitudal.A:
                long_table.add_row(*[f"{item:.3f}" for item in row])
            console.print(long_table)

            # Lateral
            lat_eigen_values = ", ".join([convert_to_str(item) for item in self.state_space.lateral.eigenvalues])
            lat_eigen_vectors = "\n".join(
                "  " + ", ".join([convert_to_str(i) for i in item]) for item in self.state_space.lateral.eigenvectors
            )
            lat_panel_content = (
                f"[bold]Eigen Values:[/bold] {lat_eigen_values}\n[bold]Eigen Vectors:[/bold]\n{lat_eigen_vectors}"
            )
            lat_panel = Panel(lat_panel_content, title="[magenta]Lateral State[/magenta]", border_style="magenta")
            console.print(lat_panel)

            lat_table = Table(title="State Space Matrix", show_header=False)
            for row in self.state_space.lateral.A:
                lat_table.add_row(*[f"{item:.3f}" for item in row])
            console.print(lat_table)

        return str(console._file)

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
