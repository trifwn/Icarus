from __future__ import annotations

import io
import os
from typing import Any
from typing import TYPE_CHECKING

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from pandas import DataFrame
from tabulate import tabulate

from .disturbances import Disturbance as dst
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .Stability.lateralFD import lateral_stability
from .Stability.longitudalFD import longitudal_stability
from .Stability.stability_derivatives import StabilityDerivativesDS
from .trim import trim_state
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Environment.definition import Environment
from ICARUS.Input_Output.GenuVP.post_process.forces import rotate_forces

if TYPE_CHECKING:
    from ICARUS.Vehicle.plane import Airplane


class State:
    """Class for the state of a vehicle."""

    def __init__(
        self,
        name: str,
        pln: Airplane,
        forces: DataFrame,
        env: Environment,
    ) -> None:
        # Set Basic State Variables
        self.name: str = name
        # self.vehicle: Airplane = pln
        self.env: Environment = env
        from ICARUS.Database import DB3D

        self.dynamics_directory: str = os.path.join(DB3D, pln.CASEDIR, "Dynamics")

        # Get Airplane Properties And State Variables
        self.mean_aerodynamic_chord: float = pln.mean_aerodynamic_chord
        self.S: float = pln.S
        self.dynamic_pressure: float = (
            0.5 * env.air_density * 20**2
        )  # VELOCITY IS ARBITRARY BECAUSE WE DO NOT KNOW THE TRIM YET
        self.inertia: FloatArray = pln.total_inertia
        self.mass: float = pln.M

        # GET TRIM STATE
        self.polars: DataFrame = self.format_polars(forces)
        self.trim: dict[str, float] = trim_state(self)
        self.dynamic_pressure = 0.5 * env.air_density * self.trim["U"] ** 2  # NOW WE UPDATE IT

        # Initialize Disturbances For Dynamic Analysis and Sensitivity Analysis
        self.disturbances: list[dst] = []
        self.pertrubation_results: DataFrame = DataFrame()
        self.sensitivity = Struct()
        self.sensitivity_results = Struct()

        # Initialize The Longitudal State Space Matrices
        self.longitudal = Struct()
        self.longitudal.stateSpace = Struct()
        self.longitudal.stateSpace.A = np.empty((4, 4), dtype=float)
        self.longitudal.stateSpace.A_DS = np.empty((4, 4), dtype=float)
        self.longitudal.stateSpace.B = np.empty((4, 1), dtype=float)
        self.longitudal.stateSpace.B_DS = np.empty((4, 1), dtype=float)

        # Initialize The Longitudal Eigenvalues And Eigenvectors
        self.longitudal.eigenValues = np.empty((4,), dtype=float)
        self.longitudal.eigenVectors = np.empty((4, 4), dtype=float)

        # Initialize The Lateral State Space Matrices
        self.lateral = Struct()
        self.lateral.stateSpace = Struct()
        self.lateral.stateSpace.A = np.empty((4, 4), dtype=float)
        self.lateral.stateSpace.A_DS = np.empty((4, 4), dtype=float)
        self.lateral.stateSpace.B = np.empty((4, 1), dtype=float)
        self.lateral.stateSpace.B_DS = np.empty((4, 1), dtype=float)

        # Initialize The Lateral Eigenvalues And Eigenvectors
        self.lateral.eigenValues = np.empty((4,), dtype=float)
        self.lateral.eigenVectors = np.empty((4, 4), dtype=float)

    def eigenvalue_analysis(self) -> None:
        # Compute Eigenvalues and Eigenvectors
        eigvalLong, eigvecLong = np.linalg.eig(self.a_long)
        self.longitudal.eigenValues = eigvalLong
        self.longitudal.eigenVectors = eigvecLong

        eigvalLat, eigvecLat = np.linalg.eig(self.a_lat)
        self.lateral.eigenValues = eigvalLat
        self.lateral.eigenVectors = eigvecLat

    def format_polars(self, forces: DataFrame) -> DataFrame:
        forces_rotated: DataFrame = rotate_forces(forces, forces["AoA"])
        return self.make_aero_coefficients(forces_rotated)

    def make_aero_coefficients(self, forces: DataFrame) -> DataFrame:
        Data: DataFrame = DataFrame()
        S: float = self.S
        MAC: float = self.mean_aerodynamic_chord
        dynamic_pressure: float = self.dynamic_pressure

        Data["CL"] = forces["Fz"] / (dynamic_pressure * S)
        Data["CD"] = forces["Fx"] / (dynamic_pressure * S)
        Data["Cm"] = forces["M"] / (dynamic_pressure * S * MAC)
        Data["Cn"] = forces["N"] / (dynamic_pressure * S * MAC)
        Data["Cl"] = forces["L"] / (dynamic_pressure * S * MAC)
        Data["AoA"] = forces["AoA"]
        return Data

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

    def stability_fd(self, polar_name: str = "2D") -> None:
        X, Z, M = longitudal_stability(self, polar_name)
        Y, L, N = lateral_stability(self, polar_name)
        self.SBderivativesDS = StabilityDerivativesDS(X, Y, Z, L, M, N)

    def plot_eigenvalues(self) -> None:
        """
        Generate a plot of the eigenvalues.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # extract real part
        x: list[float] = [ele.real for ele in self.longitudal.eigenValues]
        # extract imaginary part
        y: list[float] = [ele.imag for ele in self.longitudal.eigenValues]
        ax.scatter(x, y, color="r")

        # extract real part
        x = [ele.real for ele in self.lateral.eigenValues]
        # extract imaginary part
        y = [ele.imag for ele in self.lateral.eigenValues]
        marker_x = MarkerStyle("x")
        ax.scatter(x, y, color="b", marker=marker_x)

        ax.set_ylabel("Imaginary")
        ax.set_xlabel("Real")
        ax.grid()
        fig.show()

    def __str__(self) -> str:
        ss = io.StringIO()
        ss.write(f"State: {self.name}\n")
        ss.write(f"Trim: {self.trim}\n")
        ss.write(f"\n{45*'--'}\n")

        ss.write("\nLongitudal State:\n")
        ss.write(
            f"Eigen Values: {[round(item,3) for item in self.longitudal.eigenValues]}\n",
        )
        ss.write("Eigen Vectors:\n")
        for item in self.longitudal.eigenVectors:
            ss.write(f"\t{[round(i,3) for i in item]}\n")
        ss.write("\nThe State Space Matrix:\n")
        ss.write(
            tabulate(self.longitudal.stateSpace.A, tablefmt="github", floatfmt=".3f"),
        )

        ss.write(f"\n\n{45*'--'}\n")

        ss.write("\nLateral State:\n")
        ss.write(
            f"Eigen Values: {[round(item,3) for item in self.lateral.eigenValues]}\n",
        )
        ss.write("Eigen Vectors:\n")
        for item in self.lateral.eigenVectors:
            ss.write(f"\t{[round(i,3) for i in item]}\n")
        ss.write("\nThe State Space Matrix:\n")
        ss.write(tabulate(self.lateral.stateSpace.A, tablefmt="github", floatfmt=".3f"))
        return ss.getvalue()

    def to_json(self) -> str:
        """
        Pickle the state object to a json string.

        Returns:
            str: Json String
        """
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def save(self) -> None:
        """
        Save the state object to a json file.
        """
        fname: str = os.path.join(self.dynamics_directory, f"{self.name}.json")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @property
    def a_long(self) -> Any:
        return self.longitudal.stateSpace.A

    @a_long.setter
    def a_long(self, value: FloatArray) -> None:
        self.longitudal.stateSpace.A = value

    @property
    def astar_long(self) -> Any:
        return self.longitudal.stateSpace.A_DS

    @astar_long.setter
    def astar_long(self, value: FloatArray) -> None:
        self.longitudal.stateSpace.A_DS = value

    @property
    def a_lat(self) -> Any:
        return self.lateral.stateSpace.A

    @a_lat.setter
    def a_lat(self, value: FloatArray) -> None:
        self.lateral.stateSpace.A = value

    @property
    def astar_lat(self) -> Any:
        return self.lateral.stateSpace.A_DS

    @astar_lat.setter
    def astar_lat(self, value: FloatArray) -> None:
        self.lateral.stateSpace.A_DS = value
