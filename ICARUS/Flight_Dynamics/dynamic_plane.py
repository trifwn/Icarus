# from typing import Any
# import pandas as pd
# from numpy import dtype
# from numpy import floating
# from numpy import ndarray
# from pandas import DataFrame
# from pandas import Series
# from .disturbances import Disturbance as dst
# from .pertrubations import lateral_pertrubations
# from .pertrubations import longitudal_pertrubations
# from .Stability.lateralFD import lateral_stability
# from .Stability.longitudalFD import longitudal_stability
# from .trim import trim_state
# from ICARUS.Core.struct import Struct
# from ICARUS.Software.GenuVP3.postProcess.forces import rotate_forces
# from ICARUS.Vehicle.plane import Airplane
# class Dynamic_Airplane(Airplane):
#     """Class for the dynamic analysis of an airplane.
#     The airplane is assumed to be of the airplane class.
#     Inputs:
#     - pln: Airplane class
#     - polars3D: DataFrame with the polars of the airplane
#     """
#     def __init__(self, pln: Airplane, forces_3d: DataFrame) -> None:
#         """Initialize the dynamic airplane"""
#         self.__dict__.update(pln.__dict__)
#         self.name: str = f"dyn_{pln.name}"
#         self.polars_3d: DataFrame = self.format_polars(forces_3d)
#         # Compute Trim State
#         self.trim: dict[str, float] = trim_state(self)
#         self.define_dynamic_pressure(self.dens, self.trim["U"])
#         # Initialize Disturbances For Dynamic Analysis and Sensitivity Analysis
#         self.disturbances: list[dst] = []
#         self.sensitivity: dict[str, list[dst]] = {}
#         self.epsilons: dict[str, float] = {}
#         self.scheme: str = ""
#         # Initialize Sensitivity Analysis
#         self.pertrubation_results: DataFrame = DataFrame()
#         self.sensitivity_results: dict[str, DataFrame] = {}
#     def get_polars_3d(self) -> DataFrame:
#         """
#         Returns the polars of the airplane.
#         Returns:
#             DataFrame: _description_
#         """
#         return self.polars_3d
#     def change_polars_3d(self, polars_3d: DataFrame) -> None:
#         """
#         Change the polars of the airplane and computes new trim.
#         Args:
#             polars_3d (DataFrame): New Polars
#         """
#         self.polars_3d = polars_3d
#         self.trim = trim_state(self)
#     def format_polars(self, raw_forces: DataFrame) -> DataFrame:
#         """
#         Formats the polars of the airplane to be used in the dynamic analysis.
#         This means rotating the forces and adding the aero coefficients.
#         Args:
#             raw_forces (DataFrame): DataFrame with the raw forces from AoA analysis
#         Returns:
#             DataFrame: Polars of the airplane. Columns are: CL, CD, Cm, Cn, Cl, AoA
#         """
#         aoa: Series[float] = raw_forces["AoA"]
#         forces: DataFrame = rotate_forces(raw_forces, aoa)
#         return self.make_aero_coefficients(forces)
#     def make_aero_coefficients(self, forces: DataFrame) -> DataFrame:
#         """
#         Calculates the dimensionless aero coefficients from the forces and airplane characteristics.
#         Args:
#             forces (DataFrame): Dataframe containing the forces from the AoA analysis
#         Returns:
#             DataFrame: Dataframe containing the aero coefficients. Columns are: CL, CD, Cm, Cn, Cl, AoA
#         """
#         coeffs_df = pd.DataFrame()
#         coeffs_df["CL"] = forces["Fz"] / (self.dynamic_pressure * self.S)
#         coeffs_df["CD"] = forces["Fx"] / (self.dynamic_pressure * self.S)
#         coeffs_df["Cm"] = forces["M"] / (
#             self.dynamic_pressure * self.S * self.mean_aerodynamic_chord
#         )
#         coeffs_df["Cn"] = forces["N"] / (
#             self.dynamic_pressure * self.S * self.mean_aerodynamic_chord
#         )
#         coeffs_df["Cl"] = forces["L"] / (
#             self.dynamic_pressure * self.S * self.mean_aerodynamic_chord
#         )
#         coeffs_df["AoA"] = forces["AoA"]
#         return coeffs_df
#     def add_all_pertrubations(
#         self,
#         scheme: str,
#         epsilon: dict[str, float] | None = None,
#     ) -> None:
#         """
#         Function to add all perturbations to the airplane needed for dynamic analysis
#         Args:
#             scheme (str): Difference scheme to use for the perturbations
#             epsilon (dict[str,float] | None, optional): Dict of values to be used as amplitudes. Defaults to None.
#         """
#         self.scheme = scheme
#         self.disturbances = [
#             *longitudal_pertrubations(self, scheme, epsilon),
#             *lateral_pertrubations(self, scheme, epsilon),
#         ]
#         self.disturbances.append(dst(None, 0))
#     def sensitivity_analysis(
#         self,
#         var: str,
#         space: list[float] | ndarray[Any, dtype[floating[Any]]],
#     ) -> None:
#         """
#         Creates a sensitivity analysis for a variable. The variable is perturbed in the given space and
#         results are stored. That way we can see where the perturbation is linear and where it is not.
#         Args:
#             var (str): Name of the variable to be perturbed
#             space (list[float] | ndarray[Any, dtype[floating[Any]]]): List of values to be used as perturbation amplitudes
#         """
#         self.sensitivity[var] = []
#         for e in space:
#             self.sensitivity[var].append(dst(var, e))
#     def get_pertrub(self) -> None:
#         """
#         Print Pertruabtion Results
#         """
#         for disturbance in self.disturbances:
#             print(disturbance)
#     def stability_fd(self, scheme: str = "Central") -> None:
#         """
#         Calculates and stores stability derivatives of Dynamic Analysis using Finite Differences
#         Args:
#             scheme (str, optional): Difference Scheme to be used in the dynamic analysis. Defaults to "Central".
#         """
#         self.scheme = scheme
#         X, Z, M = longitudal_stability(self, "2D")
#         Y, L, N = lateral_stability(self, "Potential")
#         self.stability_derivatives_ds = StabilityDerivativesDS(X, Y, Z, L, M, N)
#     def __str__(self) -> str:
#         string: str = f"Dynamic AirPlane {self.name}"
#         # str += f"\nTrimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
#         # str += f"Surfaces:\n"
#         # for surfaces in self.surfaces:
#         #     str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
#         return string
from ICARUS.Core.struct import Struct


class StabilityDerivativesDS(Struct):
    """
    Class to store stability derivatives of Dynamic Analysis.
    """

    def __init__(
        self,
        X: dict[str, float],
        Y: dict[str, float],
        Z: dict[str, float],
        L: dict[str, float],
        M: dict[str, float],
        N: dict[str, float],
    ) -> None:
        """
        Initialize Stability Derivatives
        Args:
            X (dict[str, float]): Derivatives based on X
            Y (dict[str, float]): Derivatives based on Y
            Z (dict[str, float]): Derivatives based on Z
            L (dict[str, float]): Derivatives based on L
            M (dict[str, float]): Derivatives based on M
            N (dict[str, float]): Derivatives based on N
        """
        self.X: dict[str, float] = X
        self.Y: dict[str, float] = Y
        self.Z: dict[str, float] = Z
        self.L: dict[str, float] = L
        self.M: dict[str, float] = M
        self.N: dict[str, float] = N

    def __str__(self) -> str:
        """
        String Representation of Stability Derivatives
        Returns:
            str: String Representation of Stability Derivatives
        """
        string = "Dimensional Stability Derivatives:\n"
        string += "\nLongitudal Derivatives\n"
        string += f"Xu=\t{self.X['u']}\n"
        string += f"Xw=\t{self.X['w']}\n"
        string += f"Zu=\t{self.Z['u']}\n"
        string += f"Zw=\t{self.Z['w']}\n"
        string += f"Zq=\t{self.Z['q']}\n"
        string += f"Mu=\t{self.M['u']}\n"
        string += f"Mw=\t{self.M['w']}\n"
        string += f"Mq=\t{self.M['q']}\n"
        string += "\nLateral Derivatives\n"
        string += f"Yv=\t{self.Y['v']}\n"
        string += f"Yp=\t{self.Y['p']}\n"
        string += f"Yr=\t{self.Y['r']}\n"
        string += f"Lv=\t{self.L['v']}\n"
        string += f"Lp=\t{self.L['p']}\n"
        string += f"Lr=\t{self.L['r']}\n"
        string += f"Nv=\t{self.N['v']}\n"
        string += f"Np=\t{self.N['p']}\n"
        string += f"Nr=\t{self.N['r']}"
        return string
