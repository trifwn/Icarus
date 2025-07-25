"""
Aerodynamic Results Module

This module defines the AerodynamicResults class that encapsulates the results
from aerodynamic analysis including forces, moments, coefficients, and derivatives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ICARUS.core.types import FloatArray

from .aerodynamic_loads import AerodynamicLoads
from .aerodynamic_state import AerodynamicState

if TYPE_CHECKING:
    from ICARUS.vehicle.airplane import Airplane

    from .lspt_plane import LSPT_Plane


class AerodynamicResults:
    """
    Class to store and analyze aerodynamic simulation results from VLM analysis.

    This class provides comprehensive storage and analysis capabilities for
    aerodynamic simulation results including forces, moments, coefficients,
    and stability derivatives. It integrates with LSPT_Plane for reference
    geometry and the VLM analysis workflow.

    Attributes:
        plane (LSPT_Plane): Reference plane object for geometry data
        states (list[AerodynamicState]): list of aerodynamic states analyzed
        loads (list[AerodynamicLoads]): Corresponding aerodynamic loads
        reference_area (float): Reference wing area in m²
        reference_chord (float): Reference chord length in m
        reference_span (float): Reference span in m
    """

    def __init__(
        self,
        plane: LSPT_Plane | Airplane,
    ) -> None:
        """
        Initialize AerodynamicResults from LSPT_Plane object.

        Args:
            plane: LSPT_Plane object containing geometry and reference data
        """
        # Calculate reference measures
        self.plane_name = plane.name if hasattr(plane, "name") else "Unknown Plane"
        self.reference_area = plane.S
        self.reference_chord = plane.MAC
        self.reference_span = plane.span

        self.states: list[AerodynamicState] = []
        self.loads: list[AerodynamicLoads] = []

    def add_result(self, state: AerodynamicState, loads: AerodynamicLoads) -> None:
        """
        Add a simulation result.

        Args:
            state: Aerodynamic state for this result
            loads: Corresponding aerodynamic loads
        """
        self.states.append(state)
        self.loads.append(loads)

    def add_results_from_vlm_analysis(
        self,
        states: list[AerodynamicState],
        loads_list: list[AerodynamicLoads],
    ) -> None:
        """
        Add multiple results from VLM analysis.

        Args:
            states: list of aerodynamic states
            loads_list: list of corresponding aerodynamic loads
        """
        if len(states) != len(loads_list):
            raise ValueError("Number of states must match number of loads")

        for state, loads in zip(states, loads_list):
            self.add_result(state, loads)

    def clear(self) -> None:
        """Clear all stored results."""
        self.states.clear()
        self.loads.clear()

    @property
    def num_results(self) -> int:
        """Number of stored results."""
        return len(self.states)

    def get_forces_and_moments(
        self,
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> tuple[FloatArray, FloatArray]:
        """
        Extract forces and moments from all results.

        Args:
            calculation: Type of calculation ('potential' or 'viscous')

        Returns:
            tuple of (forces, moments) arrays with shape (n_results, 3)
        """
        forces = np.zeros((self.num_results, 3))
        moments = np.zeros((self.num_results, 3))

        for i, loads in enumerate(self.loads):
            lift = loads.calc_total_lift(calculation)
            drag = loads.calc_total_drag(calculation)
            mx, my, mz = loads.calc_total_moments(calculation)

            # Force convention: [Fx, Fy, Fz] = [Drag, Side, -Lift]
            forces[i] = [drag, 0.0, -lift]
            moments[i] = [mx, my, mz]

        return forces, moments

    def calculate_coefficients(
        self,
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> dict[str, FloatArray]:
        """
        Calculate aerodynamic coefficients.

        Args:
            calculation: Type of calculation ('potential' or 'viscous')

        Returns:
            Dictionary containing coefficient arrays
        """
        forces, moments = self.get_forces_and_moments(calculation)

        # Calculate dynamic pressures
        dynamic_pressures = np.array([state.dynamic_pressure for state in self.states])

        # Force coefficients
        CX = forces[:, 0] / (
            dynamic_pressures * self.reference_area
        )  # Drag coefficient
        CY = forces[:, 1] / (
            dynamic_pressures * self.reference_area
        )  # Side force coefficient
        CZ = forces[:, 2] / (
            dynamic_pressures * self.reference_area
        )  # Negative lift coefficient

        # Moment coefficients
        Cl = (
            moments[:, 0]
            / (dynamic_pressures * self.reference_area * self.reference_span)
            if self.reference_span
            else np.zeros_like(CX)
        )
        Cm = moments[:, 1] / (
            dynamic_pressures * self.reference_area * self.reference_chord
        )
        Cn = (
            moments[:, 2]
            / (dynamic_pressures * self.reference_area * self.reference_span)
            if self.reference_span
            else np.zeros_like(CX)
        )

        # Standard coefficients
        CL = -CZ  # Lift coefficient (positive upward)
        CD = CX  # Drag coefficient

        return {
            "CL": CL,
            "CD": CD,
            "CX": CX,
            "CY": CY,
            "CZ": CZ,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn,
        }

    def calculate_derivatives(self, variable: str = "alpha") -> dict[str, float]:
        """
        Calculate stability derivatives with respect to a variable.

        Args:
            variable: Variable to calculate derivatives with respect to

        Returns:
            Dictionary of stability derivatives
        """
        if self.num_results < 2:
            raise ValueError("Need at least 2 results to calculate derivatives")

        coefficients = self.calculate_coefficients()

        if variable == "alpha":
            x_vals = np.array([state.alpha_rad for state in self.states])
        elif variable == "beta":
            x_vals = np.array([state.beta_rad for state in self.states])
        else:
            raise ValueError(f"Derivative calculation for '{variable}' not implemented")

        derivatives = {}
        for coeff_name, coeff_vals in coefficients.items():
            # Use central difference for interior points, forward/backward for edges
            grad = np.gradient(coeff_vals, x_vals)
            derivatives[f"{coeff_name}_{variable}"] = grad
        return derivatives

    def to_dataframe(
        self,
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame compatible with polars interface.

        Args:
            calculation: Type of calculation ('potential' or 'viscous')

        Returns:
            DataFrame containing all results and calculated coefficients
        """
        if self.num_results == 0:
            return pd.DataFrame()

        # Extract state data
        data: dict[str, FloatArray | list[float]] = {}
        data["AoA"] = [state.alpha for state in self.states]

        # Extract forces and moments with LSPT naming convention
        forces, moments = self.get_forces_and_moments(calculation)
        data[f"LSPT {calculation.title()} Fx"] = forces[:, 0]  # Drag
        data[f"LSPT {calculation.title()} Fy"] = forces[:, 1]  # Side force
        data[f"LSPT {calculation.title()} Fz"] = -forces[:, 2]  # Lift (positive up)
        data[f"LSPT {calculation.title()} Mx"] = moments[:, 0]
        data[f"LSPT {calculation.title()} My"] = moments[:, 1]
        data[f"LSPT {calculation.title()} Mz"] = moments[:, 2]

        data["alpha_deg"] = [state.alpha for state in self.states]
        data["beta_deg"] = [state.beta for state in self.states]
        data["airspeed"] = [state.airspeed for state in self.states]
        data["density"] = [state.density for state in self.states]
        data["dynamic_pressure"] = [state.dynamic_pressure for state in self.states]

        # Conver the keys to numpy arrays for consistency
        for key in data:
            if isinstance(data[key], list):
                data[key] = np.array(data[key])

        # Calculate coefficients
        coefficients = self.calculate_coefficients(calculation)
        data.update(coefficients)
        return pd.DataFrame(data)

    def to_polars_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame format expected by polars interface.

        Returns:
            DataFrame in polars format with both potential and viscous data
        """
        df_potential = self.to_dataframe("potential")
        df_viscous = self.to_dataframe("viscous")

        # Merge potential and viscous results
        merged_data = df_potential.copy()

        # Add viscous columns
        viscous_cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        for col in viscous_cols:
            viscous_col = f"LSPT Viscous {col}"
            if viscous_col in df_viscous.columns:
                merged_data[viscous_col] = df_viscous[viscous_col]
            else:
                merged_data[viscous_col] = (
                    0.0  # Default to zero if viscous not calculated
                )
        return merged_data

    def plot_polar(
        self,
        x_coeff: str = "CD",
        y_coeff: str = "CL",
        calculation: Literal["potential", "viscous"] = "potential",
        ax: Axes | None = None,
        title: str | None = None,
    ) -> None:
        """
        Plot aerodynamic polar (e.g., CL vs CD).

        Args:
            x_coeff: Coefficient for x-axis
            y_coeff: Coefficient for y-axis
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        df = self.to_dataframe(calculation=calculation)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_fig = True
        else:
            fig = ax.get_figure()
            show_fig = False

        if not isinstance(fig, Figure):
            raise TypeError("Expected a matplotlib Figure object")

        ax.plot(df[x_coeff], df[y_coeff], "bo-", linewidth=2, markersize=6)
        ax.set_xlabel(x_coeff)
        ax.set_ylabel(y_coeff)
        ax.set_title(f"{y_coeff} vs {x_coeff} Polar")
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)

        if show_fig:
            fig.tight_layout()
            fig.show()

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for the results.

        Returns:
            Dictionary containing summary statistics
        """
        if self.num_results == 0:
            return {}

        df = self.to_dataframe()

        summary = {
            "num_cases": self.num_results,
            "alpha_range_deg": (df["alpha_deg"].min(), df["alpha_deg"].max()),
            "CL_range": (df["CL"].min(), df["CL"].max()),
            "CD_range": (df["CD"].min(), df["CD"].max()),
            "max_L_D": (df["CL"] / df["CD"]).max(),
            "alpha_max_L_D": df.loc[(df["CL"] / df["CD"]).idxmax(), "alpha_deg"],
        }

        # Add derivatives if enough points
        if self.num_results >= 2:
            try:
                derivatives = self.calculate_derivatives("alpha")
                summary["CL_alpha"] = np.mean(derivatives["CL_alpha"])
                summary["CD_alpha"] = np.mean(derivatives["CD_alpha"])
                summary["Cm_alpha"] = np.mean(derivatives["Cm_alpha"])
            except Exception:
                pass

        return summary

    def __len__(self) -> int:
        """Return number of results."""
        return self.num_results

    def __getitem__(self, index: int) -> tuple[AerodynamicState, AerodynamicLoads]:
        """Get result by index."""
        return self.states[index], self.loads[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"AerodynamicResults(plane={self.plane_name}, num_results={self.num_results}, S_ref={self.reference_area})"
