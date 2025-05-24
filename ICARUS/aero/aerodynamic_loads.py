from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
from typing import List
from typing import Optional

if TYPE_CHECKING:
    from jax import Array
    from pandas import DataFrame
    from ICARUS.aero import LSPT_Plane
    from ICARUS.flight_dynamics import State

from .strip_loads import StripLoads


class AerodynamicLoads:
    """Class to handle a collection of StripData objects and provide methods
    to iterate over them and calculate total aerodynamic loads.
    """

    def __init__(self, strips: list[StripLoads] | None = None):
        """Initialize AerodynamicLoads with a list of StripData objects.

        Args:
            strips: List of StripData objects. If None, initializes empty list.
        """
        self.strips: list[StripLoads] = strips if strips is not None else []

    def add_strip(self, strip: StripLoads) -> None:
        """Add a StripData object to the collection.

        Args:
            strip: StripData object to add
        """
        self.strips.append(strip)

    def remove_strip(self, index: int) -> None:
        """Remove a StripData object at the specified index.

        Args:
            index: Index of the strip to remove (supports negative indexing)
        """
        if -len(self.strips) <= index < len(self.strips):
            del self.strips[index]
        else:
            raise IndexError(f"Strip index {index} out of range")

    def clear(self) -> None:
        """Clear all strips from the collection."""
        self.strips.clear()

    def __len__(self) -> int:
        """Return the number of strips in the collection."""
        return len(self.strips)

    def __getitem__(self, index: int) -> StripLoads:
        """Get a strip by index."""
        return self.strips[index]

    def __setitem__(self, index: int, strip: StripLoads) -> None:
        """Set a strip at the specified index."""
        self.strips[index] = strip

    def __iter__(self) -> Iterator[StripLoads]:
        """Iterate over the strips."""
        return iter(self.strips)

    def calc_total_lift(self) -> float:
        """Calculate total lift across all strips.

        Returns:
            Total lift force
        """
        total_lift = 0.0
        for strip in self.strips:
            total_lift += strip.get_total_lift()
        return total_lift

    def calc_total_drag(self) -> float:
        """Calculate total drag across all strips.

        Returns:
            Total drag force
        """
        total_drag = 0.0
        for strip in self.strips:
            total_drag += strip.get_total_drag()
        return total_drag

    def calc_total_moments(self) -> tuple[float, float, float]:
        """Calculate total moments across all strips.

        Returns:
            Tuple of (Mx, My, Mz) total moments
        """
        total_mx = 0.0
        total_my = 0.0
        total_mz = 0.0

        for strip in self.strips:
            mx, my, mz = strip.get_total_moment()
            total_mx += mx
            total_my += my
            total_mz += mz

        return total_mx, total_my, total_mz

    def calc_trefftz_drag(self) -> float:
        """Calculate total Trefftz plane drag across all strips.

        Returns:
            Total Trefftz drag
        """
        total_trefftz_drag = 0.0
        for strip in self.strips:
            total_trefftz_drag += strip.D_trefftz
        return total_trefftz_drag

    def calc_2d_loads(self) -> tuple[float, float, float]:
        """Calculate total 2D loads across all strips.

        Returns:
            Tuple of (L_2D, D_2D, My_2D) total 2D loads
        """
        total_l_2d = 0.0
        total_d_2d = 0.0
        total_my_2d = 0.0

        for strip in self.strips:
            total_l_2d += strip.L_2D
            total_d_2d += strip.D_2D
            total_my_2d += strip.My_2D

        return total_l_2d, total_d_2d, total_my_2d

    def get_strip_by_airfoil(self, airfoil_name: str) -> list[StripLoads]:
        """Get all strips that use a specific airfoil.

        Args:
            airfoil_name: Name of the airfoil to search for

        Returns:
            List of StripData objects that use the specified airfoil
        """
        matching_strips = []
        for strip in self.strips:
            if strip.airfoil is not None and strip.airfoil.name == airfoil_name:
                matching_strips.append(strip)
        return matching_strips

    def calc_mean_values_all(self) -> None:
        """Calculate mean values for all strips in the collection."""
        for strip in self.strips:
            strip.calc_mean_values()

    def calc_aerodynamic_loads_all(self, dynamic_pressure: float) -> None:
        """Calculate aerodynamic loads for all strips in the collection.

        Args:
            dynamic_pressure: Dynamic pressure (0.5 * rho * V^2)
        """
        for strip in self.strips:
            strip.calc_aerodynamic_loads(dynamic_pressure)

    def get_summary(self) -> dict:
        """Get a summary of the aerodynamic loads.

        Returns:
            Dictionary containing summary statistics
        """
        if not self.strips:
            return {
                "num_strips": 0,
                "total_lift": 0.0,
                "total_drag": 0.0,
                "total_moments": (0.0, 0.0, 0.0),
                "total_trefftz_drag": 0.0,
                "total_2d_loads": (0.0, 0.0, 0.0),
            }

        return {
            "num_strips": len(self.strips),
            "total_lift": self.calc_total_lift(),
            "total_drag": self.calc_total_drag(),
            "total_moments": self.calc_total_moments(),
            "total_trefftz_drag": self.calc_trefftz_drag(),
            "total_2d_loads": self.calc_2d_loads(),
        }

    @classmethod
    def from_lspt_plane(cls, plane: LSPT_Plane) -> AerodynamicLoads:
        """Create an AerodynamicLoads object from an LSPT_Plane.

        Args:
            plane: LSPT_Plane object containing strip data

        Returns:
            AerodynamicLoads object with strips from the plane
        """
        # Create a new AerodynamicLoads object with the strips from the plane
        return cls(strips=list(plane.strip_data))

    def distribute_gamma_calculations(self, gammas: Array, w_induced: Array) -> None:
        """Distribute gamma and w_induced calculations to individual strips.

        Args:
            gammas: JAX array of circulation values from VLM solution
            w_induced: JAX array of induced downwash values
        """
        for strip in self.strips:
            if hasattr(strip, "panel_idxs"):
                strip_idxs = strip.panel_idxs
                strip.gammas = gammas[strip_idxs]
                strip.w_induced = w_induced[strip_idxs]
                strip.calc_mean_values()

    def calculate_potential_loads(
        self, plane: LSPT_Plane, state: State, gammas: Array, w_induced: Array,
    ) -> tuple[float, float, float]:
        """Calculate potential loads using VLM results.

        Args:
            plane: LSPT_Plane object containing geometry
            state: Flight state containing environment data
            gammas: JAX array of circulation values
            w_induced: JAX array of induced downwash values

        Returns:
            Tuple of (total_lift, total_drag, total_moment_y)
        """
        import jax.numpy as jnp
        import numpy as np

        dens = state.environment.air_density
        umag = state.u_freestream

        # Initialize panel loads
        L_pan = np.zeros(plane.NM)
        D_pan = np.zeros(plane.NM)

        total_lift = 0.0
        total_drag = 0.0
        total_moment_y = 0.0

        # Calculate loads for each strip
        for strip in self.strips:
            if hasattr(strip, "panel_idxs"):
                g_strips = gammas[strip.panel_idxs]
                w = w_induced[strip.panel_idxs]

                # Calculate panel loads within the strip
                for j in range(strip.num_panels - 1):
                    if j == 0:
                        g = g_strips[j]
                    else:
                        g = g_strips[j] - g_strips[j - 1]

                    L_pan[strip.panel_idxs[j]] = dens * umag * strip.width * g
                    D_pan[strip.panel_idxs[j]] = -dens * strip.width * g * w[j]

                # Update strip loads
                strip.L_potential = float(jnp.sum(L_pan[strip.panel_idxs]))
                strip.D_potential = float(jnp.sum(D_pan[strip.panel_idxs]))

                total_lift += strip.L_potential
                total_drag += strip.D_potential

        # Apply symmetry factor for symmetric wings (factor of 2)
        # This matches the behavior in LSPT_Plane.aseq method
        total_lift *= 2
        total_drag *= 2
        total_moment_y *= 2

        return total_lift, total_drag, total_moment_y

    def calculate_viscous_loads(self, dynamic_pressure: float) -> tuple[float, float, float]:
        """Calculate viscous loads for all strips.

        Args:
            dynamic_pressure: Dynamic pressure (0.5 * rho * V^2)

        Returns:
            Tuple of (total_viscous_lift, total_viscous_drag, total_viscous_moment)
        """
        total_viscous_lift = 0.0
        total_viscous_drag = 0.0
        total_viscous_moment = 0.0

        for strip in self.strips:
            # Calculate 2D aerodynamic loads for the strip
            if hasattr(strip, "airfoil") and strip.airfoil is not None:
                # Use airfoil data to calculate viscous loads
                # This would typically involve interpolating airfoil polars
                # For now, we'll calculate based on existing strip data
                strip.calc_aerodynamic_loads(dynamic_pressure)

                # Add viscous components (2D loads)
                total_viscous_lift += strip.L_2D
                total_viscous_drag += strip.D_2D
                total_viscous_moment += strip.My_2D

        return total_viscous_lift, total_viscous_drag, total_viscous_moment

    def run_vlm_analysis(self, plane: LSPT_Plane, state: State, angles: list[float] | Array) -> DataFrame:
        """Run complete VLM analysis workflow integrating all components.

        This method implements the complete workflow:
        1. Factorize VLM matrices
        2. Calculate gammas and w_induced for each angle
        3. Distribute calculations to strips
        4. Calculate potential loads
        5. Calculate viscous loads

        Args:
            plane: LSPT_Plane object containing geometry
            state: Flight state containing environment data
            angles: List or array of angles of attack in degrees
              Returns:
            DataFrame containing analysis results
        """
        import jax
        import jax.numpy as jnp
        import pandas as pd

        from ICARUS.aero.vlm import get_RHS

        # Step 1: Factorize VLM system matrices
        A_LU, A_piv, A_star = plane.factorize_system()
        umag = state.u_freestream
        dynamic_pressure = 0.5 * state.environment.air_density * umag**2

        # Initialize result arrays
        results = {
            "AoA": [],
            "Total_Lift_Potential": [],
            "Total_Drag_Potential": [],
            "Total_Lift_Viscous": [],
            "Total_Drag_Viscous": [],
            "Total_Lift_Combined": [],
            "Total_Drag_Combined": [],
            "CL": [],
            "CD": [],
        }

        for angle in angles:
            plane.alpha = angle * jnp.pi / 180
            plane.beta = 0

            # Step 2: Calculate freestream velocity components and RHS
            Uinf = umag * jnp.cos(plane.alpha) * jnp.cos(plane.beta)
            Vinf = umag * jnp.cos(plane.alpha) * jnp.sin(plane.beta)
            Winf = umag * jnp.sin(plane.alpha) * jnp.cos(plane.beta)

            Q = jnp.array((Uinf, Vinf, Winf))
            RHS = get_RHS(plane, Q)

            # Solve for circulations using factorized system
            gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
            w_induced = jnp.matmul(A_star, gammas)

            # Step 3: Distribute gamma calculations to strips
            self.distribute_gamma_calculations(gammas, w_induced)

            # Step 4: Calculate potential loads
            total_lift_potential, total_drag_potential, total_moment_potential = self.calculate_potential_loads(
                plane, state, gammas, w_induced,
            )

            # Step 5: Calculate viscous loads
            total_lift_viscous, total_drag_viscous, total_moment_viscous = self.calculate_viscous_loads(
                dynamic_pressure,
            )

            # Step 6: Combine loads
            total_lift_combined = total_lift_potential + total_lift_viscous
            total_drag_combined = total_drag_potential + total_drag_viscous

            # Calculate coefficients
            CL = total_lift_combined / (dynamic_pressure * plane.S)
            CD = total_drag_combined / (dynamic_pressure * plane.S)

            # Store results
            results["AoA"].append(float(angle))
            results["Total_Lift_Potential"].append(float(total_lift_potential))
            results["Total_Drag_Potential"].append(float(total_drag_potential))
            results["Total_Lift_Viscous"].append(float(total_lift_viscous))
            results["Total_Drag_Viscous"].append(float(total_drag_viscous))
            results["Total_Lift_Combined"].append(float(total_lift_combined))
            results["Total_Drag_Combined"].append(float(total_drag_combined))
            results["CL"].append(float(CL))
            results["CD"].append(float(CD))

        return pd.DataFrame(results)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AerodynamicLoads(strips={len(self.strips)})"
