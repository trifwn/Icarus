from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator
import jax.numpy as jnp
import pandas as pd

from ICARUS.aero.aerodynamic_state import AerodynamicState


if TYPE_CHECKING:
    from jax import Array
    from ICARUS.aero import LSPT_Plane

from . import StripLoads


class AerodynamicLoads:
    """Class to handle a collection of StripData objects and provide methods
    to iterate over them and calculate total aerodynamic loads.
    """

    def __init__(self, plane: LSPT_Plane) -> None:
        """Initialize AerodynamicLoads with a list of StripData objects.

        Args:
            plane: LSPT_Plane object containing strip data
        """
        self.strips: list[StripLoads] = []

        surf_panel_index = 0
        for surf in plane.surfaces:
            spans = surf.span_positions
            chords = surf.chords
            # Get all the strips by their panel indices
            for i in range(surf.N - 1):
                strip_idxs = surf_panel_index + jnp.arange(i * (surf.M - 1), (i + 1) * (surf.M - 1))

                self.strips.append(
                    StripLoads(
                        panel_idxs=strip_idxs,
                        panels=plane.panels[strip_idxs, :, :],
                        chord=(chords[i] + chords[i + 1]) / 2,
                        width=spans[i + 1] - spans[i],
                    ),
                )

            surf_panel_index += surf.num_panels + surf.num_near_wake_panels + surf.num_flat_wake_panels

    def __len__(self) -> int:
        """Return the number of strips in the collection."""
        return len(self.strips)

    def __getitem__(self, index: int) -> StripLoads:
        """Get a strip by index."""
        return self.strips[index]

    def __iter__(self) -> Iterator[StripLoads]:
        """Iterate over the strips."""
        return iter(self.strips)

    def calc_total_lift(self, calculation="potential") -> float:
        """Calculate total lift across all strips.
        Args:
            calculation: Type of lift calculation ('potential' or 'viscous')
        Returns:
            Total lift force
        """
        total_lift = 0.0
        for strip in self.strips:
            total_lift += strip.get_total_lift(calculation=calculation)
        return total_lift

    def calc_total_drag(self, calculation="potential") -> float:
        """Calculate total drag across all strips.

        Returns:
            Total drag force
        """
        total_drag = 0.0
        for strip in self.strips:
            total_drag += strip.get_total_drag(calculation=calculation)
        return total_drag

    def calc_total_moments(self, calculation="potential") -> tuple[float, float, float]:
        """Calculate total moments across all strips.

        Returns:
            Tuple of (Mx, My, Mz) total moments
        """
        total_mx = 0.0
        total_my = 0.0
        total_mz = 0.0

        for strip in self.strips:
            mx, my, mz = strip.get_total_moments(calculation=calculation)
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

    def calc_aerodynamic_loads_all(
        self,
        density: float,
        umag: float,
    ) -> None:
        """Calculate aerodynamic loads for all strips in the collection.

        Args:
            density: Air density (kg/m^3)
            umag: Freestream velocity magnitude (m/s)
        """
        for strip in self.strips:
            strip.calc_aerodynamic_loads(density=density, umag=umag)

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

    def distribute_gamma_calculations(self, gammas: Array, w_induced: Array) -> None:
        """Distribute gamma and w_induced calculations to individual strips.

        Args:
            gammas: JAX array of circulation values from VLM solution
            w_induced: JAX array of induced downwash values
        """
        for strip in self.strips:
            strip_idxs = strip.panel_idxs
            strip.gammas = gammas[strip_idxs]
            strip.w_induced = w_induced[strip_idxs]

    def calculate_potential_loads(
        self,
        state: AerodynamicState,
    ) -> tuple[float, float, float]:
        """Calculate potential loads using VLM results.

        Args:
            state: Flight state containing environment data

        Returns:
            Tuple of (total_lift, total_drag, total_moment_y)
        """
        dens = state.density
        umag = state.airspeed

        # Initialize loads
        total_lift = 0.0
        total_drag = 0.0

        total_moment_x = 0.0
        total_moment_y = 0.0
        total_moment_z = 0.0

        # Calculate loads for each strip
        for strip in self.strips:
            strip.calc_aerodynamic_loads(density=dens, umag=umag)

            total_lift += strip.get_total_lift()
            total_drag += strip.get_total_drag()

            mx, my, mz = strip.get_total_moments()
            total_moment_x += mx
            total_moment_y += my
            total_moment_z += mz

        # Apply symmetry factor for symmetric wings (factor of 2)
        # This matches the behavior in LSPT_Plane.aseq method
        total_lift *= 2
        total_drag *= 2
        total_moment_y *= 2

        return total_lift, total_drag, total_moment_y

    def calculate_viscous_loads(
        self,
        state: AerodynamicState,
    ) -> tuple[float, float, float]:
        """Calculate viscous loads for all strips.

        Args:
            state: AerodynamicState containing environment data

        Returns:
            Tuple of (total_viscous_lift, total_viscous_drag, total_viscous_moment)
        """
        total_viscous_lift = 0.0
        total_viscous_drag = 0.0
        total_viscous_moment = 0.0

        for strip in self.strips:
            # Calculate 2D aerodynamic loads for the strip
            # Use airfoil data to calculate viscous loads
            # This would typically involve interpolating airfoil polars
            # For now, we'll calculate based on existing strip data
            # strip.calc_viscous_loads(dynamic_pressure)

            # Add viscous components (2D loads)
            total_viscous_lift += strip.L_2D
            total_viscous_drag += strip.D_2D
            total_viscous_moment += strip.My_2D

        return total_viscous_lift, total_viscous_drag, total_viscous_moment

    def to_dataframe(
        self,
        aerodynamic_state: AerodynamicState,
        plane: LSPT_Plane,
    ) -> pd.Series:
        """Convert aerodynamic loads to a dictionary format for easy access."""
        L_pot = self.calc_total_lift("potential")
        D_pot = self.calc_total_drag("potential")
        L_viscous = 0  # self.calc_total_lift("viscous")
        D_viscous = 0  # self.calc_total_drag("viscous")

        # Create a pandas Series to hold the aerodynamic loads
        loads = {
            "AoA": aerodynamic_state.alpha,
            "Lift_Potential": L_pot,
            "Drag_Potential": D_pot,
            "Lift_Viscous": L_viscous,
            "Drag_Viscous": D_viscous,
            "CL": L_pot / (aerodynamic_state.dynamic_pressure * plane.S),
            "CD": D_pot / (aerodynamic_state.dynamic_pressure * plane.S),
            "CL_2D": L_viscous / (aerodynamic_state.dynamic_pressure * plane.S),
            "CD_2D": D_viscous / (aerodynamic_state.dynamic_pressure * plane.S),
        }
        return pd.Series(
            data=loads, name=f"AerodynamicLoads_{aerodynamic_state.alpha:.2f}deg", index=list(loads.keys())
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AerodynamicLoads(strips={len(self.strips)})"
