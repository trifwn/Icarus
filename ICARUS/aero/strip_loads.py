from typing import TYPE_CHECKING
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil


class StripLoads:
    def __init__(
        self,
        panels: Float[Array, "..."],
        panel_idxs: Int[Array, "..."],
        chord: Float,
        width: Float,
        airfoil: Optional["Airfoil"] = None,
    ):
        self.chord = chord
        self.width = width
        self.panels = panels
        self.panel_idxs = panel_idxs
        self.num_panels = len(panel_idxs)
        self.airfoil = airfoil

        # Initialize the data arrays
        self.mean_panel_length = jnp.zeros(self.num_panels)
        self.mean_panel_width = jnp.zeros(self.num_panels)
        self.gammas = jnp.zeros(self.num_panels)
        self.w_induced = jnp.zeros(self.num_panels)

        # Initialize the data arrays
        self.panel_L = jnp.zeros(self.num_panels)
        self.panel_D = jnp.zeros(self.num_panels)
        self.D_trefftz = 0.0
        self.Mx = 0.0
        self.My = 0.0
        self.Mz = 0.0  # 2D interpolated polars
        self.L_2D = 0.0
        self.D_2D = 0.0
        self.My_2D = 0.0

    @property
    def mean_gamma(self) -> Float:
        """Mean circulation strength across all panels."""
        return jnp.mean(self.gammas)

    @property
    def mean_w_induced(self) -> Float:
        """Mean induced velocity across all panels."""
        return jnp.mean(self.w_induced)

    def calc_aerodynamic_loads(
        self,
        density: float,
        umag: float,
    ) -> None:
        """Calculate aerodynamic loads for this strip.

        Args:
            density: Air density (kg/m^3)
            umag: Freestream velocity magnitude (m/s)
        """

        gammas = self.gammas.copy() 
        gammas = gammas.at[1:].set(
            self.gammas[1:] - self.gammas[:-1]  # Calculate gamma differences
        ) 

        if jnp.any(jnp.isnan(gammas)):
            raise ValueError("NaN values found in gammas. Check gamma calculations.")

        # Calculate mean panel lengths and widths
        print("Calculating mean panel lengths and widths...")
        
        self.panel_L = density * umag *  gammas * self.mean_panel_width
        self.panel_D = density * self.w_induced * gammas *  self.mean_panel_width

        if jnp.any(jnp.isnan(self.panel_L)):
            raise ValueError("NaN values found in panel_L. Check gamma calculations.")

    def get_total_lift(self, calculation = "potential") -> float:
        """Get total lift for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_L))
        elif calculation == "viscous":
            # return float(jnp.sum(self.panel_L * self.mean_w_induced))
            raise NotImplementedError("Viscous lift calculation not implemented.")
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def get_total_drag(self, calculation = "potential") -> float:
        """Get total drag for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_D))
        elif calculation == "viscous":
            # return float(jnp.sum(self.panel_D * self.mean_w_induced))
            raise NotImplementedError("Viscous drag calculation not implemented.")
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")
        
    def get_total_moments(self, calculation = "potential") -> tuple[float, float, float]:
        """Get total moments for this strip.

        Args:
            calculation: Type of moment calculation ('potential' or 'viscous')

        Returns:
            Tuple of (Mx, My, Mz) total moments
        """
        if calculation == "potential":
            return float(self.Mx), float(self.My), float(self.Mz)
        elif calculation == "viscous":
            # return float(jnp.sum(self.panel_D * self.mean_w_induced))
            raise NotImplementedError("Viscous moment calculation not implemented.")
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def update_2d_polars(self, L_2D: float, D_2D: float, My_2D: float) -> None:
        """Update 2D interpolated polar values.

        Args:
            L_2D: 2D lift coefficient
            D_2D: 2D drag coefficient
            My_2D: 2D moment coefficient
        """
        self.L_2D = L_2D
        self.D_2D = D_2D
        self.My_2D = My_2D

    def __str__(self) -> str:
        """String representation of the StripLoads object."""
        return (
            f"StripLoads(chord={self.chord:.3f}, width={self.width:.3f}, "
            f"num_panels={self.num_panels}, mean_gamma={self.mean_gamma:.3f}, "
            f"mean_w_induced={self.mean_w_induced:.3f})"
        )

    def __repr__(self) -> str:
        """String representation of the StripLoads object."""
        return self.__str__()
