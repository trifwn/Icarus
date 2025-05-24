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

        # Mean values
        self.mean_gamma = jnp.mean(self.gammas)
        self.mean_w_induced = jnp.mean(self.w_induced)

        # Initialize the data arrays
        self.Ls = jnp.zeros(self.num_panels)
        self.Ds = jnp.zeros(self.num_panels)
        self.D_trefftz = 0.0
        self.Mx = 0.0
        self.My = 0.0
        self.Mz = 0.0  # 2D interpolated polars
        self.L_2D = 0.0
        self.D_2D = 0.0
        self.My_2D = 0.0

        # Potential loads (from VLM)
        self.L_potential = 0.0
        self.D_potential = 0.0

    def calc_mean_values(self) -> None:
        self.mean_gamma = jnp.mean(self.gammas)
        self.mean_w_induced = jnp.mean(self.w_induced)

    def calc_aerodynamic_loads(self, dynamic_pressure: float) -> None:
        """Calculate aerodynamic loads for this strip.

        Args:
            dynamic_pressure: Dynamic pressure (0.5 * rho * V^2)
        """
        # Calculate lift and drag for each panel
        for i in range(self.num_panels):
            # Lift per unit span (basic vortex panel method)
            self.Ls = self.Ls.at[i].set(self.gammas[i] * dynamic_pressure * self.mean_panel_width[i])

            # Induced drag calculation (simplified)
            self.Ds = self.Ds.at[i].set(
                self.gammas[i] * self.w_induced[i] * dynamic_pressure * self.mean_panel_width[i],
            )

    def get_total_lift(self) -> float:
        """Get total lift for this strip."""
        return float(jnp.sum(self.Ls))

    def get_total_drag(self) -> float:
        """Get total drag for this strip."""
        return float(jnp.sum(self.Ds))

    def get_total_moment(self) -> tuple[float, float, float]:
        """Get total moments (Mx, My, Mz) for this strip."""
        return float(self.Mx), float(self.My), float(self.Mz)

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
