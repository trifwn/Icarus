import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int


class StripData:
    def __init__(
        self,
        panels: Float[Array, "..."],
        panel_idxs: Int[Array, "..."],
        chord: Float,
        width: Float,
    ):
        self.chord = chord
        self.width = width
        self.panels = panels
        self.panel_idxs = panel_idxs
        self.num_panels = len(panel_idxs)

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
        self.Mz = 0.0

        # 2D interpolated polars
        self.L_2D = 0.0
        self.D_2D = 0.0
        self.My_2D = 0.0

    def calc_mean_values(self) -> None:
        self.mean_gamma = jnp.mean(self.gammas)
        self.mean_w_induced = jnp.mean(self.w_induced)
