from typing import TYPE_CHECKING
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Polygon
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

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

        assert len(panels) == len(panel_idxs), "Panels and panel_idxs must have the same length."
        self.num_panels = len(panel_idxs)

        self.airfoil = airfoil

        # Get mean panel dimensions
        (
            self.mean_panel_length,
            self.mean_panel_width,
            self.mean_panel_height,
        ) = self.calculate_panel_dimensions()

        # Placeholders for aerodynamic properties
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

    def calculate_panel_dimensions(self) -> tuple[Float, Float, Float]:
        """Calculate mean panel lengths and widths."""

        def panel_dimensions(panel: Float[Array, "2"]) -> tuple[Float, Float, Float]:
            """Calculate the length and width of a panel."""
            dx = ((panel[3, 0] - panel[0, 0]) + (panel[2, 0] - panel[1, 0])) / 2.0

            dy = ((panel[3, 1] - panel[2, 1]) + (panel[0, 1] - panel[1, 1])) / 2.0

            dz = ((panel[3, 2] - panel[0, 2]) + (panel[2, 2] - panel[1, 2])) / 2.0
            return dx, dy, dz

        # Calculate mean panel lengths and widths
        return vmap(panel_dimensions)(self.panels)

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
        self.panel_L = density * umag * gammas * self.mean_panel_width
        self.panel_D = -density * self.w_induced * gammas * self.mean_panel_width

        if jnp.any(jnp.isnan(self.panel_L)):
            raise ValueError("NaN values found in panel_L. Check gamma calculations.")

    def get_total_lift(self, calculation="potential") -> float:
        """Get total lift for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_L))
        elif calculation == "viscous":
            # return float(jnp.sum(self.panel_L * self.mean_w_induced))
            raise NotImplementedError("Viscous lift calculation not implemented.")
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def get_total_drag(self, calculation="potential") -> float:
        """Get total drag for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_D))
        elif calculation == "viscous":
            # return float(jnp.sum(self.panel_D * self.mean_w_induced))
            raise NotImplementedError("Viscous drag calculation not implemented.")
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def get_total_moments(self, calculation="potential") -> tuple[float, float, float]:
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

    #### Plotting methods (if needed) can be added here ####
    def plot_surface(
        self,
        ax: Axes3D | None = None,
        data: Optional[Float[Array, "..."]] = None,
        scalar_map: ScalarMappable | tuple[float, float] | None = None,
        colorbar: Colorbar | None = None,
    ) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
            show_plot = True
        else:
            ax_ = ax
            fig = ax_.get_figure()
            show_plot = False

        if fig is None:
            raise ValueError("Axes must be part of a figure")

        # Add the grid panel wireframes
        for i in np.arange(0, self.panels.shape[0]):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            # ax_.plot_wireframe(xs, ys, zs, linewidth=1.5)

            ax_.plot_surface(xs, ys, zs, color="lightgray", alpha=0.5, edgecolor="k", linewidth=0.5)

        # If data is provided, plot it as a surface
        if data is not None:
            if scalar_map is None:
                norm = Normalize(vmin=np.min(data), vmax=np.max(data))
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif isinstance(scalar_map, tuple):
                norm = Normalize(vmin=scalar_map[0], vmax=scalar_map[1])
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif not isinstance(scalar_map, ScalarMappable):
                raise TypeError("scalar_map must be a ScalarMappable or a tuple of (vmin, vmax).")

            if data.shape[0] != self.panels.shape[0]:
                raise ValueError("Data must have the same length as the number of panels.")

            # Plot each panel with the corresponding data value
            for i in np.arange(0, self.panels.shape[0]):
                p1, p3, p4, p2 = self.panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))

                val = np.array(data[i])
                ax_.plot_surface(xs, ys, zs, color=scalar_map.to_rgba(val), alpha=0.5, edgecolor="k", linewidth=0.5)

            if colorbar is None:
                colorbar = fig.colorbar(
                    scalar_map,
                    ax=ax_,
                    orientation="vertical",
                )

        if show_plot:
            ax_.set_title("Grid")
            ax_.set_xlabel("x")
            ax_.set_ylabel("y")
            ax_.set_zlabel("z")
            ax_.axis("equal")
            ax_.view_init(30, 150)

            ax_.legend()
            fig.show()

    def plot_xy(
        self,
        ax: Axes | None = None,
        data: Optional[Float[Array, "..."]] = None,
        scalar_map: ScalarMappable | tuple[float, float] | None = None,
        colorbar: Colorbar | None = None,
    ) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_ = fig.add_subplot()  # type: ignore
            show_plot = True
        else:
            ax_ = ax
            fig = ax_.get_figure()
            show_plot = False

        if fig is None:
            raise ValueError("Axes must be part of a figure")

        if data is not None:
            if scalar_map is None:
                norm = Normalize(vmin=np.min(data), vmax=np.max(data))
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif isinstance(scalar_map, tuple):
                norm = Normalize(vmin=scalar_map[0], vmax=scalar_map[1])
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif not isinstance(scalar_map, ScalarMappable):
                raise TypeError("scalar_map must be a ScalarMappable or a tuple of (vmin, vmax).")

            if data.shape[0] != self.panels.shape[0]:
                raise ValueError("Data must have the same length as the number of panels.")

            # Plot each panel with the corresponding data value
            for i in np.arange(0, self.panels.shape[0]):
                verts = self.panels[i, :, :2]
                verts = np.array(verts)

                val = np.array(data[i])
                # Polygon()
                poly = Polygon(
                    verts,
                    closed=True,
                    edgecolor="k",
                    linewidth=0.5,
                    facecolor=scalar_map.to_rgba(val, alpha=0.5),
                )
                ax_.add_patch(poly)

            if colorbar is None:
                colorbar = fig.colorbar(
                    scalar_map,
                    ax=ax_,
                    orientation="vertical",
                )
        else:
            # Plot the flattened surface [x, y, _] # coordinates (skip z for 2D)
            for i in np.arange(0, self.panels.shape[0]):
                verts = self.panels[i, :, :2]
                verts = np.array(verts)
                poly = Polygon(
                    verts,
                    closed=True,
                    edgecolor="k",
                    linewidth=0.5,
                )
                ax_.add_patch(poly)

        if show_plot:
            ax_.set_title("Flat Surface")
            ax_.set_xlabel("x")
            ax_.set_ylabel("y")
            ax_.axis("equal")

            fig.show()
