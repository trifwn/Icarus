from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.aero.utils import panel_cp
from ICARUS.aero.utils import panel_cp_normal
from ICARUS.aero.utils import panel_dimensions

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil


class StripLoads:
    def __init__(
        self,
        panels: Float[Array, ...],
        panel_idxs: Int[Array, ...],
        chord: Float,
        width: Float,
        airfoil_pitch: Float = 0.0,
        airfoil: Airfoil | None = None,
    ):
        self.chord = chord
        self.width = width
        self.surface_area_proj = self.chord * self.width
        self.airfoil_pitch = airfoil_pitch
        self.airfoil = airfoil

        assert len(panels) == len(
            panel_idxs,
        ), "Panels and panel_idxs must have the same length."
        self.panels = panels
        self.panel_idxs = panel_idxs
        self.num_panels = len(panel_idxs)

        # Find the Bounding Box of the panels
        x_min = jnp.min(panels[:, :, 0])  # x_min
        x_max = jnp.max(panels[:, :, 0])  # x_max
        y_min = jnp.min(panels[:, :, 1])  # y_min
        y_max = jnp.max(panels[:, :, 1])  # y_max
        z_min = jnp.min(panels[:, :, 2])  # z_min
        z_max = jnp.max(panels[:, :, 2])  # z_max

        # Find the Quarter Chord Reference Point and set the
        # reference point to the quarter chord position
        self.x_ref = (x_min + x_max) / 2.0 + self.chord / 4.0
        self.y_ref = (y_min + y_max) / 2.0
        self.z_ref = (z_min + z_max) / 2.0

        # Get mean panel dimensions
        (
            self.mean_panel_length,
            self.mean_panel_width,
            self.mean_panel_height,
        ) = vmap(panel_dimensions)(self.panels)

        # Placeholders for aerodynamic properties
        self.gammas = jnp.zeros(self.num_panels) * jnp.nan
        self.w_induced = jnp.zeros(self.num_panels) * jnp.nan

        # Placeholders for panel loads
        self.panel_L = jnp.zeros(self.num_panels) * jnp.nan
        self.panel_D = jnp.zeros(self.num_panels) * jnp.nan

        # Placeholders for aerodynamic loads
        self.L = jnp.nan
        self.D = jnp.nan
        self.D_trefftz = jnp.nan
        self.Mx = jnp.nan
        self.My = jnp.nan
        self.Mz = jnp.nan

        # Placeholder for 2D aerodynamic loads
        self.L_2D = jnp.nan
        self.D_2D = jnp.nan
        self.My_2D = jnp.nan

        # Viscous Coefficients
        self.CL_2D = jnp.nan
        self.CD_2D = jnp.nan
        self.Cm_2D = jnp.nan

    @property
    def mean_gamma(self) -> Float:
        """Mean circulation strength across all panels."""
        return jnp.mean(self.gammas)

    @property
    def mean_w_induced(self) -> Float:
        """Mean induced velocity across all panels."""
        return jnp.mean(self.w_induced)

    def set_effective_flow_conditions(
        self,
        density: Float,
        viscosity: Float,
        velocity: Float,
    ) -> None:
        """Calculate effective flow conditions accounting for induced effects.

        This method computes the effective Reynolds number, velocity, and angle of attack
        for the strip considering the induced downwash from the VLM solution.

        Args:
            density: Air density (kg/m³)
            viscosity: Dynamic viscosity (Pa·s)
            velocity: Freestream velocity magnitude (m/s)
        """
        w_induced = self.mean_w_induced
        # Get the effective angle of attack
        effective_aoa = (
            jnp.arctan(w_induced / velocity) * 180 / jnp.pi + self.airfoil_pitch
        )

        # Get the reynolds number of each strip
        effective_velocity = jnp.sqrt(w_induced**2 + velocity**2)
        effective_reynolds = density * effective_velocity * self.chord / viscosity
        effective_dynamic_pressure = 0.5 * density * effective_velocity**2

        self.effective_aoa = effective_aoa
        self.effective_reynolds = effective_reynolds
        self.effective_velocity = effective_velocity
        self.effective_dynamic_pressure = effective_dynamic_pressure

    def interpolate_viscous_coefficients(
        self,
        density: Float,
        viscosity: Float,
        airspeed: Float,
        solver: str = "Xfoil",
    ) -> None:
        """Calculate viscous loads for this strip.

        This method is a placeholder and should be implemented based on the specific
        viscous flow model used (e.g., boundary layer theory, CFD).

        Args:
            density: Air density (kg/m³)
            viscosity: Dynamic viscosity (Pa·s)
            airspeed: Freestream velocity magnitude (m/s)
            solver: Solver to use for interpolation (default: "Xfoil")
        """
        if self.airfoil is None:
            return

        try:
            from ICARUS.database import Database

            DB = Database.get_instance()
        except Exception as e:
            raise RuntimeError(
                "Database instance could not be retrieved. Ensure the database is initialized.",
            ) from e

        try:
            CL, CD, Cm = DB.foils_db.interpolate_polars(
                reynolds=self.effective_reynolds,
                airfoil=self.airfoil,
                aoa=self.effective_aoa,
                solver=solver,
            )
        except ValueError:
            CL, CD, Cm = jnp.nan, jnp.nan, jnp.nan

        # Save the coefficients
        self.CL_2D = CL
        self.CD_2D = CD
        self.Cm_2D = Cm

    def calc_potential_loads(
        self,
        density: float,
        airspeed: float,
    ) -> None:
        """Calculate aerodynamic loads for this strip.

        Args:
            density: Air density (kg/m^3)
            umag: Freestream velocity magnitude (m/s)
        """

        gammas = self.gammas.copy()
        gammas = gammas.at[1:].set(
            self.gammas[1:] - self.gammas[:-1],  # Calculate gamma differences
        )
        w_in = self.w_induced.copy()

        if jnp.any(jnp.isnan(gammas)):
            raise ValueError("NaN values found in gammas. Check gamma calculations.")

        # Calculate mean panel lengths and widths
        self.panel_L = density * airspeed * gammas * self.mean_panel_width
        self.panel_D = -density * w_in * gammas * self.mean_panel_width

        self.D_trefftz = -density / 2 * self.width * gammas[-1] * w_in[-1]

        if jnp.any(jnp.isnan(self.panel_L)):
            raise ValueError("NaN values found in panel_L. Check gamma calculations.")

    def calc_potential_moments(
        self,
        reference_point: Float[Array, 3],
    ) -> tuple[Array, Array, Array]:
        """Calculate moments for this strip.

        Args:
            reference_point: Reference point for moment calculation

        Returns:
            Tuple of (Mx, My, Mz) total moments
        """
        # Calculate the moment contributions from each panel
        panel_cps = vmap(panel_cp)(self.panels)
        panel_normals = vmap(panel_cp_normal)(self.panels)
        lever_arms = panel_cps - reference_point

        panel_lift = self.panel_L
        panel_drag = self.panel_D

        M_lift = jnp.sum(
            panel_lift[:, None] * jnp.cross(lever_arms, panel_normals),
            axis=0,
        )
        M_drag = jnp.sum(
            panel_drag[:, None] * jnp.cross(lever_arms, panel_normals),
            axis=0,
        )

        M = M_lift + M_drag
        Mx = M[0]
        My = M[1]
        Mz = M[2]

        return Mx, My, Mz

    def calc_viscous_loads(
        self,
        density: Float,
        airspeed: Float,
    ) -> None:
        """Calculate viscous loads for this strip.

        This method is a placeholder and should be implemented based on the specific
        viscous flow model used (e.g., boundary layer theory, CFD).
        """
        CL = self.CL_2D
        CD = self.CD_2D
        # if jnp.isnan(CL) or jnp.isnan(CD):
        #     raise ValueError("Viscous coefficients are not set. Call interpolate_viscous_coefficients first.")

        surface: Float = self.surface_area_proj
        dynamic_pressure = 0.5 * density * airspeed**2
        self.L_2D = CL * dynamic_pressure * surface
        self.D_2D = CD * dynamic_pressure * surface

    def calc_viscous_moment(
        self,
        reference_point: Float[Array, 3],
    ) -> tuple[Array, Array, Array]:
        CL = self.CL_2D
        CD = self.CD_2D
        Cm = self.Cm_2D
        # if jnp.isnan(CL) or jnp.isnan(CD) or jnp.isnan(Cm):
        #     raise ValueError("Viscous coefficients are not set. Call interpolate_viscous_coefficients first.")

        surface: Float = self.surface_area_proj
        dynamic_pressure = self.effective_dynamic_pressure

        L_2D = CL * dynamic_pressure * surface
        D_2D = CD * dynamic_pressure * surface
        My_2D_at_quarter_chord = Cm * dynamic_pressure * surface * self.chord

        # Calculate the moment contributions from the 2D lift and drag
        Mx_2D = L_2D * (reference_point[1] - self.z_ref) - D_2D * (
            reference_point[1] - self.z_ref
        )

        My_2D = My_2D_at_quarter_chord + (
            L_2D * (reference_point[0] - self.x_ref)
            - D_2D * (reference_point[0] - self.x_ref)
        )

        Mz_2D = L_2D * (reference_point[0] - self.x_ref) + D_2D * (
            reference_point[0] - self.x_ref
        )

        return Mx_2D, My_2D, Mz_2D

    def get_total_lift(
        self,
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> float:
        """Get total lift for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_L))
        elif calculation == "viscous":
            return float(self.L_2D)
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def get_total_drag(
        self,
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> float:
        """Get total drag for this strip."""
        if calculation == "potential":
            return float(jnp.sum(self.panel_D))
        elif calculation == "viscous":
            return float(self.D_2D)
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

    def get_total_moments(
        self,
        reference_point: Float[Array, 3],
        calculation: Literal["potential", "viscous"] = "potential",
    ) -> tuple[Array, Array, Array]:
        """Get total moments for this strip.

        Args:
            reference_point: Reference point for moment calculation
            calculation: Type of moment calculation ('potential' or 'viscous')

        Returns:
            Tuple of (Mx, My, Mz) total moments
        """
        if calculation == "potential":
            return self.calc_potential_moments(reference_point)
        elif calculation == "viscous":
            return self.calc_viscous_moment(reference_point)
        else:
            raise ValueError("Invalid calculation type. Use 'potential' or 'viscous'.")

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
        data: Float[Array, ...] | None = None,
        scalar_map: ScalarMappable | tuple[float, float] | None = None,
        colorbar: Colorbar | None = None,
    ) -> None:
        if ax is None:
            fig: Figure | SubFigure | None = plt.figure()
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

            ax_.plot_surface(
                xs,
                ys,
                zs,
                color="lightgray",
                alpha=0.5,
                edgecolor="k",
                linewidth=0.5,
            )

        # If data is provided, plot it as a surface
        if data is not None:
            if scalar_map is None:
                norm = Normalize(vmin=np.min(data), vmax=np.max(data))
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif isinstance(scalar_map, tuple):
                norm = Normalize(vmin=scalar_map[0], vmax=scalar_map[1])
                scalar_map = ScalarMappable(norm=norm, cmap="viridis")
            elif not isinstance(scalar_map, ScalarMappable):
                raise TypeError(
                    "scalar_map must be a ScalarMappable or a tuple of (vmin, vmax).",
                )

            if data.shape[0] != self.panels.shape[0]:
                raise ValueError(
                    "Data must have the same length as the number of panels.",
                )

            # Plot each panel with the corresponding data value
            for i in np.arange(0, self.panels.shape[0]):
                p1, p3, p4, p2 = self.panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))

                val = np.array(data[i])
                ax_.plot_surface(
                    xs,
                    ys,
                    zs,
                    color=scalar_map.to_rgba(val),
                    alpha=0.5,
                    edgecolor="k",
                    linewidth=0.5,
                )

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
            if isinstance(fig, Figure):
                fig.show()

    def plot_xy(
        self,
        ax: Axes | None = None,
        data: Float[Array, ...] | None = None,
        scalar_map: ScalarMappable | tuple[float, float] | None = None,
        colorbar: Colorbar | None = None,
    ) -> None:
        if ax is None:
            fig: Figure | SubFigure | None = plt.figure()
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
                raise TypeError(
                    "scalar_map must be a ScalarMappable or a tuple of (vmin, vmax).",
                )

            if data.shape[0] != self.panels.shape[0]:
                raise ValueError(
                    "Data must have the same length as the number of panels.",
                )

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
            if isinstance(fig, Figure):
                fig.show()
