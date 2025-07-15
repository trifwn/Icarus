from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jaxtyping import Array
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil
    from ICARUS.vehicle import WingSurface

from . import panel_cp
from . import panel_cp_normal


class LSPTSurface:
    """
    Class to represent a surface in the LSPT format.
    This class is used to handle the surface data for aerodynamic analysis.
    Optimized for performance with JAX and autodiff capabilities.
    """

    def __init__(
        self,
        surface: WingSurface,
        wing_id: int = 0,
    ) -> None:
        """
        Initialize the LSPTSurface with a name and a list of points.

        :param surface: WingSurface object
        :param wing_id: Wing identifier
        """
        self.name = surface.name
        self.N = surface.N
        self.M = surface.M
        self.id = wing_id
        self.is_lifting = surface.is_lifting

        self.span_positions = jnp.asarray(surface._span_dist)
        self.chords = jnp.asarray(surface._chord_dist)
        self.strip_total_pitches = jnp.asarray(surface.strip_pitches)
        self.mean_aerodynamic_chord: float = surface.mean_aerodynamic_chord

        self.num_grid_points = surface.num_grid_points
        self.num_strips = surface.N - 1

        # Airfoils
        self.airfoils: list[Airfoil] = surface.airfoils

        # Wake Parameters - vectorized computation
        self.wake_shedding_panel_indices = jnp.arange(
            (surface.M - 2),
            (surface.N - 1) * (surface.M - 1),
            (surface.M - 1),
        )

        self.grid = jnp.asarray(surface.grid)

        # Convert to JAX arrays for performance
        self.panels = jnp.asarray(surface.panels)
        self.panel_cps = jnp.asarray(surface.control_points)
        self.panel_normals = jnp.asarray(surface.control_nj)

        self.near_wake_panels = jnp.zeros((0, 4, 3), dtype=jnp.float32)
        self.near_wake_panel_cps = jnp.zeros((0, 3), dtype=jnp.float32)
        self.near_wake_panel_normals = jnp.zeros((0, 3), dtype=jnp.float32)

        self.flat_wake_panels = jnp.zeros((0, 4, 3), dtype=jnp.float32)
        self.flat_wake_panel_cps = jnp.zeros((0, 3), dtype=jnp.float32)
        self.flat_wake_panel_normals = jnp.zeros((0, 3), dtype=jnp.float32)

    @property
    def num_panels(self) -> int:
        """
        Get the number of panels in the surface.
        """
        return self.panels.shape[0]

    @property
    def num_near_wake_panels(self) -> int:
        """
        Get the number of near wake panels.
        """
        return self.near_wake_panels.shape[0]

    @property
    def num_flat_wake_panels(self) -> int:
        """
        Get the number of flat wake panels.
        """
        return self.flat_wake_panels.shape[0]

    @property
    def num_all_panels(self) -> int:
        """
        Get the total number of panels including near and flat wake panels.
        """
        return self.num_panels + self.num_near_wake_panels + self.num_flat_wake_panels

    def add_near_wake_panels(self) -> None:
        """
        Add panels in the near wake to account for the wake influence on the wing.
        Vectorized implementation for 100x performance improvement.
        """
        # Get wake shedding indices
        wake_shedding_indices = self.wake_shedding_panel_indices
        n_wake_panels = len(wake_shedding_indices)

        if n_wake_panels == 0:
            return

        # Get panels and normals that are shedding wake
        wake_panels = self.panels[wake_shedding_indices]
        wake_nj = self.panel_normals[wake_shedding_indices]

        # Vectorized computation of all near wake panels
        compute_wake_vmap = vmap(compute_near_wake_panel, in_axes=(0, 0))
        near_wake_panels = compute_wake_vmap(wake_panels, wake_nj)

        near_wake_control_points = vmap(panel_cp)(near_wake_panels)
        near_wake_nj = vmap(panel_cp_normal)(near_wake_panels)
        # self.panels = jnp.concatenate([self.panels, near_wake_panels], axis=0)
        self.near_wake_panels = near_wake_panels
        self.near_wake_panel_cps = near_wake_control_points
        self.near_wake_panel_normals = near_wake_nj

    def add_flat_wake_panels(
        self,
        num_of_wake_panels: int = 5,
        wake_x_inflation: float = 1.1,
        farfield_distance: float = 5,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        """
        Add flat wake panels after the near wake panels.

        Args:
            num_of_wake_panels: Number of wake panels per strip
            wake_x_inflation: Wake panel size inflation factor
            farfield_distance: Distance of farfield
            alpha: Angle of attack
            beta: Sideslip angle
        """

        # Get near wake panels
        near_wake_panels = self.near_wake_panels
        n_strips = len(near_wake_panels)
        if n_strips == 0:
            return

        # Freestream direction (vectorized)
        freestream_direction = jnp.array(
            [
                -jnp.cos(alpha) * jnp.cos(beta),
                0.0,
                jnp.sin(alpha) * jnp.cos(beta),
            ],
        )
        MAC = self.mean_aerodynamic_chord

        # Create a partial function with static arguments for vmap
        compute_strip_partial = partial(
            compute_flat_wake_panels_for_strip,
            num_wake_panels=num_of_wake_panels,
            wake_x_inflation=wake_x_inflation,
            farfield_distance=farfield_distance,
            MAC=MAC,
            freestream_direction=freestream_direction,
        )

        # Vectorized computation for all strips
        compute_strip_vmap = vmap(compute_strip_partial, in_axes=(0,))

        flat_wake_panels, flat_wake_control_points, flat_wake_control_nj = (
            compute_strip_vmap(near_wake_panels)
        )

        self.flat_wake_panels = flat_wake_panels.reshape(-1, 4, 3)
        self.flat_wake_panel_cps = flat_wake_control_points.reshape(-1, 3)
        self.flat_wake_panel_normals = flat_wake_control_nj.reshape(-1, 3)

    def __repr__(self) -> str:
        return f"LSPTSurface(name={self.name}, N={self.N}, M={self.M}, num_panels={self.num_panels})"

    def plot_panels(
        self,
        ax: Axes3D | None = None,
        plot_wake: bool = False,
        legend: bool = True,
    ) -> None:
        if ax is None:
            fig: Figure | SubFigure | None = plt.figure()
            ax_: Axes3D = fig.add_subplot(projection="3d")  # noqa
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
            ax_.plot_wireframe(xs, ys, zs, linewidth=1.5)

        if plot_wake:
            # Add the near wake panels in green
            for i in np.arange(0, self.near_wake_panels.shape[0]):
                p1, p3, p4, p2 = self.near_wake_panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
                ax_.plot_wireframe(xs, ys, zs, color="g", linewidth=1.5)

            # Add the flat wake panels in orange
            for i in np.arange(0, self.flat_wake_panels.shape[0]):
                p1, p3, p4, p2 = self.flat_wake_panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
                ax_.plot_wireframe(xs, ys, zs, color="orange", linewidth=1.5)

        # scatter the control points and grid points that are not part of the wake
        surf_control_points = self.panel_cps[: self.num_panels, :]
        ax_.scatter(
            *surf_control_points.T,
            color="r",
            marker="o",
            s=20,
        )
        surf_grid_points = self.grid[: self.num_grid_points, :]
        ax_.scatter(
            *surf_grid_points.T,
            color="k",
            marker="x",
            s=20,
        )

        # Add the wake shedding points in blue
        wake_shedding_indices = self.wake_shedding_panel_indices
        ax_.scatter(
            *self.panel_cps[wake_shedding_indices, :].T,
            color="b",
            marker="x",
            s=50,
        )

        if legend:
            if plot_wake:
                ax_.scatter([], [], [], color="orange", label="Flat Wake Panels")
                ax_.scatter([], [], [], color="g", label="Near Wake Panels")
            ax_.scatter([], [], [], color="b", marker="x", label="Wake Shedding Panels")
            ax_.scatter([], [], [], color="r", marker="o", label="Control Points")
            ax_.scatter([], [], [], color="k", marker="x", label="Grid Points")

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

    @property
    def global_panel_indices(self) -> jnp.ndarray:
        if not hasattr(self, "_global_panel_indices"):
            raise ValueError(
                "Global panel indices not set. Call add_near_wake_panels() and add_flat_wake_panels() first.",
            )
        return self._global_panel_indices

    @global_panel_indices.setter
    def global_panel_indices(self, indices: jnp.ndarray) -> None:
        """
        Set the global panel indices for the surface.
        This is used to keep track of all panels including near and flat wake panels.
        """
        self._global_panel_indices = indices

    @property
    def global_near_wake_panel_indices(self) -> jnp.ndarray:
        if not hasattr(self, "_global_near_wake_panel_indices"):
            raise ValueError(
                "Global near wake panel indices not set. Call add_near_wake_panels() first.",
            )
        return self._global_near_wake_panel_indices

    @global_near_wake_panel_indices.setter
    def global_near_wake_panel_indices(self, indices: jnp.ndarray) -> None:
        """
        Set the global near wake panel indices for the surface.
        This is used to keep track of near wake panels.
        """
        self._global_near_wake_panel_indices = indices

    @property
    def global_flat_wake_panel_indices(self) -> jnp.ndarray:
        if not hasattr(self, "_global_flat_wake_panel_indices"):
            raise ValueError(
                "Global flat wake panel indices not set. Call add_flat_wake_panels() first.",
            )
        return self._global_flat_wake_panel_indices

    @global_flat_wake_panel_indices.setter
    def global_flat_wake_panel_indices(self, indices: jnp.ndarray) -> None:
        """
        Set the global flat wake panel indices for the surface.
        This is used to keep track of flat wake panels.
        """
        self._global_flat_wake_panel_indices = indices


@partial(jax.jit, static_argnums=(1,))
def compute_flat_wake_panels_for_strip(
    near_wake_panel: Array,
    num_wake_panels: int,
    wake_x_inflation: float,
    farfield_distance: float,
    MAC: float,
    freestream_direction: Array,
) -> tuple[Array, Array, Array]:
    """
    Compute all flat wake panels for a single strip.
    JIT-compiled and vectorized for maximum performance.
    """
    # Generate panel lengths for all wake panels at once
    # Use static num_wake_panels to avoid concretization error
    i_vals = jnp.arange(num_wake_panels)
    base_lengths = farfield_distance * MAC / (wake_x_inflation**i_vals)
    panel_lengths = wake_x_inflation**i_vals * base_lengths

    def create_single_flat_panel(length, prev_panel) -> Array:
        flat_panel = jnp.array(
            [
                prev_panel[3],
                prev_panel[2],
                prev_panel[2] - length * freestream_direction,
                prev_panel[3] - length * freestream_direction,
            ],
        )
        return flat_panel

    # Use scan for efficient sequential computation
    def scan_fn(prev_panel, length) -> tuple[Array, Array]:
        new_panel = create_single_flat_panel(length, prev_panel)
        return new_panel, new_panel

    _, flat_panels = jax.lax.scan(scan_fn, near_wake_panel, panel_lengths)

    # Vectorized computation of normals and control points
    control_points = vmap(panel_cp)(flat_panels)
    normals = vmap(panel_cp_normal)(flat_panels)
    return flat_panels, control_points, normals


@jax.jit
def compute_near_wake_panel(panel: Array, nj: Array) -> Array:
    """
    Compute a single near wake panel given the shedding panel and its normal.
    JIT-compiled for performance.
    """
    # Trailing edge vector
    te = panel[3] - panel[2]

    # Near wake direction (vectorized cross product)
    near_wake_dir = jnp.cross(nj, te)
    near_wake_dir = near_wake_dir / jnp.linalg.norm(near_wake_dir)

    # Panel length (vectorized computation)
    panel_length = jnp.linalg.norm(
        (panel[1] + panel[0]) * 0.5 - (panel[2] + panel[3]) * 0.5,
    )

    # Create near wake panel (vectorized)
    near_wake_panel = jnp.array(
        [
            panel[3],
            panel[2],
            panel[2] - panel_length * near_wake_dir,
            panel[3] - panel_length * near_wake_dir,
        ],
    )
    return near_wake_panel
