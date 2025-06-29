from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Array
from jaxtyping import Int
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface

from ICARUS.core.types import FloatArray
from ICARUS.vehicle import Wing
from ICARUS.visualization import create_subplots
from ICARUS.visualization import flatten_axes

from . import StripData
from .post_process import get_potential_loads
from .vlm import get_LHS
from .vlm import get_RHS

coolwarm = colormaps.get_cmap("coolwarm")
viridis = colormaps.get_cmap("viridis")


class LSPT_Plane:
    """Wing Model using the Lifting Surface Potential Theory. The wing
    is divided into panels and the potential flow is solved using
    the no penetration condition. Also, the Trefftz plane is used to
    calculate the induced drag. To calculate forces and moments db polars,
    can also be used.
    """

    def __init__(
        self,
        plane: Airplane,
    ) -> None:
        # Store the wing segments
        self.surfaces: Sequence[WingSurface] = []
        self.surface_dict: dict[str, Any] = {}

        surf_id = 0
        NM_panels: int = 0
        NM_grid: int = 0
        num_strips: int = 0
        # Get the wing segments
        for surface in plane.surfaces:
            if isinstance(surface, Wing):
                for sub_surf in surface.wing_segments:
                    self.surfaces.append(sub_surf)
                    # Get the surface information
                    self.surface_dict[sub_surf.name] = {
                        "N": sub_surf.N,
                        "M": sub_surf.M,
                        "grid": sub_surf.grid,
                        "id": surf_id,
                        "symmetric_y": True if sub_surf.is_symmetric_y else False,
                        "panel_idxs": jnp.arange(
                            NM_panels,
                            NM_panels + (sub_surf.N - 1) * (sub_surf.M - 1),
                        ),
                    }
                    NM_panels += (sub_surf.N - 1) * (sub_surf.M - 1)
                    NM_grid += sub_surf.N * sub_surf.M
                    num_strips += sub_surf.N - 1

            else:
                self.surfaces.append(surface)
                # Get the surface information
                self.surface_dict[surface.name] = {
                    "N": surface.N,
                    "M": surface.M,
                    "grid": surface.grid,
                    "id": surf_id,
                    "symmetric_y": True if surface.is_symmetric_y else False,
                    "panel_idxs": jnp.arange(
                        NM_panels,
                        NM_panels + (surface.N - 1) * (surface.M - 1),
                    ),
                }
                NM_panels += (surface.N - 1) * (surface.M - 1)
                NM_grid += surface.N * surface.M
                num_strips += surface.N - 1
            surf_id += 1

        self.num_strips = num_strips
        self.lifting_surfaces = [surface for surface in self.surfaces if surface.is_lifting]

        # Plane properties
        self.S: float = plane.S
        self.CG: FloatArray = plane.CG
        self.MAC: float = plane.mean_aerodynamic_chord

        self._alpha: float = 0
        self._beta: float = 0

        # We need to create the grid
        self.NM = NM_panels
        self.NM_grid = NM_grid
        self.A = jnp.zeros((NM_panels, NM_panels))
        self.A_star = jnp.zeros((NM_panels, NM_panels))
        self.grid = jnp.zeros((NM_grid, 3))
        self.panels = jnp.zeros(
            (NM_panels, 4, 3),
        )  # Panels are defined by 4 points in 3D space
        self.gammas = jnp.zeros(NM_panels)
        self.control_points = jnp.zeros(
            (NM_panels, 3),
        )  # Control points are the center of the panels
        self.control_nj = jnp.zeros(
            (NM_panels, 3),
        )  # Normal vector at the control points

        NM_panels = 0
        NM_grid = 0
        strips = []
        self.strip_data: list[StripData] = []
        for surface in self.surfaces:
            self.panels = self.panels.at[
                NM_panels : NM_panels + (surface.N - 1) * (surface.M - 1),
                :,
                :,
            ].set(
                surface.panels,
            )
            self.control_points = self.control_points.at[
                NM_panels : NM_panels + (surface.N - 1) * (surface.M - 1),
                :,
            ].set(surface.control_points)
            self.control_nj = self.control_nj.at[
                NM_panels : NM_panels + (surface.N - 1) * (surface.M - 1),
                :,
            ].set(
                surface.control_nj,
            )
            self.grid = self.grid.at[NM_grid : NM_grid + surface.N * surface.M, :].set(
                surface.grid,
            )
            # Get the wake shedding indices
            self.surface_dict[surface.name]["wake_shedding_panel_indices"] = NM_panels + jnp.arange(
                (surface.M - 2),
                (surface.N - 1) * (surface.M - 1),
                surface.M - 1,
            )
            self.surface_dict[surface.name]["wake_shedding_grid_indices"] = NM_grid + jnp.arange(
                (surface.M - 1),
                surface.N * surface.M,
                surface.M,
            )

            chords = surface._chord_dist
            span_dist = surface._span_dist
            # Get all the strips by their panel indices
            for i in range(surface.N - 1):
                strip_idxs = jnp.arange(i * (surface.M - 1), (i + 1) * (surface.M - 1)) + NM_panels

                strips.append(strip_idxs)
                self.strip_data.append(
                    StripData(
                        panel_idxs=strip_idxs,
                        panels=self.panels[strip_idxs, :, :],
                        chord=(chords[i] + chords[i + 1]) / 2,
                        width=span_dist[i + 1] - span_dist[i],
                    ),
                )

            NM_panels += (surface.N - 1) * (surface.M - 1)
            NM_grid += surface.N * surface.M
        self.strips = strips

        # Get the wake shedding indices
        self.wake_shedding_panel_indices: Int[Array, ...] = jnp.concatenate(
            [self.surface_dict[surface.name]["wake_shedding_panel_indices"] for surface in self.surfaces],
        )
        self.wake_shedding_grid_indices: Int[Array, ...] = jnp.concatenate(
            [self.surface_dict[surface.name]["wake_shedding_grid_indices"] for surface in self.surfaces],
        )

        # Add the near wake panels
        self.create_near_wake_panels()
        self.create_flat_wake_panels()

        self.near_wake_indices: Int[Array, ...] = jnp.concatenate(
            [self.surface_dict[surface.name]["near_wake_panel_indices"] for surface in self.surfaces],
        )
        self.flat_wake_panel_indices: Int[Array, ...] = jnp.concatenate(
            [self.surface_dict[surface.name]["flat_wake_panel_indices"] for surface in self.surfaces],
        )
        self.PANEL_NUM: int = NM_panels + len(self.near_wake_indices)

    def create_near_wake_panels(self) -> None:
        """For each surface that is shedding wake, we will add a panel in the near wake
        to account for the wake influence on the wing.
        """
        for surface in self.surfaces:
            # Get wake shedding indices
            wake_shedding_indices: Array = self.surface_dict[surface.name]["wake_shedding_panel_indices"]
            # Getting the panel indices that are shedding wake gives us the orientation of the wake
            # For each panel that is shedding wake, we will add a panel in the near wake
            near_wake_panel_indices = jnp.zeros_like(wake_shedding_indices)
            for i, panel_idx in enumerate(wake_shedding_indices):
                # Get the panel that is shedding wake and its control point, normal vector, and length
                panel = self.panels[panel_idx]
                nj = self.control_nj[panel_idx]
                # From nj get the near wake direction.
                # The near wake direction is the direction normal to the trailing edge of the panel
                te = panel[3] - panel[2]
                near_wake_dir = jnp.cross(nj, te)
                # Normalize the near wake direction
                near_wake_dir = near_wake_dir / jnp.linalg.norm(near_wake_dir)

                # Get the panel length that is the average of the two chords
                panel_length = jnp.linalg.norm(
                    (panel[1] + panel[0]) / 2 - (panel[2] + panel[3]) / 2,
                )

                # Assuming this is the TE panel, we will add a panel in the near wake
                near_wake_panel = jnp.zeros_like(panel)
                near_wake_panel = near_wake_panel.at[0].set(panel[3])
                near_wake_panel = near_wake_panel.at[1].set(panel[2])
                near_wake_panel = near_wake_panel.at[2].set(
                    panel[2] - panel_length * near_wake_dir,
                )
                near_wake_panel = near_wake_panel.at[3].set(
                    panel[3] - panel_length * near_wake_dir,
                )

                near_wake_panel_nj = jnp.cross(
                    near_wake_panel[1] - near_wake_panel[0],
                    near_wake_panel[2] - near_wake_panel[0],
                )

                # Append the near wake panel to the panels array
                self.panels = jnp.concatenate(
                    [self.panels, jnp.expand_dims(near_wake_panel, axis=0)],
                    axis=0,
                )
                # Append the control point to the control points array
                self.control_points = jnp.concatenate(
                    [
                        self.control_points,
                        jnp.expand_dims(near_wake_panel.mean(axis=0), axis=0),
                    ],
                    axis=0,
                )
                # Append the normal vector to the control nj array
                self.control_nj = jnp.concatenate(
                    [
                        self.control_nj,
                        jnp.expand_dims(
                            near_wake_panel_nj / jnp.linalg.norm(near_wake_panel_nj),
                            axis=0,
                        ),
                    ],
                    axis=0,
                )
                # Append the near wake panel index to the near wake panel indices
                near_wake_panel_indices = near_wake_panel_indices.at[i].set(
                    len(self.panels) - 1,
                )

            self.surface_dict[surface.name]["near_wake_panel_indices"] = near_wake_panel_indices

    def create_flat_wake_panels(
        self,
        num_of_wake_panels: int = 10,
        wake_x_inflation: float = 1.1,
        farfield_distance: float = 5,
    ):
        """For each surface that is shedding wake, we will add a panels after the near wake
        that are flat and parallel to the freestream direction.

        Args:
            num_of_wake_panels (int, optional): Number of Wake panels. Defaults to 10.
            wake_x_inflation (float, optional): Wake inflation. Defaults to 1.1.
            farfield_distance (int, optional): Distance of farfield. Defaults to 30.

        """
        # Get the Freestream direction
        alpha = self.alpha
        beta = self.beta
        freestream_direction = jnp.array(
            [
                -jnp.cos(alpha) * jnp.cos(beta),
                0,
                jnp.sin(alpha) * jnp.cos(beta),
            ],
        )

        for surface in self.surfaces:
            num = 0
            MAC = surface.mean_aerodynamic_chord

            # Get the near wake panels
            near_wake_indices = self.surface_dict[surface.name]["near_wake_panel_indices"]
            near_wake_panels = self.panels[near_wake_indices]

            self.surface_dict[surface.name]["flat_wake_panel_indices"] = jnp.zeros(
                len(near_wake_indices) * num_of_wake_panels,
                dtype=jnp.int32,
            )

            for near_wake_panel in near_wake_panels:
                flat_wake_idxs = jnp.zeros(num_of_wake_panels)
                for i in range(num_of_wake_panels):
                    panel_length = farfield_distance * MAC / (wake_x_inflation**i)
                    # The panel length must satisfy the farfield distance given the wake_x_inflation

                    pan_length = wake_x_inflation**i * panel_length
                    # Create the flat wake panel
                    flat_wake_panel = jnp.zeros_like(near_wake_panel)
                    flat_wake_panel = flat_wake_panel.at[0].set(near_wake_panel[3])
                    flat_wake_panel = flat_wake_panel.at[1].set(near_wake_panel[2])
                    flat_wake_panel = flat_wake_panel.at[2].set(
                        near_wake_panel[2] - pan_length * freestream_direction,
                    )
                    flat_wake_panel = flat_wake_panel.at[3].set(
                        near_wake_panel[3] - pan_length * freestream_direction,
                    )

                    # Get the normal vector of the flat wake panel
                    flat_wake_panel_nj = jnp.cross(
                        flat_wake_panel[1] - flat_wake_panel[0],
                        flat_wake_panel[2] - flat_wake_panel[0],
                    )

                    # Append the flat wake panel to the panels array
                    self.panels = jnp.concatenate(
                        [self.panels, jnp.expand_dims(flat_wake_panel, axis=0)],
                        axis=0,
                    )
                    # Append the control point to the control points array
                    self.control_points = jnp.concatenate(
                        [
                            self.control_points,
                            jnp.expand_dims(flat_wake_panel.mean(axis=0), axis=0),
                        ],
                        axis=0,
                    )
                    # Append the normal vector to the control nj array
                    self.control_nj = jnp.concatenate(
                        [
                            self.control_nj,
                            jnp.expand_dims(flat_wake_panel_nj, axis=0),
                        ],
                        axis=0,
                    )

                    # Append the flat wake panel index to the flat wake panel indices
                    flat_wake_idxs = flat_wake_idxs.at[i].set(len(self.panels) - 1)

                    # Note the wake panel indices
                    self.surface_dict[surface.name]["flat_wake_panel_indices"] = (
                        self.surface_dict[surface.name]["flat_wake_panel_indices"].at[num].set(flat_wake_idxs[i])
                    )
                    num += 1
                    near_wake_panel = flat_wake_panel

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    def aseq(
        self,
        angles: FloatArray,
        state: State,
    ) -> DataFrame:
        self.factorize_system()
        umag = state.u_freestream
        # dens = state.environment.air_density

        Ls = jnp.zeros(len(angles))
        Ds = jnp.zeros(len(angles))
        Mys = jnp.zeros(len(angles))
        CLs = jnp.zeros(len(angles))
        CDs = jnp.zeros(len(angles))
        Cms = jnp.zeros(len(angles))
        Ls_2D = jnp.zeros(len(angles))
        Ds_2D = jnp.zeros(len(angles))
        Mys_2D = jnp.zeros(len(angles))
        # CLs_2D = jnp.zeros(len(angles))
        # CDs_2D = jnp.zeros(len(angles))
        # Cms_2D = jnp.zeros(len(angles))

        for i, aoa in enumerate(angles):
            self.alpha = aoa * jnp.pi / 180
            self.beta = 0

            Uinf = umag * jnp.cos(self.alpha) * jnp.cos(self.beta)
            Vinf = umag * jnp.cos(self.alpha) * jnp.sin(self.beta)
            Winf = umag * jnp.sin(self.alpha) * jnp.cos(self.beta)

            Q = jnp.array((Uinf, Vinf, Winf))
            RHS = get_RHS(self, Q)

            gammas = jax.scipy.linalg.lu_solve((self.A_LU, self.A_piv), RHS)
            self.gammas = gammas
            w = jnp.matmul(self.A_star, gammas)

            # strips_w_induced = jnp.zeros(len(self.strips))
            # strips_gammas = jnp.zeros(len(self.strips))
            for strip in self.strip_data:
                strip_idxs = strip.panel_idxs
                strip.gammas = gammas[strip_idxs]
                strip.w_induced = w[strip_idxs]
                strip.calc_mean_values()

            (L, D, D2, Mx, My, Mz, CL, CD, Cm, L_pan, D_pan) = get_potential_loads(
                plane=self,
                state=state,
                ws=w,
                gammas=gammas,
            )
            # Store the results
            self.L_pan = L_pan
            self.D_pan = D_pan

            # No pen
            Ls = Ls.at[i].set(L)
            Ds = Ds.at[i].set(D)
            Mys = Mys.at[i].set(My)
            CLs = CLs.at[i].set(CL)
            CDs = CDs.at[i].set(CD)
            Cms = Cms.at[i].set(Cm)

            if True:
                Ls = Ls.at[i].set(2 * Ls[i])
                Ds = Ds.at[i].set(2 * Ds[i])
                Mys = Mys.at[i].set(2 * Mys[i])
                CLs = CLs.at[i].set(2 * CLs[i])
                CDs = CDs.at[i].set(2 * CDs[i])
                Cms = Cms.at[i].set(2 * Cms[i])

            # 2D polars
            # L_2D, D_2D, My_2D, CL_2D, CD_2D, Cm_2D =
            # Ls_2D[i] = L_2D
            # Ds_2D[i] = D_2D
            # Mys_2D[i] = My_2D
            # CLs_2D[i] = CL_2D
            # CDs_2D[i] = CD_2D
            # Cms_2D[i] = Cm_2D

        df = pd.DataFrame(
            {
                "AoA": angles,
                "LSPT Potential Fz": Ls,
                "LSPT Potential Fx": Ds,
                "LSPT Potential My": Mys,
                "LSPT 2D Fz": Ls_2D,
                "LSPT 2D Fx": Ds_2D,
                "LSPT 2D My": Mys_2D,
            },
        )
        return df

    def factorize_system(
        self,
    ) -> None:
        A, A_star = get_LHS(self)
        # Perform LU decomposition on A and A_star
        A_LU, A_piv = jax.scipy.linalg.lu_factor(A)
        # A_star_LU, A_star_piv = jax.scipy.linalg.lu_factor(A_star)
        self.A_LU = A_LU
        self.A_piv = A_piv
        self.A_star = A_star

    def plot_panels(
        self,
        ax: Axes3D | None,
        plot_near_wake: bool = False,
    ) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
        else:
            ax_ = ax
            fig = ax_.get_figure()

        if fig is None:
            raise ValueError("Axes must be part of a figure")

        # Add the grid panel wireframes
        for i in np.arange(0, self.NM):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_.plot_wireframe(xs, ys, zs, linewidth=1.5)

        # Add the near wake panels in green
        near_wake_indices = self.near_wake_indices
        for i in near_wake_indices:
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_.plot_wireframe(xs, ys, zs, color="g", linewidth=1.5)

        # Add the flat wake panels in orange
        flat_wake_indices = self.flat_wake_panel_indices
        for i in flat_wake_indices:
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_.plot_wireframe(xs, ys, zs, color="orange", linewidth=1.5)

        # Add the wake shedding points in blue
        wake_shedding_indices = self.wake_shedding_panel_indices
        ax_.scatter(
            *self.control_points[wake_shedding_indices, :].T,
            color="b",
            label="Wake Shedding Panels",
            marker="x",
            s=50,
        )

        # scatter the control points and grid points that are not part of the wake
        surf_control_points = self.control_points[: self.NM, :]
        ax_.scatter(
            *surf_control_points.T,
            color="r",
            label="Control Points",
            marker="o",
            s=20,
        )
        surf_grid_points = self.grid[: self.NM_grid, :]
        ax_.scatter(
            *surf_grid_points.T,
            color="k",
            label="Grid Points",
            marker="x",
            s=20,
        )

        ax_.set_title("Grid")
        ax_.set_xlabel("x")
        ax_.set_ylabel("y")
        ax_.set_zlabel("z")
        ax_.axis("equal")
        ax_.view_init(30, 150)

        if plot_near_wake:
            # Add the near wake control points in green
            near_wake_indices = self.near_wake_indices
            ax_.scatter(
                *self.control_points[near_wake_indices, :].T,
                color="g",
                label="Near Wake Panels",
                marker="x",
                s=50,
            )

            # Add the flat wake control points in orange
            flat_wake_indices = self.flat_wake_panel_indices
            ax_.scatter(
                *self.control_points[flat_wake_indices, :].T,
                color="orange",
                label="Flat Wake Panels",
                marker="x",
                s=50,
            )
        ax_.legend()
        fig.show()

    def plot_gammas(self, ax: Axes3D | None = None) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
        else:
            ax_now = ax
            fig = ax_now.get_figure()

        if fig is None:
            raise ValueError("Axes must be part of a figure")
        self.plot_panels(ax_now)

        # Plot the gammas by coloring the panels
        gammas = self.gammas
        # Normalize the gammas
        gammas = gammas / jnp.max(jnp.abs(gammas))

        for i in np.arange(0, self.PANEL_NUM):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            # Color the area inside the wireframe
            ax_now.plot_surface(
                xs,
                ys,
                zs,
                color=coolwarm(gammas[i]),
                alpha=0.9,
            )

        # Add colorbar
        norm = Normalize(vmin=jnp.min(gammas).item(), vmax=jnp.max(gammas).item())
        sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_now, label="Gamma")

        ax_now.set_title("Grid")
        ax_now.set_xlabel("x")
        ax_now.set_ylabel("y")
        ax_now.set_zlabel("z")
        ax_now.legend()

    def plot_surface_gamma_distribution(self, axs: list[Axes] | None = None) -> None:
        if axs is not None:
            axs_: list[Axes] = flatten_axes(axs)
            fig = axs_[0].get_figure()
            if fig is None:
                raise ValueError("Axes must be part of a figure")
        else:
            fig, axs_ = create_subplots(
                nrows=len(self.surfaces),
                ncols=1,
                squeeze=False,
            )

        # Plot the gamma distribution on the wing surface
        for surface_name, surf_dict in self.surface_dict.items():
            surf_id: int = surf_dict["id"]
            ax: Axes = axs_[surf_id]
            ax.set_title(f"{surface_name} Gamma Distribution")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # Add a 2D plot of the gamma distribution
            gammas = self.gammas
            # Find the indices of the panels that belong to the surface
            panel_indices = surf_dict["panel_idxs"]
            gammas_surf = gammas[panel_indices]
            N: int = surf_dict["N"]
            M: int = surf_dict["M"]

            # Reshape the gammas to the grid
            gammas_surf = jnp.reshape(gammas_surf, (N - 1, M - 1))
            ax.matshow(
                gammas_surf,
                cmap=viridis,
            )
            # Add colorbar
            norm = Normalize(
                vmin=jnp.min(gammas_surf).item(),
                vmax=jnp.max(gammas_surf).item(),
            )
            sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Gamma")

    def plot_L_pan(self, ax: Axes3D | None = None) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
        else:
            ax_now = ax
            fig = ax_now.get_figure()
        if fig is None:
            raise ValueError("Axes must be part of a figure")
        self.plot_panels(ax_now)

        # Plot the L_pan by coloring the panels
        L_pan = self.L_pan

        for i in np.arange(0, self.NM):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_now.plot_surface(xs, ys, zs, color=coolwarm(L_pan[i]), alpha=0.9)

        # Add colorbar
        norm = Normalize(vmin=jnp.min(L_pan).item(), vmax=jnp.max(L_pan).item())
        sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_now, label="L_pan")

    def plot_D_pan(self, ax: Axes3D | None = None) -> None:
        if ax is None:
            fig: Figure | None = plt.figure()
            ax_now: Axes3D = fig.add_subplot(projection="3d")  # type: ignore
        else:
            ax_now = ax
            fig = ax_now.get_figure()

        if fig is None:
            raise ValueError("Axes must be part of a figure")

        self.plot_panels(ax_now)

        # Plot the D_pan by coloring the panels
        D_pan = self.D_pan

        for i in np.arange(0, self.NM):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_now.plot_surface(xs, ys, zs, color=coolwarm(D_pan[i]), alpha=0.9)

        # Add colorbar
        norm = Normalize(vmin=jnp.min(D_pan).item(), vmax=jnp.max(D_pan).item())
        sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_now, label="D_pan")
