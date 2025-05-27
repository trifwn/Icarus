from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from jaxtyping import Int
from matplotlib import colormaps
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from pandas import DataFrame
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

from ICARUS.core.types import FloatArray

from . import StripLoads
from .lspt_surface import LSPTSurface

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
        self.surfaces: Sequence[LSPTSurface] = []
        self.surface_dict: dict[str, Any] = {}

        # Get the wing segments
        panel_index = 0
        grid_index = 0
        for i, surf in enumerate(plane.wings):
            for segment in surf.wing_segments:
                lspt_surface = LSPTSurface(
                    surface=segment,
                    panel_index=panel_index,
                    grid_index= grid_index,
                    wing_id=i,
                )
                self.surfaces.append(lspt_surface)

                panel_index += lspt_surface.num_panels
                grid_index += lspt_surface.num_grid_points

        # Plane properties
        self.S: float = plane.S
        self.CG: FloatArray = plane.CG
        self.MAC: float = plane.mean_aerodynamic_chord

        num_panels = self.num_panels
        num_grid_points = self.num_grid_points

        # Flattened Grid
        self.grid = jnp.zeros((num_grid_points, 3))
        # Panels are defined by 4 points in 3D space
        self.panels = jnp.zeros((num_panels, 4, 3))
        # Each panel has a circulation gamma
        self.gammas = jnp.zeros(num_panels)

        # Control points are the Quarter Chord of the panels
        self.control_points = jnp.zeros((num_panels, 3))
        # Normal vector at the control points
        self.control_nj = jnp.zeros((num_panels, 3))

        # Influence matrices
        self.A = jnp.zeros((num_panels, num_panels))
        self.A_star = jnp.zeros((num_panels, num_panels))

        self.strip_data: list[StripLoads] = []
        for surf in self.surfaces:
            # Set Panels
            self.panels = self.panels.at[surf.panel_indices, :, :].set(surf.panels)

            # Set Control Points and Normal Vectors
            self.control_points = self.control_points.at[surf.panel_indices, :].set(surf.control_points)
            self.control_nj = self.control_nj.at[surf.panel_indices, :].set(surf.control_nj)

            # Set Grid Points
            self.grid = self.grid.at[surf.grid_indices, :].set(surf.grid)

        self.strip_data: list[StripLoads] = []
        for surf in self.surfaces:
            spans = surf.span_positions
            chords = surf.chords
            # Get all the strips by their panel indices
            start_idx = surf.panel_indices[0]
            for i in range(surf.N - 1):
                strip_idxs = jnp.arange(i * (surf.M - 1), (i + 1) * (surf.M - 1)) + start_idx

                self.strip_data.append(
                    StripLoads(
                        panel_idxs=strip_idxs,
                        panels=self.panels[strip_idxs, :, :],
                        chord=(chords[i] + chords[i + 1]) / 2,
                        width=spans[i + 1] - spans[i],
                    ),
                )

        # Get the wake shedding indices
        self.wake_shedding_panel_indices: Int[Array, ...] = jnp.concatenate(
            [surface.wake_shedding_panel_indices for surface in self.surfaces],
        )

        for surface in self.surfaces:
            # Add the near wake panels
            surface.add_near_wake_panels()

            # Add the flat wake panels
            surface.add_flat_wake_panels()

        self.num_near_wake_panels: int = len(self.near_wake_indices)

    @property 
    def near_wake_indices(self) -> Int[Array, ...]:
        """Indices of the near wake panels across all surfaces."""

        return jnp.concatenate(
            [surface.near_wake_panel_indices for surface in self.surfaces],
        )
    
    @property
    def flat_wake_panel_indices(self) -> Int[Array, ...]:
        """Indices of the flat wake panels across all surfaces."""
        return jnp.concatenate(
            [surface.flat_wake_panel_indices for surface in self.surfaces],
        )

    @property
    def num_panels(self) -> int:
        """Total number of panels in the LSPT plane."""
        return sum(surface.num_panels for surface in self.surfaces)

    @property
    def num_grid_points(self) -> int:
        """Total number of grid points in the LSPT plane."""
        return sum(surface.num_grid_points for surface in self.surfaces)

    @property
    def num_strips(self) -> int:
        """Total number of strips in the LSPT plane."""
        return sum(surface.num_strips for surface in self.surfaces)

    @property
    def lifting_surfaces(self) -> list[LSPTSurface]:
        """List of lifting surfaces in the LSPT plane."""
        return [surface for surface in self.surfaces if surface.is_lifting]

    def factorize_system(self) -> tuple[Array, Array, Array]:
        """Factorize the VLM system matrices using LU decomposition.

        Returns:
            Tuple of (A_LU, A_piv, A_star) for efficient solving
        """
        import jax

        from ICARUS.aero.vlm import get_LHS

        A, A_star = get_LHS(self)
        # Perform LU decomposition on A
        A_LU, A_piv = jax.scipy.linalg.lu_factor(A)

        # Store the factorized matrices
        self.A_LU = A_LU
        self.A_piv = A_piv
        self.A_star = A_star

        return A_LU, A_piv, A_star

    def aseq(self, angles: FloatArray | list[float], state: State) -> DataFrame:
        """Run angle sequence analysis using VLM with JAX acceleration.

        Args:
            angles: List or array of angles of attack in degrees
            state: Flight state containing environment data

        Returns:
            DataFrame containing results for each angle
        """
        import jax
        import jax.numpy as jnp
        import pandas as pd

        from ICARUS.aero.post_process import get_potential_loads
        from ICARUS.aero.vlm import get_RHS

        # Factorize system matrices once
        A_LU, A_piv, A_star = self.factorize_system()
        umag = state.u_freestream

        # Initialize result arrays
        Ls = jnp.zeros(len(angles))
        Ds = jnp.zeros(len(angles))
        Mys = jnp.zeros(len(angles))
        CLs = jnp.zeros(len(angles))
        CDs = jnp.zeros(len(angles))
        Cms = jnp.zeros(len(angles))
        Ls_2D = jnp.zeros(len(angles))
        Ds_2D = jnp.zeros(len(angles))
        Mys_2D = jnp.zeros(len(angles))

        for i, aoa in enumerate(angles):

            self.alpha = aoa * jnp.pi / 180
            self.beta = 0
            # self.move_wake_panels(alpha = self.alpha, beta=self.beta)

            # Calculate freestream velocity components
            Uinf = umag * jnp.cos(self.alpha) * jnp.cos(self.beta)
            Vinf = umag * jnp.cos(self.alpha) * jnp.sin(self.beta)
            Winf = umag * jnp.sin(self.alpha) * jnp.cos(self.beta)

            Q = jnp.array((Uinf, Vinf, Winf))
            RHS = get_RHS(self, Q)

            # Solve for circulations using factorized system
            gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
            self.gammas = gammas
            w = jnp.matmul(A_star, gammas)

            # Distribute gamma calculations to strips
            for strip in self.strip_data:
                strip_idxs = strip.panel_idxs
                strip.gammas = gammas[strip_idxs]
                strip.w_induced = w[strip_idxs]
                strip.calc_mean_values()

            # Calculate potential loads
            (L, D, D2, Mx, My, Mz, CL, CD, Cm, L_pan, D_pan) = get_potential_loads(
                plane=self,
                state=state,
                ws=w,
                gammas=gammas,
            )

            # Store panel loads
            self.L_pan = L_pan
            self.D_pan = D_pan

            # Store results
            Ls = Ls.at[i].set(L)
            Ds = Ds.at[i].set(D)
            Mys = Mys.at[i].set(My)
            CLs = CLs.at[i].set(CL)
            CDs = CDs.at[i].set(CD)
            Cms = Cms.at[i].set(Cm)

            # Apply symmetry factor if needed
            if True:  # Assuming symmetric wing
                Ls = Ls.at[i].set(2 * Ls[i])
                Ds = Ds.at[i].set(2 * Ds[i])
                Mys = Mys.at[i].set(2 * Mys[i])
                CLs = CLs.at[i].set(2 * CLs[i])
                CDs = CDs.at[i].set(2 * CDs[i])
                Cms = Cms.at[i].set(2 * Cms[i])

        # Create results DataFrame
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

    def plot_panels(
        self,
        ax: Axes3D | None = None,
        plot_wake: bool = False,
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
        for i in np.arange(0, self.num_panels):
            p1, p3, p4, p2 = self.panels[i, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
            ax_.plot_wireframe(xs, ys, zs, linewidth=1.5)

        if plot_wake:
            # Add the near wake panels in green
            near_wake_indices = self.near_wake_indices
            for i in near_wake_indices:
                p1, p3, p4, p2 = self.panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
                ax_.plot_wireframe(xs, ys, zs, color="g", linewidth=1.5)
            
            ax_.scatter([], [], [], color="g", label="Near Wake Panels")

            # Add the flat wake panels in orange
            flat_wake_indices = self.flat_wake_panel_indices
            for i in flat_wake_indices:
                p1, p3, p4, p2 = self.panels[i, :, :]
                xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
                ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
                zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
                ax_.plot_wireframe(xs, ys, zs, color="orange", linewidth=1.5)

            ax_.scatter([], [], [], color="orange", label="Flat Wake Panels")

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
        surf_control_points = self.control_points[: self.num_panels, :]
        ax_.scatter(
            *surf_control_points.T,
            color="r",
            label="Control Points",
            marker="o",
            s=20,
        )
        surf_grid_points = self.grid[: self.num_grid_points, :]
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

        ax_.legend()
        fig.show()
