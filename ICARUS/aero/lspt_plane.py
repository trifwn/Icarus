from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from pandas import DataFrame
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

from ICARUS.core.types import FloatArray

from . import StripLoads
from .lspt_surface import LSPTSurface


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
        self.strip_data: list[StripLoads] = []

        # Plane properties
        self.S: float = plane.S
        self.CG: FloatArray = plane.CG
        self.MAC: float = plane.mean_aerodynamic_chord

        # Get the wing segments
        panel_index = 0
        grid_index = 0
        for i, surf in enumerate(plane.wings):
            for segment in surf.wing_segments:
                lspt_surface = LSPTSurface(
                    surface=segment,
                    wing_id=i,
                )
                self.surfaces.append(lspt_surface)

                panel_index += lspt_surface.num_panels
                grid_index += lspt_surface.num_grid_points

        for surface in self.surfaces:
            # Add the near wake panels
            surface.add_near_wake_panels()
            # Add the flat wake panels
            surface.add_flat_wake_panels()

        num_panels = self.num_panels + self.num_near_wake_panels + self.num_flat_wake_panels
        num_grid_points = self.num_grid_points

        # Flattened Grid
        self.grid = jnp.zeros((num_grid_points, 3))
        # Panels are defined by 4 points in 3D space
        self.panels = jnp.zeros((num_panels, 4, 3))
        # Control points are the Quarter Chord of the panels
        self.panel_cps = jnp.zeros((num_panels, 3))
        # Normal vector at the control points
        self.panel_normals = jnp.zeros((num_panels, 3))
        # Each panel has a circulation gamma
        self.gammas = jnp.zeros(num_panels)

        surf_panel_index = 0
        surf_grid_index = 0
        for surf in self.surfaces:
            # Indices for the current surface
            surf_panel_indices = surf_panel_index + jnp.arange(surf.num_panels)
            near_wake_indices = surf_panel_indices[-1] + jnp.arange(surf.num_near_wake_panels)
            flat_wake_indices = near_wake_indices[-1] + jnp.arange(surf.num_flat_wake_panels)

            # Set Panels
            self.panels = self.panels.at[surf_panel_indices, :, :].set(surf.panels)
            self.panels = self.panels.at[near_wake_indices, :, :].set(surf.near_wake_panels)
            self.panels = self.panels.at[flat_wake_indices, :, :].set(surf.flat_wake_panels)

            # Set Control Points
            self.panel_cps = self.panel_cps.at[surf_panel_indices, :].set(surf.panel_cps)
            self.panel_cps = self.panel_cps.at[near_wake_indices, :].set(surf.near_wake_panel_cps)
            self.panel_cps = self.panel_cps.at[flat_wake_indices, :].set(surf.flat_wake_panel_cps)

            # Set Control Normals
            self.panel_normals = self.panel_normals.at[surf_panel_indices, :].set(surf.panel_normals)
            self.panel_normals = self.panel_normals.at[near_wake_indices, :].set(surf.near_wake_panel_normals)
            self.panel_normals = self.panel_normals.at[flat_wake_indices, :].set(surf.flat_wake_panel_normals)

            # Set Grid Points
            grid_indices = surf_grid_index + jnp.arange(surf.num_grid_points)
            self.grid = self.grid.at[grid_indices, :].set(surf.grid)

            # Update indices for next surface
            surf_panel_index += surf.num_panels + surf.num_near_wake_panels + surf.num_flat_wake_panels
            surf_grid_index += surf.num_grid_points

        # Influence matrices
        self.A = jnp.zeros((num_panels, num_panels))
        self.A_star = jnp.zeros((num_panels, num_panels))

        surf_panel_index = 0 
        for surf in self.surfaces:
            spans = surf.span_positions
            chords = surf.chords
            # Get all the strips by their panel indices
            for i in range(surf.N - 1):
                strip_idxs = surf_panel_index + jnp.arange(i * (surf.M - 1), (i + 1) * (surf.M - 1)) 

                self.strip_data.append(
                    StripLoads(
                        panel_idxs=strip_idxs,
                        panels=self.panels[strip_idxs, :, :],
                        chord=(chords[i] + chords[i + 1]) / 2,
                        width=spans[i + 1] - spans[i],
                    ),
                )
            
            surf_panel_index += surf.num_panels + surf.num_near_wake_panels + surf.num_flat_wake_panels

    @property
    def num_panels(self) -> int:
        """Total number of panels in the LSPT plane."""
        return sum(surface.num_panels for surface in self.surfaces)

    @property
    def num_near_wake_panels(self) -> int:
        """Total number of near wake panels in the LSPT plane."""
        return sum(surface.num_near_wake_panels for surface in self.surfaces)

    @property
    def num_flat_wake_panels(self) -> int:
        """Total number of flat wake panels in the LSPT plane."""
        return sum(surface.num_flat_wake_panels for surface in self.surfaces)

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

        for surf in self.surfaces:
            surf.plot_panels(
                ax=ax_,
                plot_wake=plot_wake,
                legend=False,
            )

        if plot_wake:
            ax_.scatter([], [], [], color="orange", label="Flat Wake Panels")
            ax_.scatter([], [], [], color="g", label="Near Wake Panels")
        ax_.scatter([], [], [], color="b", marker="x", label="Wake Shedding Panels")
        ax_.scatter([], [], [], color="r", marker="o", label="Control Points")
        ax_.scatter([], [], [], color="k", marker="x", label="Grid Points")

        ax_.set_title("Grid")
        ax_.set_xlabel("x")
        ax_.set_ylabel("y")
        ax_.set_zlabel("z")
        ax_.axis("equal")
        ax_.view_init(30, 150)

        ax_.legend()
        fig.show()
