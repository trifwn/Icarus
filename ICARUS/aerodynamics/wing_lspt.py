from typing import Any
from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Array
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.axes import SubplotBase
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.airfoils.airfoil import Airfoil
from ICARUS.database import DB
from ICARUS.environment.definition import Environment
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.surface import WingSurface


class Wing_LSPT:
    """
    Wing Model using the Lifting Surface Potential Theory. The wing
    is divided into panels and the potential flow is solved using
    the no penetration condition. Also, the Trefftz plane is used to
    calculate the induced drag. To calculate forces and moments db polars,
    can also be used.
    """

    def __init__(
        self,
        plane: Airplane,
        environment: Environment,
        alpha: float,
        beta: float = 0,
        ground_clearence: float = 5,
        wake_geom_type: str = "TE-Geometrical",
    ) -> None:
        # Get the environment properties
        self.dens: float = environment.air_density
        self.visc: float = environment.air_dynamic_viscosity

        # Store the wing segments
        self.wing_segments: list[WingSurface] = plane.surfaces

        # Add the ground effect distance
        self.ground_effect_dist: float = ground_clearence

        # Get the angle of attack and sideslip
        self._alpha: float = alpha
        self._beta: float = beta

        # Store the wake geometry type
        self.wake_geom_type: str = wake_geom_type

        self.N: int = 0
        self.M: int = plane.surfaces[0].M
        self.is_symmetric: bool = True

        for segment in plane.surfaces:
            if not segment.is_symmetric_y:
                self.is_symmetric = False
            self.N += segment.N
            if segment.M != self.M:
                raise ValueError("All wing segments must have the same number of chordwise panels")

        # Plane properties
        self.S: float = plane.S
        self.CG = plane.CG
        self.MAC: float = plane.mean_aerodynamic_chord

        # Create the span distribution
        span_dist = []
        for segment in plane.surfaces:
            span_dist.append(segment._span_dist + segment.origin[1])
        self.span_dist = jnp.concatenate(span_dist)

        self.grid: Float[Array, "{self.N} {self.M}+1 3"] = jnp.empty((self.N, self.M + 1, 3))

        # find maximum chord to set wake distance
        max_chord: float = 0.0
        for lifting_surface in self.wing_segments:
            if float(jnp.max(lifting_surface._chord_dist)) > max_chord:
                max_chord = float(jnp.max(lifting_surface._chord_dist))
        self.max_chord: float = max_chord

        # Get the angle of the trailing edge of each wing segment
        te_angle_dist: list[Float[Array, "..."]] = []
        for lifting_surface in self.wing_segments:
            airfoil: Airfoil = lifting_surface.root_airfoil
            # The trailing edge is the last point of the airfoil
            # We will get the angle of the trailing edge by getting numerical derivative
            # of the airfoil coordinates.
            # We will use the last 3 points to get the derivative
            x = airfoil._x_lower[-3:]
            y = airfoil.camber_line(x)
            dydx = jnp.repeat(np.gradient(y, x), lifting_surface.N)
            te_angle_dist.append(jnp.arctan(dydx))
        self.te_angle_dist = jnp.concatenate(te_angle_dist)

        # Create the grid
        N_start: int = 0
        for segment in self.wing_segments:
            N_end: int = N_start + segment.N
            self.grid[N_start:N_end, :-1, :] = segment.grid.reshape(segment.N, segment.M, 3)
            N_start = N_end + 0

        # THIS CALCULATIONS DEPEND ON THE ORIENTATION OF THE INFLOW
        self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)
        self.delete_middle_panels()

        # Define the variables that will be used in the solver
        self.a_np = jnp.empty(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))
        self.b_np = jnp.empty(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))
        self.RHS_np = jnp.empty((self.N - 1) * (self.M))

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value
        if self.wake_geom_type != "TE-Geometrical":
            self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value
        if self.wake_geom_type != "TE-Geometrical":
            self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)

    def calc_strip_chords(self) -> None:
        # Get the chord of each strip adding the chordwise distance of each panel
        self.chords = jnp.zeros(self.N - 1)
        for i in jnp.arange(0, self.N - 1):
            for j in jnp.arange(0, self.M - 1):
                self.chords[i] += (self.grid[i, j + 1, :] - self.grid[i, j, :])[0]

    def make_nj(
        self,
    ) -> None:
        rotation = jnp.array(
            [
                jnp.sin(self.alpha) * jnp.cos(self.beta),
                jnp.cos(self.alpha) * jnp.sin(self.beta),
                jnp.cos(self.alpha) * jnp.cos(self.beta),
            ],
        )
        self.nj = jnp.repeat(
            rotation,
            (self.N - 1) * (self.M - 1),
            axis=0,
        )

    def make_wake_grid_points(
        self,
    ) -> None:
        N_start: int = 0
        for segment in self.wing_segments:
            N_end: int = N_start + segment.N

            # Create the wake grid points
            wake_dist = jnp.repeat(100 * self.max_chord, segment.N)
            self.grid[N_start:N_end, -1, 0] = wake_dist
            self.grid[N_start:N_end, -1, 1] = segment._span_dist + segment.origin[1]

            if self.wake_geom_type == "Inflow-TE":
                self.grid[N_start:N_end, -1, 2] = self.grid[N_start:N_end, -2, 2] + (
                    wake_dist - self.grid[N_start:N_end, -2, 0]
                ) * jnp.tan(self.alpha)
            elif self.wake_geom_type == "Inflow-Uniform":
                self.grid[N_start:N_end, -1, 2] = wake_dist * jnp.tan(self.alpha)
            elif self.wake_geom_type == "TE-Geometrical":
                self.grid[N_start:N_end, -1, 2] = self.grid[N_start:N_end, -2, 2] + (
                    wake_dist - self.grid[N_start:N_end, -2, 0]
                ) * jnp.tan(self.te_angle_dist[N_start:N_end])
            else:
                raise ValueError("Invalid wake geometry type")
            N_start = N_end + 0

    def grid_to_panels(self, grid):
        """
        Convert Grid to Panels

        Args:
            grid (FloatArray): Grid to convert

        Returns:
            FloatArray: Panels
        """
        panels = jnp.empty((self.N - 1, self.M, 4, 3), dtype=float)
        control_points = jnp.empty((self.N - 1, self.M, 3), dtype=float)
        control_nj = jnp.empty((self.N - 1, self.M, 3), dtype=float)

        # Create indices for vectorized operations
        i_indices, j_indices = jnp.meshgrid(jnp.arange(0, self.N - 1), jnp.arange(0, self.M), indexing='ij')

        panels[i_indices, j_indices, 0, :] = grid[i_indices + 1, j_indices]
        panels[i_indices, j_indices, 1, :] = grid[i_indices, j_indices]
        panels[i_indices, j_indices, 2, :] = grid[i_indices, j_indices + 1]
        panels[i_indices, j_indices, 3, :] = grid[i_indices + 1, j_indices + 1]

        control_points[i_indices, j_indices, :] = (
            self.grid[i_indices, j_indices] + self.grid[i_indices + 1, j_indices]
        ) / 2 + 3 / 4 * (
            (self.grid[i_indices, j_indices + 1] + self.grid[i_indices + 1, j_indices + 1]) / 2
            - (self.grid[i_indices, j_indices] + self.grid[i_indices + 1, j_indices]) / 2
        )

        Ak = panels[i_indices, j_indices, 0, :] - panels[i_indices, j_indices, 2, :]
        Bk = panels[i_indices, j_indices, 1, :] - panels[i_indices, j_indices, 3, :]
        cross_prod = jnp.cross(Ak, Bk)
        control_nj[i_indices, j_indices, :] = cross_prod / jnp.linalg.norm(cross_prod, axis=-1, keepdims=True)

        return panels, control_points, control_nj

    def delete_middle_panels(self) -> None:
        # Remove the panels that are in between wing segments
        Ns_to_del: list[int] = []
        N = 0
        for wing_seg in self.wing_segments[:-1]:
            Ns_to_del.append(N + wing_seg.N - 1)
            N += wing_seg.N
        Ns_to_delete = jnp.array(Ns_to_del)

        self.N = self.N - len(Ns_to_delete)

        # Remove every row that has index in Ns_to_delete
        self.panels = jnp.delete(self.panels, Ns_to_delete, axis=0)
        self.control_points = jnp.delete(self.control_points, Ns_to_delete, axis=0)
        self.control_nj = jnp.delete(self.control_nj, Ns_to_delete, axis=0)
        self.grid = jnp.delete(self.grid, Ns_to_delete, axis=0)
        self.span_dist = jnp.delete(self.span_dist, Ns_to_delete, axis=0)

    def get_RHS(self, Q):
        RHS_np = jnp.zeros((self.N - 1) * (self.M))
        for i in jnp.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            if kp == self.M - 1:
                RHS_np[i] = 0
            else:
                RHS_np[i] = -jnp.dot(Q, self.control_nj[lp, kp])
        return RHS_np

    def get_LHS(
        self,
        solve_fun: Callable[..., tuple[Float[Array, '3'], Float[Array, "3"]]],
    ) -> tuple[Float[Array, "n j"], Float[Array, "n j"]]:
        a_np = jnp.zeros(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))
        b_np = jnp.zeros(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))

        for i in jnp.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            if kp == self.M - 1:
                a_np[i, i] = 1
                a_np[i, i - 1] = -1
                b_np[i, i] = 1
                b_np[i, i - 1] = -1
                continue

            for j in jnp.arange(0, (self.N - 1) * (self.M)):
                l, k = divmod(j, (self.M))

                U, Ustar = solve_fun(
                    self.control_points[lp, kp, 0],
                    self.control_points[lp, kp, 1],
                    self.control_points[lp, kp, 2],
                    l,
                    k,
                    self.grid,
                )
                b_np[i, j] = jnp.dot(Ustar, self.control_nj[lp, kp])
                a_np[i, j] = jnp.dot(U, self.control_nj[lp, kp])
        return a_np, b_np

    def solve_wing_horseshoe(self) -> None:
        # ! TODO: IMPLEMENT ACCORDING TO KATZ
        print("Not implemented yet")
        pass

    def solve_wing_singularities(self) -> None:
        # ! TODO: IMPLEMENT ACCORDING TO KATZ
        print("Not implemented yet")
        pass

    def aseq(
        self,
        angles: list[float] | Float[Array, 'n'] | Float[np.ndarray, 'n'],
        umag: float,
        solver_fun: Callable[..., tuple[Float[Array, '3'], Float[Array, "3"]]],
        verbose: bool = False,
        solver2D: str = "Xfoil",
    ) -> pd.DataFrame:
        n = len(angles)
        # Using no penetration condition
        CL = jnp.zeros(len(angles))
        CD = jnp.zeros(len(angles))
        Cm = jnp.zeros(len(angles))

        Ls = jnp.zeros(len(angles))
        Ds = jnp.zeros(len(angles))
        Mys = jnp.zeros(len(angles))

        # Using 2D polars
        CL_2D = jnp.zeros(len(angles))
        CD_2D = jnp.zeros(len(angles))
        Cm_2D = jnp.zeros(len(angles))

        Ls_2D = jnp.zeros(len(angles))
        Ds_2D = jnp.zeros(len(angles))
        Mys_2D = jnp.zeros(len(angles))

        for i, aoa in enumerate(angles):
            self.alpha = aoa * jnp.pi / 180

            Uinf = umag * jnp.cos(self.alpha) * jnp.cos(self.beta)
            Vinf = umag * jnp.cos(self.alpha) * jnp.sin(self.beta)
            Winf = umag * jnp.sin(self.alpha) * jnp.cos(self.beta)
            Q = jnp.array((Uinf, Vinf, Winf))

            self.solve_wing_panels(Q, solver_fun)
            self.get_gamma_distribution()
            get_aerodynamic_loads(umag, verbose=verbose)

            # No pen
            Ls[i] = self.L
            Ds[i] = self.D
            Mys[i] = self.My

            CL[i] = 2 * self.L / (self.dens * (umag**2) * self.S)
            CD[i] = 2 * self.D / (self.dens * (umag**2) * self.S)
            Cm[i] = 2 * self.My / (self.dens * (umag**2) * self.S * self.MAC)

            # 2D polars
            Ls_2D[i] = self.L_2D
            Ds_2D[i] = self.D_2D
            Mys_2D[i] = self.My_2D

            CL_2D[i] = self.CL_2D
            CD_2D[i] = self.CD_2D
            Cm_2D[i] = self.Cm_2D

        if self.is_symmetric:
            CL = 2 * CL
            CD = 2 * CD
            Cm = 2 * Cm
            Ls = 2 * Ls
            Ds = 2 * Ds
            Mys = 2 * Mys

            CL_2D = 2 * CL_2D
            CD_2D = 2 * CD_2D
            Cm_2D = 2 * Cm_2D
            Ls_2D = 2 * Ls_2D
            Ds_2D = 2 * Ds_2D
            Mys_2D = 2 * Mys_2D

        # Create DataFrames with the results
        df = pd.DataFrame(
            {
                "AoA": angles,
                "LSPT Potential Fz": Ls,
                "LSPT Potential Fx": Ds,
                "LSPT Potential My": Mys,
                "LSPT 2D Fz": Ls_2D,
                "LSPT 2D Fx": Ds_2D,
                "LSPT 2D My": Mys_2D,
                # "CL": CL,
                # "CD": CD,
                # "Cm": Cm,
                # "CL_2D": CL_2D,
                # "CD_2D": CD_2D,
                # "Cm_2D": Cm_2D,
            },
        )
        return df
