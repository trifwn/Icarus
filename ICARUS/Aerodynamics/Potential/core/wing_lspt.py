from re import I
from tabnanny import verbose
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
import pandas as pd
from ICARUS.Enviroment.definition import Environment
from ICARUS.Airfoils.airfoilD import AirfoilD

from ICARUS.Core.types import FloatArray
from ICARUS.Database.db import DB
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing_segment import Wing_Segment


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
        db: DB,
        plane: Airplane,
        environment: Environment,
        alpha: float,
        beta: float = 0,
        ground_clearence: float = 5,
        wake_geom_type: str = "TE-Geometrical",
    ) -> None:
        # Store the database
        self.db: DB = db

        # Get the environment properties
        self.dens: float = environment.air_density
        self.visc: float = environment.air_dynamic_viscosity

        # Store the wing segments
        self.wing_segments: list[Wing_Segment] = plane.surfaces

        # Add the ground effect distance ! NOT IMPLEMENTED AS OF NOW
        self.ground_effect_dist: float = ground_clearence

        # Get the angle of attack and sideslip
        self._alpha: float = alpha
        self._beta: float = beta

        # Store the wake geometry type
        self.wake_geom_type: str = wake_geom_type

        self.N: int = 0
        self._wing_segments: list[Wing_Segment] = plane.surfaces
        self.M: int = plane.surfaces[0].M
        self.is_symmetric: bool = True

        for segment in plane.surfaces:
            if not segment.is_symmetric:
                self.is_symmetric = False
            self.N += segment.N
            if segment.M != self.M:
                raise ValueError("All wing segments must have the same number of chordwise panels")

        # Calculate the wing area
        self.S: float = plane.S

        self.cog = plane.CG

        self.MAC: float = plane.mean_aerodynamic_chord

        # Create the span distribution
        span_dist = []
        for segment in plane.surfaces:
            span_dist.append(segment._span_dist + segment.origin[1])
        self.span_dist = np.concatenate(span_dist)

        self.grid: ndarray[Any, dtype[floating]] = np.empty((self.N, self.M + 1, 3))

        # find maximum chord to set wake distance
        max_chord = 0
        for wing_segment in self.wing_segments:
            if np.max(wing_segment._chord_dist) > max_chord:
                max_chord = np.max(wing_segment._chord_dist)
        self.max_chord = max_chord

        # Get the angle of the trailing edge of each wing segment
        self.te_angle_dist = []
        for wing_segment in self.wing_segments:
            airfoil = wing_segment.airfoil
            # The trailing edge is the last point of the airfoil
            # We will get the angle of the trailing edge by getting numerical derivative
            # of the airfoil coordinates.
            # We will use the last 3 points to get the derivative
            x = airfoil._x_lower[-3:]
            y: ndarray[Any, dtype[Any]] = airfoil.camber_line(x)
            dydx = np.repeat(np.gradient(y, x), wing_segment.N)
            self.te_angle_dist.append(np.arctan(dydx))
        self.te_angle_dist = np.concatenate(self.te_angle_dist)

        # Create the grid
        N_start = 0
        for segment in self.wing_segments:
            N_end: int = N_start + segment.N
            self.grid[N_start:N_end, :-1, :] = segment.grid
            N_start: int = N_end + 0

        # THIS CALCULATIONS DEPEND ON THE ORIENTATION OF THE INFLOW
        self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)
        self.delete_middle_panels()

        # Define the variables that will be used in the solver
        self.a_np: ndarray[Any, dtype[floating]] = None  # type: ignore
        self.b_np: ndarray[Any, dtype[floating]] = None  # type: ignore
        self.RHS_np: ndarray[Any, dtype[floating]] = None  # type: ignore

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        if self.wake_geom_type != "TE-Geometrical":
            self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        if self.wake_geom_type != "TE-Geometrical":
            self.make_wake_grid_points()
        self.make_nj()
        (self.panels, self.control_points, self.control_nj) = self.grid_to_panels(self.grid)

    def make_nj(
        self,
    ) -> None:
        self.nj = np.repeat(
            [
                (
                    np.sin(self.alpha) * np.cos(self.beta),
                    np.cos(self.alpha) * np.sin(self.beta),
                    np.cos(self.alpha) * np.cos(self.beta),
                ),
            ],
            (self.N - 1) * (self.M - 1),
            axis=0,
        )

    def make_wake_grid_points(
        self,
    ) -> None:
        N_start = 0
        for segment in self.wing_segments:
            N_end: int = N_start + segment.N

            # Create the wake grid points
            wake_dist = np.repeat(100 * self.max_chord, segment.N)
            self.grid[N_start:N_end, -1, 0] = wake_dist
            self.grid[N_start:N_end, -1, 1] = segment._span_dist + segment.origin[1]

            if self.wake_geom_type == "Inflow-TE":
                self.grid[N_start:N_end, -1, 2] = self.grid[N_start:N_end, -2, 2] + (
                    wake_dist - self.grid[N_start:N_end, -2, 0]
                ) * np.tan(self.alpha)
            elif self.wake_geom_type == "Inflow-Uniform":
                self.grid[N_start:N_end, -1, 2] = wake_dist * np.tan(self.alpha)
            elif self.wake_geom_type == "TE-Geometrical":
                self.grid[N_start:N_end, -1, 2] = self.grid[N_start:N_end, -2, 2] + (
                    wake_dist - self.grid[N_start:N_end, -2, 0]
                ) * np.tan(self.te_angle_dist[N_start:N_end])
            else:
                raise ValueError("Invalid wake geometry type")
            N_start: int = N_end + 0

    def grid_to_panels(self, grid: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Convert Grid to Panels

        Args:
            grid (FloatArray): Grid to convert

        Returns:
            FloatArray: Panels
        """
        panels: FloatArray = np.empty((self.N - 1, self.M, 4, 3), dtype=float)
        control_points: FloatArray = np.empty((self.N - 1, self.M, 3), dtype=float)
        control_nj: FloatArray = np.empty((self.N - 1, self.M, 3), dtype=float)

        # We need to remove all panels that are in between wing segments
        # and create new panels for the wake
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M):
                panels[i, j, 0, :] = grid[i + 1, j]
                panels[i, j, 1, :] = grid[i, j]
                panels[i, j, 2, :] = grid[i, j + 1]
                panels[i, j, 3, :] = grid[i + 1, j + 1]

                control_points[i, j, :] = (self.grid[i, j] + self.grid[i + 1, j]) / 2 + 3 / 4 * (
                    (self.grid[i, j + 1] + self.grid[i + 1, j + 1]) / 2 - (self.grid[i, j] + self.grid[i + 1, j]) / 2
                )

                Ak = panels[i, j, 0, :] - panels[i, j, 2, :]
                Bk = panels[i, j, 1, :] - panels[i, j, 3, :]
                cross_prod = np.cross(Ak, Bk)
                control_nj[i, j, :] = cross_prod / np.linalg.norm(cross_prod)

        return panels, control_points, control_nj

    def delete_middle_panels(self) -> None:
        # Remove the panels that are in between wing segments
        Ns_to_delete = []
        N = 0
        for wing_seg in self.wing_segments[:-1]:
            Ns_to_delete.append(N + wing_seg.N - 1)
            N += wing_seg.N

        self.N = self.N - len(Ns_to_delete)

        # Remove every row that has index in Ns_to_delete
        self.panels = np.delete(self.panels, Ns_to_delete, axis=0)
        self.control_points = np.delete(self.control_points, Ns_to_delete, axis=0)
        self.control_nj = np.delete(self.control_nj, Ns_to_delete, axis=0)
        self.grid = np.delete(self.grid, Ns_to_delete, axis=0)
        self.span_dist = np.delete(self.span_dist, Ns_to_delete, axis=0)

    def plot_grid(self, show_wake=False, show_airfoils=False) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        dehidral_prev = 0
        offset_prev = 0
        if show_airfoils:
            for wing_seg in self.wing_segments:
                for i in np.arange(0, wing_seg.N - 1):
                    ax.plot(
                        wing_seg.airfoil._x_upper * wing_seg._chord_dist[i] + wing_seg._offset_dist[i] + offset_prev,
                        np.repeat(wing_seg.grid[i, 0, 1], len(wing_seg.airfoil._y_upper)),
                        wing_seg.airfoil._y_upper + wing_seg._dihedral_dist[i] + dehidral_prev,
                        "-",
                        color="red",
                    )

                    ax.plot(
                        wing_seg.airfoil._x_lower * wing_seg._chord_dist[i] + wing_seg._offset_dist[i] + offset_prev,
                        np.repeat(wing_seg.grid[i, 0, 1], len(wing_seg.airfoil._y_lower)),
                        wing_seg.airfoil._y_lower + wing_seg._dihedral_dist[i] + dehidral_prev,
                        "-",
                        color="red",
                    )
                dehidral_prev += wing_seg._dihedral_dist[-1]
                offset_prev += wing_seg._offset_dist[-1]
        if show_wake:
            M = self.M
        else:
            M = self.M - 1
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, M):
                p1, p3, p4, p2 = self.panels[i, j, :, :]
                xs = np.reshape([p1[0], p2[0], p3[0], p4[0]], (2, 2))
                ys = np.reshape([p1[1], p2[1], p3[1], p4[1]], (2, 2))
                zs = np.reshape([p1[2], p2[2], p3[2], p4[2]], (2, 2))
                ax.scatter(*self.control_points[i, j, :], color="k")
                ax.plot_wireframe(xs, ys, zs, linewidth=0.5)
        ax.set_title("Grid")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.axis("scaled")
        ax.view_init(30, 150)
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        fig.show()

    def get_RHS(self, Q):
        RHS_np = np.zeros((self.N - 1) * (self.M))
        for i in np.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            if kp == self.M - 1:
                RHS_np[i] = 0
            else:
                RHS_np[i] = -np.dot(Q, self.control_nj[lp, kp])
        return RHS_np

    def get_LHS(self, solve_fun):
        a_np = np.zeros(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))
        b_np = np.zeros(((self.N - 1) * (self.M), (self.N - 1) * (self.M)))

        for i in np.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            if kp == self.M - 1:
                a_np[i, i] = 1
                a_np[i, i - 1] = -1
                b_np[i, i] = 1
                b_np[i, i - 1] = -1
                continue

            for j in np.arange(0, (self.N - 1) * (self.M)):
                l, k = divmod(j, (self.M))

                U, Ustar = solve_fun(
                    self.control_points[lp, kp, 0],
                    self.control_points[lp, kp, 1],
                    self.control_points[lp, kp, 2],
                    l,
                    k,
                    self.grid,
                )
                b_np[i, j] = np.dot(Ustar, self.control_nj[lp, kp])
                a_np[i, j] = np.dot(U, self.control_nj[lp, kp])
        return a_np, b_np

    def solve_wing_panels(self, Q, solve_fun):
        self.solve_fun = solve_fun
        RHS_np = self.get_RHS(Q)
        self.RHS_np = RHS_np

        if (self.a_np is not None) and self.wake_geom_type == "TE-Geometrical":
            print(f"Using previous solution for a and b! Be smart")
            return
        else:
            print(f"Solving for a and b")
            if self.wake_geom_type == "TE-Geometrical":
                print("We should be using LU decomposition")
            a_np, b_np = self.get_LHS(solve_fun)
            self.a_np = a_np
            self.b_np = b_np

    def solve_wing_horseshoe(self):
        # ! TODO: IMPLEMENT ACCORDING TO KATZ
        print("Not implemented yet")
        pass

    def solve_wing_singularities(self):
        # ! TODO: IMPLEMENT ACCORDING TO KATZ
        print("Not implemented yet")
        pass

    def induced_vel_calc(self, i, j, gammas_mat):
        Us = 0
        Uss = 0
        for l in np.arange(0, self.N - 1):
            for k in np.arange(0, self.M):
                U, Ustar = self.solve_fun(
                    self.control_points[i, j, 0],
                    self.control_points[i, j, 1],
                    self.control_points[i, j, 2],
                    l,
                    k,
                    self.grid,
                    gamma=gammas_mat[l, k],
                )
                Us = Us + U
                Uss = Uss + Ustar
        return Us, Uss

    def get_gamma_distribution(
        self,
    ):
        if self.a_np is None:
            raise ValueError("You must solve the wing panels first")
        gammas = np.linalg.solve(self.a_np, self.RHS_np)
        self.w = np.matmul(self.b_np, gammas)

        self.gammas_mat = np.zeros((self.N - 1, self.M))
        self.w_mat = np.zeros((self.N - 1, self.M))

        for i in np.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            self.gammas_mat[lp, kp] = gammas[i]
            self.w_mat[lp, kp] = self.w[i]
        self.calculate_strip_induced_velocities()

    def plot_gamma_distribution(self):
        if self.gammas_mat is None:
            self.get_gamma_distribution()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(self.gammas_mat)
        ax.set_title("Gamma Distribution")
        ax.set_xlabel("Chordwise Panels")
        ax.set_ylabel("Spanwise Panels")
        # Add the colorbar
        fig.colorbar(ax.matshow(self.gammas_mat))
        fig.show()

    def get_aerodynamic_loads(self, umag, verbose=True) -> None:
        if self.gammas_mat is None:
            self.get_gamma_distribution()

        L_pan = np.zeros((self.N - 1, self.M))
        D_pan = np.zeros((self.N - 1, self.M))
        D_trefftz = 0
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M):
                dy = self.grid[i + 1, j, 1] - self.grid[i, j, 1]
                if j == 0:
                    g = self.gammas_mat[i, j]
                else:
                    g = self.gammas_mat[i, j] - self.gammas_mat[i, j - 1]
                L_pan[i, j] = self.dens * umag * dy * g
                D_pan[i, j] = -self.dens * dy * g * self.w_mat[i, j]

                if j == self.M - 2:
                    D_trefftz += -self.dens / 2 * dy * self.gammas_mat[i, j] * self.w_mat[i, j]

        # Calculate the torque. The torque is calculated w.r.t. the CG
        # and is the sum of the torques of each panel times the distance
        # from the CG to the control point of each panel
        M = np.array([0, 0, 0], dtype=float)
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                M += L_pan[i, j] * np.cross(self.control_points[i, j, :] - self.cog, self.control_nj[i, j, :])
                M += D_pan[i, j] * np.cross(self.control_points[i, j, :] - self.cog, self.control_nj[i, j, :])
        Mx, My, Mz = M

        self.L_pan = L_pan
        self.D_pan = D_pan

        self.L: np.float64 = np.sum(L_pan)
        self.D: float = D_trefftz  # np.sum(D_pan)
        self.D2: np.float64 = np.sum(D_pan)
        self.Mx: float = Mx
        self.My: float = My
        self.Mz: float = Mz

        self.CL = 2 * self.L / (self.dens * (umag**2) * self.S)
        self.CD = 2 * self.D / (self.dens * (umag**2) * self.S)
        self.Cm = 2 * self.My / (self.dens * (umag**2) * self.S * self.MAC)

        interpolation_success: bool = True
        try:
            self.integrate_polars_from_reynolds(umag)
        except ValueError as e:
            print("\tCould not interpolate polars! Got error:")
            print(f"\t{e}")
            interpolation_success = False

        if verbose:
            print(f"- Angle {self.alpha * 180 /np.pi}")
            print(f"\t--Using no penetration condition:")
            print(f"\t\tL:{self.L}\t|\tD (Trefftz Plane):{self.D}\tD2:{self.D2}\t|\tMy:{self.My}")
            print(f"\t\tCL:{self.CL}\t|\tCD_ind:{self.CD}\t|\tCm:{self.Cm}")
            if interpolation_success:
                print(f"\t--Using 2D polars:")
                print(f"\t\tL:{self.L_2D}\t|\tD:{self.D_2D}\t|\tMy:{self.My_2D}")
                print(f"\t\tCL:{self.CL_2D}\t|\tCD:{self.CD_2D}\t|\tCm:{self.Cm_2D}")

    def calculate_strip_induced_velocities(self) -> None:
        if self.w_mat is None:
            self.get_gamma_distribution()
        self.w_induced_strips = np.mean(self.w_mat, axis=1)

    def calculate_strip_gamma(self) -> None:
        if self.gammas_mat is None:
            self.get_gamma_distribution()
        self.gamma_strips = np.mean(self.gammas_mat, axis=1)

    def plot_w_induced_strips(
        self,
        umag,
    ):
        if self.w_induced_strips is None:
            self.calculate_strip_induced_velocities()
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(self.span_dist[:-1], self.w_induced_strips)
        ax[0].set_title("Induced Velocity [m/s]")
        ax[0].set_xlabel("Span [m]")
        ax[0].set_ylabel("Induced Velocity")
        ax[0].grid()

        # Plot the induced velocity as an angle w.r.t. the freestream velocity
        ax[1].plot(self.span_dist[:-1], np.arctan2(self.w_induced_strips, umag) * 180 / np.pi)
        ax[1].set_title("Induced Angle vs Span w.r.t. Freestream")
        ax[1].set_xlabel("Span [m]")
        ax[1].set_ylabel("Induced Angle [deg]")
        ax[1].grid()

        # Plot the induced gamma
        if self.gamma_strips is None:
            self.calculate_strip_gamma()
        ax[2].plot(self.span_dist[:-1], self.gamma_strips)
        ax[2].set_title("Gamma Distribution")
        ax[2].set_xlabel("Span [m]")
        ax[2].set_ylabel("Gamma")
        ax[2].grid()

        fig.show()

    def plot_lift_drag_strips(
        self,
    ):
        if self.L_pan is None:
            self.get_aerodynamic_loads(umag=20)

        fig = plt.figure()
        ax = fig.subplots(1, 2)

        ax[0].bar(self.span_dist[1:], np.mean(self.L_pan, axis=1))
        ax[0].set_title("Lift Distribution")
        ax[0].set_xlabel("Span")
        ax[0].set_ylabel("Lift")

        ax[1].bar(self.span_dist[1:], np.mean(self.D_pan, axis=1))
        ax[1].set_title("Drag Distribution")
        ax[1].set_xlabel("Span")
        ax[1].set_ylabel("Drag")
        fig.show()

    def plot_lift_drag_panels(self, umag):
        if self.L_pan is None:
            self.get_aerodynamic_loads(umag=umag)

        fig = plt.figure()
        ax = fig.subplots(1, 2)

        ax[0].matshow(self.L_pan)
        ax[0].set_title("Lift Distribution")
        ax[0].set_ylabel("Span")
        ax[0].set_xlabel("Chord")

        ax[1].matshow(self.D_pan)
        ax[1].set_title("Drag Distribution")
        ax[1].set_ylabel("Span")
        ax[1].set_xlabel("Chord")

        fig.colorbar(ax[0].matshow(self.L_pan))
        fig.colorbar(ax[1].matshow(self.D_pan))
        fig.show()

    def aseq(self, angles, umag, solver_fun) -> pd.DataFrame:
        # Using no penetration condition
        CL = np.zeros(len(angles))
        CD = np.zeros(len(angles))
        Cm = np.zeros(len(angles))

        Ls = np.zeros(len(angles))
        Ds = np.zeros(len(angles))
        Mys = np.zeros(len(angles))

        # Using 2D polars
        CL_2D = np.zeros(len(angles))
        CD_2D = np.zeros(len(angles))
        Cm_2D = np.zeros(len(angles))

        Ls_2D = np.zeros(len(angles))
        Ds_2D = np.zeros(len(angles))
        Mys_2D = np.zeros(len(angles))

        for i, aoa in enumerate(angles):
            self.alpha = aoa * np.pi / 180

            Uinf = umag * np.cos(self.alpha) * np.cos(self.beta)
            Vinf = umag * np.cos(self.alpha) * np.sin(self.beta)
            Winf = umag * np.sin(self.alpha) * np.cos(self.beta)
            Q = np.array((Uinf, Vinf, Winf))

            self.solve_wing_panels(Q, solver_fun)
            self.get_gamma_distribution()
            self.get_aerodynamic_loads(umag)

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
                "L": Ls,
                "D": Ds,
                "My": Mys,
                "L_2D": Ls_2D,
                "D_2D": Ds_2D,
                "My_2D": Mys_2D,
                "CL": CL,
                "CD": CD,
                "Cm": Cm,
                "CL_2D": CL_2D,
                "CD_2D": CD_2D,
                "Cm_2D": Cm_2D,
            }
        )
        return df

    def get_strip_chords(self):
        # Get the chord of each strip adding the chordwise distance of each panel
        self.chords = np.zeros(self.N - 1)
        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M - 1):
                self.chords[i] += (self.grid[i, j + 1, :] - self.grid[i, j, :])[0]

    def get_strip_reynolds(self, umag):
        if self.w_induced_strips is None:
            self.calculate_strip_induced_velocities()
        self.get_strip_chords()

        # Get the effective angle of attack of each strip
        self.strip_effective_aoa = np.arctan(self.w_induced_strips / umag) * 180 / np.pi + self.alpha * 180 / np.pi

        # Get the reynolds number of each strip
        strip_vel = np.sqrt(self.w_induced_strips**2 + umag**2)
        self.strip_reynolds = self.dens * strip_vel * self.chords / self.visc

        # Scan all wing segments and get the orientation of each airfoil
        # Match that orientation with the each strip and get the effective aoa
        # That is the angle of attack that the airfoil sees
        self.strip_airfoil_effective_aoa = np.zeros(self.N - 1)
        N: int = 0
        for wing_seg in self.wing_segments:
            for j in np.arange(0, wing_seg.N - 1):
                self.strip_airfoil_effective_aoa[N + j] = self.strip_effective_aoa[N + j] + wing_seg.orientation[0]
            N += wing_seg.N - 1

    def integrate_polars_from_reynolds(self, uinf: float, solver="Xfoil") -> None:
        # self.get_strip_reynolds(20, 1.225, 1.7894e-5)
        # if self.strip_reynolds is None:
        self.strip_CL_2D = np.zeros(self.N - 1)
        self.strip_CD_2D = np.zeros(self.N - 1)
        self.strip_Cm_2D = np.zeros(self.N - 1)

        self.L_2D: float | None = None
        self.D_2D: float | None = None
        self.My_2D: float | None = None

        self.CL_2D: float | None = None
        self.CD_2D: float | None = None
        self.Cm_2D: float | None = None

        self.get_strip_reynolds(uinf)

        N: int = 0
        L: float = 0
        D: float = 0
        My_at_quarter_chord: float = 0
        for wing_seg in self.wing_segments:
            airfoil: AirfoilD = wing_seg.airfoil
            for j in np.arange(0, wing_seg.N - 1):
                dy = np.mean(self.grid[N + j +1, :, 1] - self.grid[N + j, :, 1])

                CL, CD, Cm = self.db.foilsDB.interpolate_polars(
                    reynolds=self.strip_reynolds[N + j],
                    airfoil_name=airfoil.name,
                    aoa=float(self.strip_airfoil_effective_aoa[N + j]),
                    solver=solver,
                )
                self.strip_CL_2D[N + j] = CL
                self.strip_CD_2D[N + j] = CD
                self.strip_Cm_2D[N + j] = Cm

                surface = self.chords[N + j] * dy
                vel_mag = np.sqrt(self.w_induced_strips[N + j] ** 2 + uinf**2)
                dynamic_pressure = 0.5 * self.dens * vel_mag**2

                # "Integrate" the CL and CD of each strip to get the total L, D and My
                L += CL * surface * dynamic_pressure
                D += CD * surface * dynamic_pressure
                My_at_quarter_chord += Cm * surface * dynamic_pressure * self.chords[N + j]

            N += wing_seg.N - 1

        self.L_2D = L
        self.D_2D = D

        # Calculate Total Moment moving the moment from the quarter chord
        # to the cg and then add the moment of the lift and drag
        
        self.My_2D = My_at_quarter_chord - D * self.cog[0] + L * self.cog[0]
    
        self.CL_2D = 2 * self.L_2D / (self.dens * (uinf**2) * self.S)
        self.CD_2D = 2 * self.D_2D / (self.dens * (uinf**2) * self.S)
        self.Cm_2D = 2 * self.My_2D / (self.dens * (uinf**2) * self.S * self.MAC)
