from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.wing_segment import Wing_Segment


class PotentialWing:
    """
    ! This class is an example of a potential flow wing solver.
    ! It has to be optimized and generalized to be used in the code.
    """

    def __init__(
        self,
        wing_segments: list[Wing_Segment],
        alpha: float,
        beta: float = 0,
        ground_clearence: float = 5,
        wake_geom_type: str = "TE-Geometrical",
    ) -> None:
        self.wing_segments: list[Wing_Segment] = wing_segments
        self.ground_effect_dist: float = ground_clearence
        self._alpha: float = alpha
        self._beta: float = beta
        self.wake_geom_type: str = wake_geom_type

        self.N: int = 0
        self._wing_segments: list[Wing_Segment] = wing_segments
        self.M: int = wing_segments[0].M
        for segment in wing_segments:
            self.N += segment.N
            if segment.M != self.M:
                raise ValueError("All wing segments must have the same number of chordwise panels")

        # Calculate the wing area
        self.S: float = 0
        for wing_segment in self.wing_segments:
            self.S += wing_segment.S

        # Create the span distribution
        span_dist = []
        for segment in wing_segments:
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
            wake_dist = np.repeat(5 * self.max_chord, segment.N)
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

                control_points[i, j, 0] = (self.grid[i, j][0] + self.grid[i + 1, j][0]) / 2 + 3 / 4 * (
                    (self.grid[i, j + 1][0] + self.grid[i + 1, j + 1][0]) / 2
                    - (self.grid[i, j][0] + self.grid[i + 1, j][0]) / 2
                )
                control_points[i, j, 1] = (self.grid[i, j][1] + self.grid[i + 1, j][1]) / 2 + 1 / 2 * (
                    (self.grid[i, j + 1][1] + self.grid[i + 1, j + 1][1]) / 2
                    - (self.grid[i, j][1] + self.grid[i + 1, j][1]) / 2
                )
                control_points[i, j, 2] = (self.grid[i, j][2] + self.grid[i + 1, j][2]) / 2 + 1 / 2 * (
                    (self.grid[i, j + 1][2] + self.grid[i + 1, j + 1][2]) / 2
                    - (self.grid[i, j][2] + self.grid[i + 1, j][2]) / 2
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

    def plot_grid(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        dehidral_prev = 0
        offset_prev = 0
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

        for i in np.arange(0, self.N - 1):
            for j in np.arange(0, self.M):
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
        b_np = np.zeros(((self.N - 1) * (self.M - 1), (self.N - 1) * (self.M - 1)))

        for i in np.arange(0, (self.N - 1) * (self.M)):
            lp, kp = divmod(i, (self.M))
            if kp == self.M - 1:
                a_np[i, i] = 1
                a_np[i, i - 1] = -1
                continue

            for j in np.arange(0, (self.N - 1) * (self.M)):
                l, k = divmod(j, (self.M))
                if k == self.M - 1:
                    U, Ustar = solve_fun(
                        self.control_points[lp, kp, 0],
                        self.control_points[lp, kp, 1],
                        self.control_points[lp, kp, 2],
                        l,
                        k,
                        self.grid,
                    )
                else:
                    U, Ustar = solve_fun(
                        self.control_points[lp, kp, 0],
                        self.control_points[lp, kp, 1],
                        self.control_points[lp, kp, 2],
                        l,
                        k,
                        self.grid,
                    )

                    l1, k1 = divmod(i, (self.M))
                    l2, k2 = divmod(j, (self.M))
                    try:
                        b_np[l1 * (self.M) - lp + k1, l2 * (self.M) - l + k2] = np.dot(Ustar, self.control_nj[lp, kp])
                    except IndexError as error:
                        print(f"i:{i}\tj:{j}\tl1:{l1}\tk1:{k1}\tl2:{l2}\tk2:{k2}")
                        raise IndexError(error)
                a_np[i, j] = np.dot(U, self.control_nj[lp, kp])
        return a_np, b_np

    def solve_wing_panels(self, Q, solve_fun):
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

    def induced_vel_calc(self, fun, i, j, gammas):
        Us = 0
        Uss = 0
        for l in np.arange(0, self.N):
            for k in np.arange(0, self.M - 1):
                U, Ustar = fun(
                    self.control_points[i, j, 0],
                    self.control_points[i, j, 1],
                    self.control_points[i, j, 2],
                    l,
                    k,
                    self.grid,
                    gamma=gammas[l, k],
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
        self.gammas_mat = gammas.reshape((self.N - 1, self.M))[:, :-1]
        self.gammas = self.gammas_mat.reshape((self.N - 1) * (self.M - 1))
        self.w = np.matmul(self.b_np, self.gammas)
        self.w_mat = self.w.reshape((self.N - 1, self.M - 1))

    def plot_gamma_distribution(self):
        if self.gammas_mat is None:
            self.get_gamma_distribution()

        fig = plt.figure()
        ax = fig.add_subplot()
        # ax.matshow(self.gammas_mat)
        ax.set_title("Gamma Distribution")
        ax.set_xlabel("Chordwise Panels")
        ax.set_ylabel("Spanwise Panels")
        fig.colorbar(ax.matshow(self.gammas_mat))
        fig.show()

    def get_aerodynamic_loads(self, dens, umag):
        if self.gammas is None:
            self.get_gamma_distribution()

        L_pan = np.zeros((self.N, self.M - 1))
        D_pan = np.zeros((self.N, self.M - 1))
        for i in np.arange(0, self.N - 1):
            dx = self.grid[i, 1, 0] - self.grid[i, 0, 0]
            for j in np.arange(0, self.M - 2):
                dy = self.grid[i + 1, j, 1] - self.grid[i, j, 1]
                if j == 0:
                    g = self.gammas_mat[i, j + 1] - self.gammas_mat[i, j]
                else:
                    g = self.gammas_mat[i, j] - self.gammas_mat[i, j - 1]
                L_pan[i, j] = dens * umag * dy * g
                D_pan[i, j] = -0.5 * dens * self.w_mat[i, j] * dy * g
                # _, Ui = pot_wing.induced_vel_calc(symm_wing,i,j,Gammas_mat)
                # D_pan[i,j] = -dens * dy * Umag * g * alfi
            # L_pan[i,:] *= dx
            # D_pan[i,:] *= dx
        L = np.sum(L_pan)
        D = np.sum(D_pan)
        self.L_pan = L_pan
        self.D_pan = D_pan
        print(f"Using No Penetration\nL:{L}\t|\tD:{D}")
        print(f"CL:{2*L/(dens*(umag**2)*self.S)}\t|\tCD_ind:{2*D/(dens*(umag**2)*self.S)}")

    def calculate_induced_velocities(self):
        if self.w_mat is None:
            self.get_gamma_distribution()
        self.w_induced_strips = np.mean(self.w_mat, axis=1)

    def plot_w_induced_strips(
        self,
    ):
        if self.w_induced_strips is None:
            self.calculate_induced_velocities()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.span_dist[:-1], self.w_induced_strips)
        ax.set_title("Induced Velocity")
        ax.set_xlabel("Span")
        ax.set_ylabel("Induced Velocity")
        fig.show()

    def plot_lift_drag_strips(
        self,
    ):
        if self.L_pan is None:
            self.get_aerodynamic_loads(1.225, 20)

        L = np.sum(self.L_pan)
        D = np.sum(self.D_pan)

        fig = plt.figure()
        ax = fig.subplots(1, 2)

        ax[0].bar(self.span_dist[:-2], self.L_pan[:-2, 0])
        ax[0].set_title("Lift Distribution")
        ax[0].set_xlabel("Span")
        ax[0].set_ylabel("Lift")

        ax[1].bar(self.span_dist[:-2], self.D_pan[:-2, 0])
        ax[1].set_title("Drag Distribution")
        ax[1].set_xlabel("Span")
        ax[1].set_ylabel("Drag")
        fig.show()

    def plot_lift_drag_panels(self):
        if self.L_pan is None:
            self.get_aerodynamic_loads(1.225, 20)

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

    def aseq(self, aoa_min, aoa_max, aoa_step, umag, dens, solver_fun):
        aoa = np.arange(aoa_min, aoa_max, aoa_step)
        CL = np.zeros(len(aoa))
        CD = np.zeros(len(aoa))

        for i in np.arange(0, len(aoa)):
            self.alpha = aoa[i] * np.pi / 180

            Uinf = umag * np.cos(self.alpha) * np.cos(self.beta)
            Vinf = umag * np.cos(self.alpha) * np.sin(self.beta)
            Winf = umag * np.sin(self.alpha) * np.cos(self.beta)
            Q = np.array((Uinf, Vinf, Winf))

            self.solve_wing_panels(Q, solver_fun)
            self.get_gamma_distribution()
            self.get_aerodynamic_loads(dens, umag)
            CL[i] = 2 * np.sum(self.L_pan) / (dens * (umag**2) * self.S)
            CD[i] = 2 * np.sum(self.D_pan) / (dens * (umag**2) * self.S)
        return aoa, CL, CD
