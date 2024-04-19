import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_lift_drag_strips(
    self,
) -> None:
    if self.L_pan is None:
        self.get_aerodynamic_loads(umag=20)

    fig = plt.figure()
    ax: list[Axes] = fig.subplots(1, 2)

    if not isinstance(ax, list):
        ax = ax.flatten()

    ax[0].bar(self.span_dist[1:], np.mean(self.L_pan, axis=1))
    ax[0].set_title("Lift Distribution")
    ax[0].set_xlabel("Span")
    ax[0].set_ylabel("Lift")

    ax[1].bar(self.span_dist[1:], np.mean(self.D_pan, axis=1))
    ax[1].set_title("Drag Distribution")
    ax[1].set_xlabel("Span")
    ax[1].set_ylabel("Drag")
    fig.show()

    def plot_lift_drag_panels(self, umag: float) -> None:
        if self.L_pan is None:
            self.get_aerodynamic_loads(umag=umag)

        fig: Figure = plt.figure()
        ax = fig.subplots(1, 2)

        if not isinstance(ax, list):
            raise ValueError("Expected a ndarray of Axes")

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


def plot_w_induced_strips(
    self,
    umag: float,
) -> None:
    if self.w_induced_strips is None:
        self.calculate_strip_induced_velocities()
    fig, ax = plt.subplots(3)
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


def plot_gamma_distribution(self) -> None:
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


def plot_grid(self, show_wake: bool = False, show_airfoils: bool = False) -> None:
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore

    dehidral_prev = 0
    offset_prev = 0
    if show_airfoils:
        for wing_seg in self.wing_segments:
            for i in np.arange(0, wing_seg.N - 1):
                ax.plot(
                    wing_seg.root_airfoil._x_upper * wing_seg._chord_dist[i] + wing_seg._xoffset_dist[i] + offset_prev,
                    np.repeat(wing_seg.grid[i, 0, 1], len(wing_seg.root_airfoil._y_upper)),
                    wing_seg.root_airfoil._y_upper + wing_seg._zoffset_dist[i] + dehidral_prev,
                    "-",
                    color="red",
                )

                ax.plot(
                    wing_seg.root_airfoil._x_lower * wing_seg._chord_dist[i] + wing_seg._xoffset_dist[i] + offset_prev,
                    np.repeat(wing_seg.grid[i, 0, 1], len(wing_seg.root_airfoil._y_lower)),
                    wing_seg.root_airfoil._y_lower + wing_seg._zoffset_dist[i] + dehidral_prev,
                    "-",
                    color="red",
                )
            dehidral_prev += wing_seg._zoffset_dist[-1]
            offset_prev += wing_seg._xoffset_dist[-1]
    if show_wake:
        M: int = self.M
    else:
        M = self.M - 1
    for i in np.arange(0, self.N - 1):
        for j in np.arange(0, M):
            p1, p3, p4, p2 = self.panels[i, j, :, :]
            xs = np.reshape(np.array([p1[0], p2[0], p3[0], p4[0]]), (2, 2))
            ys = np.reshape(np.array([p1[1], p2[1], p3[1], p4[1]]), (2, 2))
            zs = np.reshape(np.array([p1[2], p2[2], p3[2], p4[2]]), (2, 2))
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
