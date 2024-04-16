import time
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ICARUS.core.types import FloatArray
from ICARUS.mission.trajectory.trajectory import MissionTrajectory


#! TODO: Make Callback
def setup_plot() -> tuple[Figure, list[Axes]]:
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # DUMMY PLOTS
    zeros = np.zeros(100)
    axs[0, 0].plot(zeros, zeros, label="Actual")
    axs[0, 0].plot(zeros, zeros, label="Goal")

    # Plot Distance Travelled
    axs[0, 1].plot(zeros, zeros, label=f"Course")

    # Plot Elevator Angle
    axs[0, 2].plot(zeros, zeros, label=f"Angle")

    # Plot Velocity
    axs[1, 0].plot(zeros, zeros, label=f"Magnitude")
    axs[1, 0].plot(zeros, zeros, label=f"Vx")
    axs[1, 0].plot(zeros, zeros, label=f"Vh")

    # Plot Thrust
    axs[1, 1].plot(zeros, zeros, label=f"Required")
    axs[1, 1].plot(zeros, zeros, label=f"Min Available")
    axs[1, 1].plot(zeros, zeros, label=f"Max Available")

    # Plot AOA
    axs[1, 2].plot(zeros, zeros, label=f"AoA")
    # axs[1,2].plot(zeros, zeros, label=f"Trajectory Angle")

    # Set Labels, Titles and Grids
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_xlabel("x [m]")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].grid()

    axs[0, 1].set_title("Distance Travelled")
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("x [m]")
    axs[0, 1].grid()

    axs[0, 2].set_title("Elevator Angle")
    axs[0, 2].set_xlabel("t [s]")
    axs[0, 2].set_ylabel("Angle [deg]")
    axs[0, 2].grid()

    axs[1, 0].set_title("Velocity")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("v [m/s]")
    axs[1, 0].grid()

    axs[1, 1].set_title("Thrust")
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("Thrust [N]")
    axs[1, 1].grid()

    axs[1, 2].set_title("Angle of Attack")
    axs[1, 2].set_xlabel("t [s]")
    axs[1, 2].set_ylabel("AoA [deg]")
    axs[1, 2].grid()

    return fig, axs


def update_plot(
    trajectories: list[MissionTrajectory],
    traj_ts: list[FloatArray],
    traj_xs: list[FloatArray],
    traj_vs: list[FloatArray],
    fig: Figure,
    axs: list[Axes],
) -> None:
    i = 0
    fig.canvas.flush_events()
    for trajectory, ts, xs, vs in zip(trajectories, traj_ts, traj_xs, traj_vs):
        i += 1
        title: str = trajectory.name
        t = ts[: len(xs)]
        # Make title display the polynomial equation using latex

        fig.suptitle(title, fontsize=16)

        # x_goal = np.linspace(0, traj[-1][0], len(traj))
        x_goal = jnp.linspace(0, 1000, len(xs))
        y_goal = trajectory.fun(x_goal)

        # Plot Trajectory
        line1, line2 = list(axs[0].get_lines())

        line1.set_xdata(x_goal)
        line1.set_ydata(y_goal)
        line1.set_color("red")

        line2.set_xdata([x[0] for x in xs])
        line2.set_ydata([x[1] for x in xs])
        line2.set_color("blue")
        line2.set_label(f"Trajectory_{i}")
        axs[0].relim()
        axs[0].autoscale()

        # Plot Distance Travelled
        line1 = list(axs[1].get_lines())[0]

        line1.set_xdata(t)
        line1.set_ydata([x[0] for x in xs])
        # line1.set_color('blue')
        line1.set_label(f"Course_{i}")

        axs[1].relim()
        axs[1].autoscale()

        # # Plot Elevator Angle
        # line1 = list(axs[2].get_lines())[0]

        # line1.set_xdata(t[: len(elev_angle)])
        # line1.set_ydata([np.rad2deg(a) for a in elev_angle])
        # axs[2].relim()
        # axs[2].autoscale()

        # Plot Velocity
        line1, line2, line3 = list(axs[3].get_lines())

        line1.set_xdata(t)
        line1.set_ydata([np.linalg.norm(v) for v in vs])

        line2.set_xdata(t)
        line2.set_ydata([v[0] for v in vs])

        line3.set_xdata(t)
        line3.set_ydata([v[1] for v in vs])

        line1.set_label(f"Course_{i}")
        line2.set_label(f"Vx_{i}")
        line3.set_label(f"Vy_{i}")

        axs[3].relim()
        axs[3].autoscale()

        # # Plot Thrust
        # line1, line2, line3 = list(axs[4].get_lines())

        # line1.set_xdata(t[: len(thrust_req)])
        # line1.set_ydata([x for x in thrust_req])
        # line1.set_label(f"Required_{i}")

        # thrust_avail = motor_data["thrust_available"]
        # line2.set_xdata(t)
        # line2.set_ydata(thrust_avail[:, 0])
        # line2.set_label(f"Min Available_{i}")

        # line3.set_xdata(t)
        # line3.set_ydata(thrust_avail[:, 1])
        # line3.set_label(f"Max Available_{i}")
        # axs[4].relim()
        # axs[4].autoscale()

        # # Plot AOA
        # line1 = list(axs[5].get_lines())[0]

        # line1.set_xdata(t)
        # line1.set_ydata([np.rad2deg(x[2]) for x in xs])

        # axs[5].relim()
        # axs[5].autoscale()

    axs[0].legend(loc="best")
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    axs[5].legend()

    fig.canvas.draw()
    time.sleep(0.1)
