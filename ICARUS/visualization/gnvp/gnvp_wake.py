from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.database import angle_to_case
from ICARUS.database import case_to_angle

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


def plot_gnvp3_wake(
    plane: Airplane,
    state: State,
    case: str,
    scale: bool = True,
    figsize: tuple[int, int] = (16, 7),
) -> None:
    plot_gnvp_wake(gnvp_version=3, state=state, plane=plane, case=case, scale=scale, figsize=figsize)


def plot_gnvp7_wake(
    plane: Airplane,
    state: State,
    case: str,
    scale: bool = True,
    figsize: tuple[int, int] = (16, 7),
) -> None:
    plot_gnvp_wake(gnvp_version=7, state=state, plane=plane, case=case, scale=scale, figsize=figsize)


def plot_gnvp_wake(
    gnvp_version: int,
    plane: Airplane,
    state: State,
    case: str | float,
    scale: bool = True,
    figsize: tuple[int, int] = (16, 7),
) -> None:
    """Visualize the wake of a given plane

    Args:
        plane (Airplane): Plane Object
        case (str): Case Name
        scale (str): Whether to plot on a true scale or not
        figsize (tuple[int,int]): Figure Size. Defaults to (16, 7).

    """
    if isinstance(case, float):
        case_str = angle_to_case(case)
    elif isinstance(case, str):
        case_str = case
    else:
        raise ValueError(f"Case must be a string or a float, got {type(case)}")

    if gnvp_version == 3:
        from ICARUS.computation.solvers.GenuVP.post_process import get_wake_data_3

        get_wake_data = get_wake_data_3
    elif gnvp_version == 7:
        from ICARUS.computation.solvers.GenuVP.post_process import get_wake_data_7

        get_wake_data = get_wake_data_7
    else:
        raise ValueError(f"GNVP Version error! Got Version {gnvp_version} ")

    XP, QP, VP, GP, B1, C1 = get_wake_data(plane, state, case_str)

    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore

    ax.set_title(f"{plane.name} wake with GNVP{gnvp_version} for case {case_to_angle(case_str)}")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 150)
    ax.axis("scaled")
    ax.set_xlim(-plane.span / 2, plane.span / 2)
    ax.set_ylim(-plane.span / 2, plane.span / 2)
    ax.set_zlim(-1, 1)

    p = ax.scatter(
        xs=XP[:, 0],
        ys=XP[:, 1],
        zs=XP[:, 2],
        c=np.linalg.norm(QP, axis=1),
        s=5,
    )  # WAKE   # type: ignore
    fig.colorbar(p, ax=ax)

    ax.scatter(
        xs=B1[:, 0],
        ys=B1[:, 1],
        zs=B1[:, 2],
        color="k",
        s=5,
    )  # NEARWAKE   # type: ignore
    ax.scatter(
        xs=C1[:, 0],
        ys=C1[:, 1],
        zs=C1[:, 2],
        color="g",
        s=5,
    )  # GRID       # type: ignore

    plane.plot(fig, ax, movement=-np.array(plane.CG), show_masses=False)
    if scale:
        ax.set_aspect("equal", "box")
    plt.show()


# def gnvp_wake_video(plane: Airplane, case: str, figsize=(16, 7)) -> None:
#     """
#     TODO: Make a video of the wake

#     Args:
#         plane (Airplane): Plane Object
#         case (str): Case Name
#         figsize (tuple, optional): Figure Size. Defaults to (16, 7).
#     """
#     pass

if __name__ == "__main__":
    ## In the folder we are working on
    import os

    folder_name: str = os.path.basename(os.getcwd())
    from ICARUS.database import Database

    DB = Database.get_instance()

    print(DB.get_vehicle_names())
    try:
        plane: Airplane = DB.get_vehicle(folder_name)
        states = DB.get_vehicle_states(plane)
        state: State = list(states.items())[0][1]
    except KeyError:
        print(f"Plane {folder_name} not found in the database")
        exit()

    # Read from sys.argv
    # arg -g is the genu version
    # arg -a is the angle of attack

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gnvp", type=int, default=3, help="Genu Version")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        default=2.0,
        help="Angle of Attack",
    )
    args = parser.parse_args()
    gnvp_version: int = args.gnvp
    case: float = args.angle

    case_str: str = angle_to_case(case)
    if gnvp_version == 3:
        plot_gnvp3_wake(plane, state, case_str)
    elif gnvp_version == 7:
        plot_gnvp7_wake(plane, state, case_str)
    else:
        print(f"GNVP Version error! Got Version {gnvp_version} ")
        sys.exit()
