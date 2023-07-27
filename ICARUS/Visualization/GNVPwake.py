import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ICARUS.Software.GenuVP3.postProcess.getWakeData import get_wake_data
from ICARUS.Vehicle.plane import Airplane


def gnvp_wake(plane: Airplane, case: str, figsize: tuple[int, int] = (16, 7)) -> None:
    """
    Visualize the wake of a given plane

    Args:
        plane (Airplane): Plane Object
        case (str): Case Name
        figsize (tuple[int,int]): Figure Size. Defaults to (16, 7).
    """
    A1, B1, C1 = get_wake_data(plane, case)

    fig: Figure = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(projection="3d")
    ax.set_title(f"{plane.name} wake")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 150)
    ax.axis("scaled")
    ax.set_xlim(-plane.span / 2, plane.span / 2)
    ax.set_ylim(-plane.span / 2, plane.span / 2)
    ax.set_zlim(-1, 1)

    ax.scatter(A1[:, 0], A1[:, 1], A1[:, 2], color="r", s=5.0)  # WAKE
    ax.scatter(B1[:, 0], B1[:, 1], B1[:, 2], color="k", s=5.0)  # NEARWAKE
    ax.scatter(C1[:, 0], C1[:, 1], C1[:, 2], color="g", s=5.0)  # GRID

    plane.visualize(fig, ax, movement=-np.array(plane.CG))
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
