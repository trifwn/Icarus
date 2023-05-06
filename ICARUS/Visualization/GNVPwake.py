import matplotlib.pyplot as plt
import numpy as np

from ICARUS.Software.GenuVP3.postProcess.getWakeData import getWakeData


def GNVPwake(plane, case, figsize=(16, 7)):
    A1, B1, C1 = getWakeData(plane, case)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.set_title(f"{plane.name} WAKE")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(30, 150)
    ax.axis("scaled")
    ax.set_xlim(-plane.span / 2, plane.span / 2)
    ax.set_ylim(-plane.span / 2, plane.span / 2)
    ax.set_zlim(-1, 1)

    ax.scatter(A1[:, 0], A1[:, 1], A1[:, 2], color="r", s=5.0)  # WAKE
    # ax.scatter(B1[:, 0], B1[:, 1], B1[:, 2], color='k', s=5.)  # NEARWAKE
    ax.scatter(C1[:, 0], C1[:, 1], C1[:, 2], color="g", s=5.0)  # GRID

    plane.visAirplane(fig, ax, movement=-np.array(plane.CG))
    plt.show()


def GNVPwakeVideo(plane, case, figsize=(16, 7)):
    pass
