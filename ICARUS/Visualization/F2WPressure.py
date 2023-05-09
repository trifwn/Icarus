import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_angle_cp(angle: float) -> None:
    file_name = "COEFPRE.OUT"
    if angle < 0:
        anglef: str = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
    else:
        anglef = str(angle)[::-1].zfill(7)[::-1]
    fullname: str = os.path.join(anglef, file_name)
    data = np.loadtxt(fullname).T
    c: ArrayLike = data[0]
    p1: ArrayLike = data[1]
    plt.title("Pressure Coefficient")
    plt.xlabel("x/c")
    plt.ylabel("C_p")
    plt.plot(c, p1)
    plt.show()


def plot_multiple_cps(angles: list[float]) -> None:
    fname = "COEFPRE.OUT"
    for angle in angles:
        print(angle)
        if angle < 0:
            anglef: str = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        else:
            anglef = str(angle)[::-1].zfill(7)[::-1]
        floc: str = os.path.join(anglef, fname)
        data = np.loadtxt(floc).T
        c = data[0]
        p1 = data[1]
        plt.title("Pressure Coefficient")
        plt.xlabel("x/c")
        plt.ylabel("C_p")
        plt.plot(c, p1)
    plt.show()
