import os

from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np


def plotCP(angle) -> None:
    fname = "COEFPRE.OUT"
    if angle < 0:
        anglef: str = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
    else:
        anglef = str(angle)[::-1].zfill(7)[::-1]
    fname: str = os.path.join(anglef, fname)
    data = np.loadtxt(fname).T
    c: ArrayLike = data[0]
    p1: ArrayLike = data[1]
    plt.title("Pressure Coefficient")
    plt.xlabel("x/c")
    plt.ylabel("C_p")
    plt.plot(c, p1)
    plt.show()


def plotMultipleCPs(angles) -> None:
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
