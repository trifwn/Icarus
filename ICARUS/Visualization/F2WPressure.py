import matplotlib.pyplot as plt
import os
import numpy as np

from . import colors, markers


def plotCP(angle):
    fname = 'COEFPRE.OUT'
    if angle < 0:
        anglef = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1]
    else:
        anglef = str(angle)[::-1].zfill(7)[::-1]
    fname = os.path.join(anglef,fname)
    data = np.loadtxt(fname).T
    c = data[0]
    p1 = data[1]
    plt.title('Pressure Coefficient')
    plt.xlabel('x/c')
    plt.ylabel('C_p')
    plt.plot(c, p1)
    plt.show()


def plotMultipleCPs(angles):
    fname = 'COEFPRE.OUT'
    for angle in angles:
        print(angle)
        if angle < 0:
            anglef = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1]
        else:
            anglef = str(angle)[::-1].zfill(7)[::-1]
        floc = os.path.join(anglef,fname)
        data = np.loadtxt(floc).T
        c = data[0]
        p1 = data[1]
        plt.title('Pressure Coefficient')
        plt.xlabel('x/c')
        plt.ylabel('C_p')
        plt.plot(c, p1)
    plt.show()
