import os

import numpy as np

from .getMaxiter import getMaxiter
from ICARUS.Database import DB3D


def getWakeData(plane, case):
    fname = os.path.join(DB3D, plane.CASEDIR, case, "YOURS.WAK")
    with open(fname) as file:
        data = file.readlines()
    a = []
    b = []
    c = []
    iteration = 0
    flag = True
    maxiter = getMaxiter(plane, case)
    for i, line in enumerate(data):
        if line.startswith("  WAKE"):
            foo = line.split()
            iteration = int(foo[3])
            continue
        if iteration >= maxiter:
            foo = line.split()
            if (len(foo) == 4) and flag:
                _, x, y, z = (float(i) for i in foo)
                a.append([x, y, z])
            elif len(foo) == 3:
                x, y, z = (float(i) for i in foo)
                flag = False
                b.append([x, y, z])
            elif len(foo) == 4:
                _, x, y, z = (float(i) for i in foo)
                c.append([x, y, z])

    A1 = np.array(a, dtype=float)
    B1 = np.array(b, dtype=float)
    C1 = np.array(c, dtype=float)

    return A1, B1, C1
