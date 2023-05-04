import os

from ICARUS.Database import DB3D


def getMaxiter(plane, case):
    fname = os.path.join(DB3D, plane.CASEDIR, case, "dfile.yours")
    with open(fname) as file:
        data = file.readlines()
    maxiter = int(data[35].split()[0])
    return maxiter
