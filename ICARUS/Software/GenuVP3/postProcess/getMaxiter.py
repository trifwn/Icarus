import os


def getMaxiter(plane, case):
    fname = os.path.join(plane.CASEDIR, case,"dfile.yours")
    with open(fname, "r") as file:
        data = file.readlines()
    maxiter = int(data[35].split()[0])
    return maxiter
