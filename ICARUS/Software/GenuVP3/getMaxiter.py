import os


def getMaxiter(plane, case):
    os.chdir(plane.CASEDIR)
    os.chdir(case)
    fname = "dfile.yours"
    with open(fname, "r") as file:
        data = file.readlines()
    maxiter = int(data[35].split()[0])

    os.chdir(plane.HOMEDIR)
    return maxiter
