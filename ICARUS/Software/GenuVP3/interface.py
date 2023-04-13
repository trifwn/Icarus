import os
from . import filesGNVP as fgnvp


def GNVPexe(HOMEDIR, ANGLEDIR):
    os.chdir(ANGLEDIR)
    os.system("./gnvp3 < input > gnvp.out")
    # os.system(f"cat LOADS_aer.dat >>  res.dat")
    os.chdir(HOMEDIR)


def makePolar(CASEDIR, HOMEDIR):
    return fgnvp.makePolar(CASEDIR, HOMEDIR)


def logResults(DYNDIR, HOMEDIR):
    return fgnvp.logPertrub(DYNDIR, HOMEDIR)


def runGNVPcase(CASEDIR, HOMEDIR, GENUBASE, movements, bodies, params, airfoils, polars, solver2D):
    fgnvp.makeInput(CASEDIR, HOMEDIR, GENUBASE, movements,
                    bodies, params, airfoils, polars, solver2D)
    GNVPexe(HOMEDIR, CASEDIR)
