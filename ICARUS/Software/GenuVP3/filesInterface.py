import os
from .filesGNVP import makeInput
from .postProcess.forces import forces2polars, forces2pertrubRes


def GNVPexe(HOMEDIR, ANGLEDIR):
    os.chdir(ANGLEDIR)
    os.system("./gnvp3 < input > gnvp.out")
    os.chdir(HOMEDIR)


def makePolar(CASEDIR, HOMEDIR):
    return forces2polars(CASEDIR, HOMEDIR)


def pertrResults(DYNDIR, HOMEDIR):
    return forces2pertrubRes(DYNDIR, HOMEDIR)


def runGNVPcase(CASEDIR, HOMEDIR, GENUBASE, movements, bodies, params, airfoils, polars, solver2D):
    makeInput(CASEDIR, HOMEDIR, GENUBASE, movements,
              bodies, params, airfoils, polars, solver2D)
    GNVPexe(HOMEDIR, CASEDIR)
