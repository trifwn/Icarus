import os
import subprocess

from .filesGNVP import makeInput
from .postProcess.forces import forces_to_pertrubation_results
from .postProcess.forces import forces_to_polars


def GNVPexe(HOMEDIR, ANGLEDIR):
    os.chdir(ANGLEDIR)

    fin = open("input")
    fout = open("gnvp.out", "w")
    res = subprocess.check_call(
        [os.path.join(ANGLEDIR, "gnvp3")],
        stdin=fin,
        stdout=fout,
    )
    fin.close()
    fout.close()

    os.chdir(HOMEDIR)
    return res


def makePolar(CASEDIR, HOMEDIR):
    return forces_to_polars(CASEDIR, HOMEDIR)


def pertrResults(DYNDIR, HOMEDIR):
    return forces_to_pertrubation_results(DYNDIR, HOMEDIR)


def runGNVPcase(
    CASEDIR,
    HOMEDIR,
    GENUBASE,
    movements,
    bodies,
    params,
    airfoils,
    foildb,
    solver2D,
):
    makeInput(
        CASEDIR,
        HOMEDIR,
        GENUBASE,
        movements,
        bodies,
        params,
        airfoils,
        foildb,
        solver2D,
    )
    GNVPexe(HOMEDIR, CASEDIR)
