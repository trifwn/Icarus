import os
from subprocess import call

from ICARUS.Computation.Solvers import logOFscript


def get_convergence_data(HOMEDIR: str, CASEDIR: str) -> None:
    """Function to create convergence data for OpenFoam results using the FoamLog script

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    os.chdir(CASEDIR)
    call(["/bin/bash", "-c", f"{logOFscript}"])
    os.chdir("logs")

    os.chdir(HOMEDIR)
