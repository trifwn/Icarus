import subprocess

from ICARUS.computation.solvers import logOFscript


def get_convergence_data(case_directory: str) -> None:
    """Function to create convergence data for OpenFoam results using the FoamLog script

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory

    """
    subprocess.call(["/bin/bash", "-c", f"{logOFscript}"], cwd=case_directory)
