import subprocess

from ICARUS.solvers import logOFscript


def get_convergence_data(case_directory: str) -> None:
    """Function to create convergence data for OpenFoam results using the FoamLog script

    Args:
        case_directory (str): Directory where the OpenFoam case is located
    """
    subprocess.call(["/bin/bash", "-c", f"{logOFscript}"], cwd=case_directory)
