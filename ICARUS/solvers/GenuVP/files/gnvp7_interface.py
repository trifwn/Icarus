import os
import subprocess

from pandas import DataFrame

from ICARUS import GenuVP7_exe

from ..post_process import log_forces
from ..utils import GenuCaseParams
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from .files_gnvp7 import make_input_files


def linux_enable_mpi() -> None:
    """Enable MPI for Linux systems by loading necessary modules"""
    try:
        subprocess.run(
            "module load mkl compiler openmpi/intel",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error loading modules: {e}")
        raise
    except FileNotFoundError:
        print("Module command not found. Ensure you are using a compatible shell.")


def gnvp7_execute(case_directory: str) -> int:
    """Execute GNVP7 after setting up the inputs

    Args:
        case_directory (str): Directory where the case files are located

    Returns:
        int: Error Code

    """
    os.makedirs(os.path.join(case_directory, "vpm_case"), exist_ok=True)

    # If Linux, enable MPI
    try:
        if os.name == "posix":
            linux_enable_mpi()
    except Exception:
        pass

    finput = os.path.join(case_directory, "input")
    foutput = os.path.join(case_directory, "gnvp7.out")
    with open(finput, encoding="utf-8") as fin:
        with open(foutput, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [GenuVP7_exe],
                shell=True,
                stdin=fin,
                stdout=fout,
                stderr=fout,
                cwd=case_directory,
            )
    return res


def make_polars_7(CASEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory

    Returns:
        DataFrame: _description_

    """
    return log_forces(CASEDIR, 7)


def gnvp7_case(
    case_directory: str,
    movements: list[list[GNVP_Movement]],
    genu_bodies: list[GenuSurface],
    params: GenuCaseParams,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        movements (list[list[Movement]]): List of Movements for each body
        genu_bodies (list[GenuSurface]): List of Bodies in Genu format
        params (GenuParameters): Parameters for the simulation
        solver2D (str): Name of 2D Solver to be used
    """
    make_input_files(
        case_directory=case_directory,
        movements=movements,
        genu_bodies=genu_bodies,
        params=params,
    )
    gnvp7_execute(case_directory)
