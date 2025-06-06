import os
import subprocess

from pandas import DataFrame

from ICARUS import GenuVP7_exe

from ..post_process import log_forces
from ..utils import GenuParameters
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from .files_gnvp7 import make_input_files


def gnvp7_execute(HOMEDIR: str, ANGLEDIR: str) -> int:
    """Execute GNVP7 after setting up the inputs

    Args:
        HOMEDIR (str): Home Directory
        ANGLEDIR (str): Angle Directory

    Returns:
        int: Error Code

    """
    os.chdir(ANGLEDIR)
    os.makedirs(os.path.join(ANGLEDIR, "vpm_case"), exist_ok=True)

    # cmd = f"module load mkl compiler openmpi/intel > /dev/null ; mpirun '-n' '4' '{GenuVP7_exe}'"
    cmd = f"module load mkl compiler openmpi/intel > /dev/null ; '{GenuVP7_exe}'"
    # cmd = "gnvp7"
    with open("input", encoding="utf-8") as fin:
        with open("gnvp7.out", "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                cmd,
                shell=True,
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )

    os.chdir(HOMEDIR)
    return res


def make_polars_7(CASEDIR: str, HOMEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory

    Returns:
        DataFrame: _description_

    """
    return log_forces(CASEDIR, HOMEDIR, 7)


def gnvp7_case(
    CASEDIR: str,
    HOMEDIR: str,
    movements: list[list[GNVP_Movement]],
    genu_bodies: list[GenuSurface],
    params: GenuParameters,
    airfoils: list[str],
    solver2D: str,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        movements (list[list[Movement]]): List of Movements for each body
        genu_bodies (list[GenuSurface]): List of Bodies in Genu format
        params (GenuParameters): Parameters for the simulation
        airfoils (list[str]): List with the names of all airfoils
        solver2D (str): Name of 2D Solver to be used

    """
    make_input_files(
        ANGLEDIR=CASEDIR,
        HOMEDIR=HOMEDIR,
        movements=movements,
        genu_bodies=genu_bodies,
        params=params,
        airfoils=airfoils,
        solver=solver2D,
    )
    gnvp7_execute(HOMEDIR, CASEDIR)
