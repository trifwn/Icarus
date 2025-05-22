import os
import subprocess

from pandas import DataFrame

from ICARUS import GenuVP3_exe
from ICARUS.computation.solvers.GenuVP.post_process.forces import log_forces
from ICARUS.computation.solvers.GenuVP.utils.genu_movement import Movement
from ICARUS.computation.solvers.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.computation.solvers.GenuVP.utils.genu_surface import GenuSurface

from .files_gnvp3 import make_input_files


def gnvp3_execute(HOMEDIR: str, ANGLEDIR: str) -> int:
    """Execute GNVP3 after setting up the inputs

    Args:
        HOMEDIR (str): _description_
        ANGLEDIR (str): _description_

    Returns:
        int: Error Code

    """
    os.chdir(ANGLEDIR)

    with open("input", encoding="utf-8") as fin:
        with open("gnvp3.out", "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [GenuVP3_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )

    os.chdir(HOMEDIR)
    return res


def make_polars_3(CASEDIR: str, HOMEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory

    Returns:
        DataFrame: _description_

    """
    return log_forces(CASEDIR, HOMEDIR, 3)


def gnvp3_case(
    CASEDIR: str,
    HOMEDIR: str,
    movements: list[list[Movement]],
    genu_bodies: list[GenuSurface],
    params: GenuParameters,
    airfoils: list[str],
    solver2D: str,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        GENUBASE (str): Base of GenuVP3
        movements (list[list[Movement]]): List of Movements for each body
        genu_bodies (list[dict[GenuSurface]): List of Bodies in GenuSurface format
        params (GenuParameters): Parameters for the simulation
        airfoils (list[str]): List with the names of all airfoils
        solver2D (str): Name of 2D Solver to be used

    """
    make_input_files(
        CASEDIR,
        HOMEDIR,
        movements,
        genu_bodies,
        params,
        solver2D,
    )
    gnvp3_execute(HOMEDIR, CASEDIR)
