import os
import subprocess

from pandas import DataFrame

from ICARUS import GenuVP3_exe

from ..post_process import log_forces
from ..utils import GenuParameters
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from .files_gnvp3 import make_input_files


def gnvp3_execute(case_directory: str) -> int:
    """Execute GNVP3 after setting up the inputs

    Args:
        HOMEDIR (str): _description_
        ANGLEDIR (str): _description_

    Returns:
        int: Error Code

    """
    finput = os.path.join(case_directory, "input")
    foutput = os.path.join(case_directory, "gnvp3.out")
    with open(finput, encoding="utf-8") as fin:
        with open(foutput, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [GenuVP3_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
                cwd=case_directory,
            )

    return res


def make_polars_3(CASEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory

    Returns:
        DataFrame: _description_

    """
    return log_forces(CASEDIR, 3)


def gnvp3_case(
    case_directory: str,
    movements: list[list[GNVP_Movement]],
    genu_bodies: list[GenuSurface],
    params: GenuParameters,
    solver2D: str,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        movements (list[list[Movement]]): List of Movements for each body
        genu_bodies (list[dict[GenuSurface]): List of Bodies in GenuSurface format
        params (GenuParameters): Parameters for the simulation
        solver2D (str): Name of 2D Solver to be used

    """
    make_input_files(
        case_directory,
        movements,
        genu_bodies,
        params,
        solver2D,
    )
    gnvp3_execute(case_directory)
