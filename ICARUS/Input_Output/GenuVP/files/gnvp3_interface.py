import os
import subprocess
from typing import Any

from pandas import DataFrame

from .files_gnvp3 import make_input_files
from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Input_Output.GenuVP.post_process.forces import forces_to_pertrubation_results
from ICARUS.Input_Output.GenuVP.post_process.forces import log_forces
from ICARUS.Input_Output.GenuVP.utils.genu_movement import Movement
from ICARUS.Input_Output.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.Input_Output.GenuVP.utils.genu_surface import GenuSurface


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
        with open("gnvp.out", "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [os.path.join(ANGLEDIR, "gnvp3")],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )

    os.chdir(HOMEDIR)
    return res


def make_polars(CASEDIR: str, HOMEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory

    Returns:
        DataFrame: _description_
    """
    return log_forces(CASEDIR, HOMEDIR)


def pertubation_results(DYNDIR: str, HOMEDIR: str) -> DataFrame:
    """Returns a dataframe with the results of a disturbance/pertubation analysis

    Args:
        DYNDIR (str): _description_
        HOMEDIR (str): _description_

    Returns:
        DataFrame: _description_
    """
    return forces_to_pertrubation_results(DYNDIR, HOMEDIR)


def run_gnvp3_case(
    CASEDIR: str,
    HOMEDIR: str,
    movements: list[list[Movement]],
    bodies_dicts: list[GenuSurface],
    params: GenuParameters,
    airfoils: list[str],
    foildb: Database_2D,
    solver2D: str,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        GENUBASE (str): Base of GenuVP3
        movements (list[list[Movement]]): List of Movements for each body
        bodies (list[dict[GenuSurface]): List of Bodies in GenuSurface format
        params (GenuParameters): Parameters for the simulation
        airfoils (list[str]): List with the names of all airfoils
        foildb (Database_2D): 2D Foil Database
        solver2D (str): Name of 2D Solver to be used
    """
    make_input_files(
        CASEDIR,
        HOMEDIR,
        GENUBASE,
        movements,
        bodies_dicts,
        params,
        airfoils,
        foildb.data,
        solver2D,
    )
    gnvp3_execute(HOMEDIR, CASEDIR)
