import os
import subprocess
from typing import Any

from pandas import DataFrame

from .files_gnvp7 import make_input_files
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Input_Output.GenuVP.post_process.forces import forces_to_pertrubation_results
from ICARUS.Input_Output.GenuVP.post_process.forces import log_forces
from ICARUS.Input_Output.GenuVP.utils.genu_movement import Movement
from ICARUS.Input_Output.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.Input_Output.GenuVP.utils.genu_surface import GenuSurface


def gnvp7_execute(HOMEDIR: str, ANGLEDIR: str) -> int:
    """Execute GNVP7 after setting up the inputs

    Args:
        HOMEDIR (str): Home Directory
        ANGLEDIR (str): Angle Directory

    Returns:
        int: Error Code
    """
    os.chdir(ANGLEDIR)

    cmd = "module load compiler openmpi > /dev/null ; mpirun '-n' '4' 'gnvp7'"
    with open("input", encoding="utf-8") as fin:
        with open("gnvp.out", "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                cmd,
                shell=True,
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


def run_gnvp7_case(
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
        movements (list[list[Movement]]): List of Movements for each body
        bodies_dicts (list[GenuSurface]): List of Bodies in Genu format
        params (GenuParameters): Parameters for the simulation
        airfoils (list[str]): List with the names of all airfoils
        foildb (Database_2D): 2D Foil Database
        solver2D (str): Name of 2D Solver to be used
    """
    make_input_files(
        ANGLEDIR=CASEDIR,
        HOMEDIR=HOMEDIR,
        movements=movements,
        bodies_dicts=bodies_dicts,
        params=params,
        airfoils=airfoils,
        foil_dat=foildb.data,
        solver=solver2D,
    )
    gnvp7_execute(HOMEDIR, CASEDIR)
