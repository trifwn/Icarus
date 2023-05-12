import os
import subprocess
from io import TextIOWrapper
from typing import Any

from pandas import DataFrame

from .filesGNVP import makeInput
from .postProcess.forces import forces_to_pertrubation_results
from .postProcess.forces import forces_to_polars
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Software.GenuVP3.utils import Movement


def gnvp_execute(HOMEDIR: str, ANGLEDIR: str) -> int:
    """Execute GNVP3 after setting up the inputs

    Args:
        HOMEDIR (str): _description_
        ANGLEDIR (str): _description_

    Returns:
        int: Error Code
    """
    os.chdir(ANGLEDIR)

    fin: TextIOWrapper = open("input", encoding="utf-8")
    fout: TextIOWrapper = open("gnvp.out", "w", encoding="utf-8")
    res: int = subprocess.check_call(
        [os.path.join(ANGLEDIR, "gnvp3")],
        stdin=fin,
        stdout=fout,
    )
    fin.close()
    fout.close()

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
    return forces_to_polars(CASEDIR, HOMEDIR)


def pertubation_results(DYNDIR: str, HOMEDIR: str) -> DataFrame:
    """Returns a dataframe with the results of a disturbance/pertubation analysis

    Args:
        DYNDIR (str): _description_
        HOMEDIR (str): _description_

    Returns:
        DataFrame: _description_
    """
    return forces_to_pertrubation_results(DYNDIR, HOMEDIR)


def run_gnvp_case(
    CASEDIR: str,
    HOMEDIR: str,
    GENUBASE: str,
    movements: list[list[Movement]],
    bodies: list[dict[str, Any]],
    params: dict[str, Any],
    airfoils: list[float],
    foildb: Database_2D,
    solver2D: str,
) -> None:
    """Makes input and runs GNVP3, for a specified Case

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        GENUBASE (str): Base of GenuVP3
        movements (list[list[Movement]]): List of Movements for each body
        bodies (list[dict[str, Any]]): List of Bodies in dict format
        params (dict[str, Any]): Parameters for the simulation
        airfoils (list[float]): List with the names of all airfoils
        foildb (Database_2D): 2D Foil Database
        solver2D (str): Name of 2D Solver to be used
    """
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
    gnvp_execute(HOMEDIR, CASEDIR)
