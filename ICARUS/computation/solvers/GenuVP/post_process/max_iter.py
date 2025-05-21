from __future__ import annotations

from typing import TYPE_CHECKING
import os

from ICARUS.database.db import Database

if TYPE_CHECKING:
    from ICARUS.flight_dynamics.state import State
    from ICARUS.vehicle.airplane import Airplane


def get_max_iterations_3(plane: Airplane, state: State, case: str) -> int:
    """Function to get Max Iterations that simulation ran for given an
    airplane object and a case directory

    Args:
        plane (Airplane): Plane Object
        case (str): Relative Case Directory

    Returns:
        int: Max Iterations

    """
    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="GenuVP3",
        case=case,
    )
    fname: str = os.path.join(CASEDIR, "dfile.yours")
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    maxiter = int(data[35].split()[0])
    return maxiter
