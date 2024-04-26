import os

from ICARUS.database import DB3D, DB
from ICARUS.vehicle.plane import Airplane


def get_max_iterations_3(plane: Airplane, case: str) -> int:
    """Function to get Max Iterations that simulation ran for given an
    airplane object and a case directory

    Args:
        plane (Airplane): Plane Object
        case (str): Relative Case Directory

    Returns:
        int: Max Iterations
    """
    CASEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="GenuVP3",
        case = case
    )
    fname: str = os.path.join(CASEDIR, "dfile.yours")
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    maxiter = int(data[35].split()[0])
    return maxiter
