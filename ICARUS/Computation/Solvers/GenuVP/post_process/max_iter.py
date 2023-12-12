import os

from ICARUS.Database import DB3D
from ICARUS.Vehicle.plane import Airplane


def get_max_iterations_3(plane: Airplane, case: str) -> int:
    """Function to get Max Iterations that simulation ran for given an
    airplane object and a case directory

    Args:
        plane (Airplane): Plane Object
        case (str): Relative Case Directory

    Returns:
        int: Max Iterations
    """
    fname: str = os.path.join(DB3D, plane.directory, "GenuVP3", case, "dfile.yours")
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    maxiter = int(data[35].split()[0])
    return maxiter
