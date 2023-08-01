import os

import numpy as np

from .getMaxiter import get_max_iterations
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB3D
from ICARUS.Vehicle.plane import Airplane


def get_wake_data(
    plane: Airplane,
    case: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid
    """
    fname: str = os.path.join(DB3D, plane.CASEDIR, case, "YOURS.WAK")
    with open(fname) as file:
        data: list[str] = file.readlines()
    a: list[list[float]] = []
    b: list[list[float]] = []
    c: list[list[float]] = []
    iteration = 0
    flag: bool = True
    maxiter: int = get_max_iterations(plane, case)
    for i, line in enumerate(data):
        if line.startswith("  WAKE"):
            foo: list[str] = line.split()
            iteration = int(foo[3])
            continue
        if iteration >= maxiter:
            foo = line.split()
            if (len(foo) == 4) and flag:
                _, x, y, z = (float(i) for i in foo)
                a.append([x, y, z])
            elif len(foo) == 3:
                x, y, z = (float(i) for i in foo)
                flag = False
                b.append([x, y, z])
            elif len(foo) == 4:
                _, x, y, z = (float(i) for i in foo)
                c.append([x, y, z])

    A1: FloatArray = np.array(a, dtype=float)
    B1: FloatArray = np.array(b, dtype=float)
    C1: FloatArray = np.array(c, dtype=float)

    return A1, B1, C1
