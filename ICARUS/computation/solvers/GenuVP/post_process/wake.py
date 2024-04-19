import os

import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.database import DB3D
from ICARUS.vehicle.plane import Airplane

from .max_iter import get_max_iterations_3


def get_wake_data_3(
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
    fname: str = os.path.join(DB3D, plane.directory, "GenuVP3", case, "YOURS.WAK")
    with open(fname) as file:
        data: list[str] = file.readlines()
    a: list[list[float]] = []
    b: list[list[float]] = []
    c: list[list[float]] = []
    iteration = 0
    flag: bool = True
    maxiter: int = get_max_iterations_3(plane, case)
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


def nwake_data_7(
    plane: Airplane,
    case: str,
) -> FloatArray:
    """
    Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid
    """
    try:
        fname: str = os.path.join(DB3D, plane.directory, "GenuVP7", case, "NWAKE_FINAL")

        with open(fname) as file:
            data: list[str] = file.readlines()

    except FileNotFoundError:
        fname = os.path.join(DB3D, plane.directory, "GenuVP7", case, "NWAKE00f")
        with open(fname) as file:
            data = file.readlines()

    a: list[list[float]] = []
    for i, line in enumerate(data):
        foo: list[str] = line.split()
        if len(foo) == 0:
            continue
        try:
            x, y, z = (float(num) for num in foo)
            a.append([x, y, z])
        except ValueError:
            pass
            # print(foo)

    B: FloatArray = np.array(a, dtype=float)

    return B


def wake_data_7(
    plane: Airplane,
    case: str,
) -> FloatArray:
    """
    Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid
    """
    fname: str = os.path.join(DB3D, plane.directory, "GenuVP7", case, "VORTPF")

    with open(fname) as file:
        data: list[str] = file.readlines()
    a: list[list[float]] = []
    for line in data:
        foo: list[str] = line.split()
        if len(foo) == 0:
            continue
        try:
            foo = foo[:3]
            x, y, z = (float(num) for num in foo)
            a.append([x, y, z])
        except ValueError:
            pass
            # print(foo)

    A: FloatArray = np.array(a, dtype=float)

    return A


def grid_data_7(
    plane: Airplane,
    case: str,
) -> FloatArray:
    """
    Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid
    """
    try:
        fname: str = os.path.join(DB3D, plane.directory, "GenuVP7", case, "GWING_FINAL")
        with open(fname) as file:
            data: list[str] = file.readlines()
    except FileNotFoundError:
        fname = os.path.join(DB3D, plane.directory, case, "GenuVP7", "GWING000")
        with open(fname) as file:
            data = file.readlines()
    a: list[list[float]] = []
    for i, line in enumerate(data):
        foo: list[str] = line.split()
        if len(foo) == 0:
            continue
        try:
            x, y, z = (float(num) for num in foo)
            a.append([x, y, z])
        except ValueError:
            pass
            # print(foo)

    C: FloatArray = np.array(a, dtype=float)

    return C


def get_wake_data_7(
    plane: Airplane,
    case: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    A = wake_data_7(plane, case)
    B = nwake_data_7(plane, case)
    C = grid_data_7(plane, case)

    return A, B, C
