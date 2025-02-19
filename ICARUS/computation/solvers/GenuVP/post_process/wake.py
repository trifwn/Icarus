import os

import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.vehicle.plane import Airplane

from .max_iter import get_max_iterations_3


def get_wake_data_3(
    plane: Airplane,
    case: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid

    """
    DB = Database.get_instance()
    CASEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="GenuVP3",
        case=case,
    )
    fname: str = os.path.join(CASEDIR, "YOURS.WAK")
    with open(fname) as file:
        data: list[str] = file.readlines()
    positions: list[list[float]] = []
    vorticity: list[list[float]] = []
    velocity: list[list[float]] = []
    deformation: list[list[float]] = []

    b: list[list[float]] = []
    c: list[list[float]] = []
    iteration = 0
    maxiter: int = get_max_iterations_3(plane, case)
    for i, line in enumerate(data):
        if line.startswith("WAKE"):
            foo: list[str] = line.split()
            iteration = int(foo[3])
            continue
        if iteration >= maxiter:
            foo = line.split()
            if len(foo) == 13:
                i, x, y, z, vx, vy, vz, ux, uy, uz, gx, gy, gz = (float(i) for i in foo)
                positions.append([x, y, z])
                vorticity.append([vx, vy, vz])
                velocity.append([ux, uy, uz])
                deformation.append([gx, gy, gz])
            elif len(foo) == 3:
                x, y, z = (float(i) for i in foo)
                b.append([x, y, z])
            elif len(foo) == 4:
                _, x, y, z = (float(i) for i in foo)
                c.append([x, y, z])

    XP: FloatArray = np.array(positions, dtype=float)
    QP: FloatArray = np.array(vorticity, dtype=float)
    VP: FloatArray = np.array(velocity, dtype=float)
    GP: FloatArray = np.array(deformation, dtype=float)

    B1: FloatArray = np.array(b, dtype=float)
    C1: FloatArray = np.array(c, dtype=float)

    return XP, QP, VP, GP, B1, C1


def nwake_data_7(
    plane: Airplane,
    case: str,
) -> FloatArray:
    """Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid

    """
    DB = Database.get_instance()
    CASEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="GenuVP7",
        case=case,
    )
    try:
        fname: str = os.path.join(CASEDIR, "NWAKE_FINAL")

        with open(fname) as file:
            data: list[str] = file.readlines()

    except FileNotFoundError:
        fname = os.path.join(CASEDIR, "NWAKE00f")
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
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid

    """
    DB = Database.get_instance()
    CASEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="GenuVP7",
        case=case,
    )
    fname: str = os.path.join(CASEDIR, "VORTPF")

    with open(fname) as file:
        data: list[str] = file.readlines()
    positions: list[list[float]] = []
    charges: list[list[float]] = []
    for line in data:
        foo: list[str] = line.split()
        if len(foo) == 0:
            continue
        try:
            (
                x, y, z , 
                i, ivr_nb, 
                vx, vy, vz,
                ale, gam, aavr, rvrp,
                xpn, ypn, zpn,
            ) = (float(num) for num in foo)
            positions.append([x, y, z])
            charges.append([vx, vy, vz])
        except ValueError as e:
            pass

    XP: FloatArray = np.array(positions, dtype=float)
    QP: FloatArray = np.array(charges, dtype=float)
    UP: FloatArray = np.array(positions, dtype=float)
    GP: FloatArray = np.array(positions, dtype=float)
    return XP, QP, UP, GP


def grid_data_7(
    plane: Airplane,
    case: str,
) -> FloatArray:
    """Get the wake data from a given case by reading the YOURS.WAK file.

    Args:
        plane (Airplane): Airplane Object
        case (str): Case Directory

    Returns:
        tuple[FloatArray, FloatArray, FloatArray]: A1: The Particle Wake, B1: The near Wake, C1: The Grid

    """
    DB = Database.get_instance()
    CASEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="GenuVP7",
        case=case,
    )
    try:
        fname: str = os.path.join(CASEDIR, "GWING_FINAL")
        with open(fname) as file:
            data: list[str] = file.readlines()
    except FileNotFoundError:
        fname = os.path.join(CASEDIR, "GWING000")
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
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    XP, QP, UP, GP = wake_data_7(plane, case)
    B = nwake_data_7(plane, case)
    C = grid_data_7(plane, case)

    return XP, QP, UP, GP, B, C
