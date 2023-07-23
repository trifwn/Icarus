from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database.db import DB
from ICARUS.Software import runOFscript


import os
import numpy as np
from numpy import ndarray, dtype, floating
from typing import Any
from subprocess import call

from ICARUS.Software.OpenFoam.filesOpenFoam import MeshType, setup_open_foam
from ICARUS.Database import BASEOPENFOAM

def run_angle(CASEDIR: str, angle: float) -> str:
    """Function to run OpenFoam for a given angle given it is already setup
    Args:
        CASEDIR (str): CASE DIRECTORY
        angle (float): Angle to run
    """
    if angle >= 0:
        folder: str = str(angle)[::-1].zfill(7)[::-1]
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]

    ANGLEDIR: str = os.path.join(CASEDIR, folder)
    os.chdir(ANGLEDIR)
    print(f"{angle} deg: Simulation Starting")
    call(["/bin/bash", "-c", f"{runOFscript}"])
    os.chdir(CASEDIR)

    return f"{angle} deg: Simulation Done"


def angles_serial(
    db: DB,
    airfoil: AirfoilD,
    angles: list[float] | ndarray[Any, dtype[floating]],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:

    #! TODO : ADD TO DB 
    os.chdir(db.foilsDB.DATADIR)
    AFDIR = os.path.join(
        db.foilsDB.DATADIR,
        f"NACA{airfoil.name}"
    )
    os.makedirs(AFDIR, exist_ok=True)
    exists = False
    for i in os.listdir():
        if i.startswith("naca"):
            exists = True
    if not exists:
            airfoil.save(AFDIR)
    os.chdir(AFDIR)

    reynolds_str: str = np.format_float_scientific(
        reynolds,
        sign=False,
        precision=3,
    )
    
    REYNDIR: str = os.path.join(
        AFDIR,
        f"Reynolds_{reynolds_str.replace('+', '')}",
    )

    HOMEDIR: str = db.HOMEDIR
    setup_open_foam(
        HOMEDIR,
        AFDIR,
        REYNDIR,
        BASEOPENFOAM,
        airfoil.file_name,
        reynolds,
        mach,
        angles,
        solver_options
    )
    for angle in angles:
        msg = run_angle(REYNDIR, angle)
        print(msg)
    os.chdir(HOMEDIR)

def angles_parallel(
    db: DB,
    airfoil: AirfoilD,
    angles: list[float] | ndarray[Any, dtype[floating]],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    #! TODO : ADD TO DB 
    os.chdir(db.foilsDB.DATADIR)
    AFDIR = os.path.join(
        db.foilsDB.DATADIR,
        f"NACA{airfoil.name}"
    )
    os.makedirs(AFDIR, exist_ok=True)
    exists = False
    for i in os.listdir():
        if i.startswith("naca"):
            exists = True
    if not exists:
            airfoil.save(AFDIR)
    os.chdir(AFDIR)

    reynolds_str: str = np.format_float_scientific(
        reynolds,
        sign=False,
        precision=3,
    )
    
    REYNDIR: str = os.path.join(
        AFDIR,
        f"Reynolds_{reynolds_str.replace('+', '')}",
    )

    HOMEDIR: str = db.HOMEDIR
    setup_open_foam(
        HOMEDIR,
        AFDIR,
        REYNDIR,
        BASEOPENFOAM,
        airfoil.file_name,
        reynolds,
        mach,
        angles,
        solver_options
    )
    from multiprocessing import Pool
    with Pool() as pool:
        args_list = [(REYNDIR, angle) for angle in angles]
        res = pool.starmap(run_angle, args_list)
    
    for msg in res:
        print(msg)