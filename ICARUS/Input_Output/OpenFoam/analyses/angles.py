import os
from multiprocessing import Pool
from subprocess import call
from threading import Thread
from typing import Any

from numpy import dtype
from numpy import floating
from numpy import ndarray
from tqdm.auto import tqdm

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Database import BASEOPENFOAM
from ICARUS.Database.db import DB
from ICARUS.Input_Output import runOFscript
from ICARUS.Input_Output.OpenFoam.analyses.monitor_progress import parallel_monitor
from ICARUS.Input_Output.OpenFoam.analyses.monitor_progress import serial_monitor
from ICARUS.Input_Output.OpenFoam.filesOpenFoam import setup_open_foam


def run_angle(
    REYNDIR: str,
    ANGLEDIR: str,
) -> None:
    """Function to run OpenFoam for a given angle given it is already setup

    Args:
        REYNDIR (str): REYNOLDS CASE DIRECTORY
        ANGLEDIR (float): ANGLE DIRECTORY
    """
    pass
    os.chdir(ANGLEDIR)
    call(["/bin/bash", "-c", f"{runOFscript}"])
    os.chdir(REYNDIR)


def run_angles(
    REYNDIR: str,
    ANGLEDIRS: list[str],
) -> None:
    """
    Function to run multiple Openfoam Simulations (many AoAs) after they
    are already setup

    Args:
        REYNDIR (str): Reynolds Parent Directory
        ANGLEDIRS (list[str]): Angle Directory
    """
    with Pool(processes=12) as pool:
        args_list: list[tuple[str, str]] = [(REYNDIR, angle_dir) for angle_dir in ANGLEDIRS]
        pool.starmap(run_angle, args_list)


def angles_serial(
    db: DB,
    airfoil: Airfoil,
    angles: list[float] | ndarray[Any, dtype[floating[Any]]],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    """
    Runs OpenFoam for multiple angles in serial (same subproccess)

    Args:
        db (DB): Database
        airfoil (Airfoil): Airfoil Object
        angles (list[float] | ndarray[Any, dtype[floating[Any]]]): List of angles to run
        reynolds (float): Reynolds Number
        mach (float): Mach Number
        solver_options (dict[str, Any]): Solver Options in a dictionary
    """
    HOMEDIR, AFDIR, REYNDIR, ANGLEDIRS = db.foilsDB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    setup_open_foam(
        HOMEDIR,
        AFDIR,
        REYNDIR,
        BASEOPENFOAM,
        airfoil.file_name,
        reynolds,
        mach,
        angles,
        solver_options,
    )
    max_iter: int = solver_options["max_iterations"]

    progress_bars = []
    for pos, angle_dir in enumerate(ANGLEDIRS):
        job = Thread(target=run_angle, args=(REYNDIR, angle_dir))

        pbar = tqdm(
            total=max_iter,
            desc=f"\t\t{angles[pos]} Progress:",
            position=pos,
            leave=True,
            colour="#003366",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)

        job_monitor = Thread(
            target=serial_monitor,
            kwargs={
                "progress_bars": progress_bars,
                "ANGLEDIR": angle_dir,
                "position": pos,
                "lock": None,
                "max_iter": max_iter,
                "refresh_progress": 2,
            },
        )
        # Start Jobs
        job.start()
        job_monitor.start()

        # Join
        job.join()
        job_monitor.join()
    os.chdir(HOMEDIR)


def angles_parallel(
    db: DB,
    airfoil: Airfoil,
    angles: list[float] | ndarray[Any, dtype[floating[Any]]],
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    """
    Runs OpenFoam for multiple angles in parallel (different subproccesses)

    Args:
        db (DB): Database
        airfoil (Airfoil): Airfoil Object
        angles (list[float] | ndarray[Any, dtype[floating[Any]]]): List of angles
        reynolds (float): Reynolds Number
        mach (float): Mach Number
        solver_options (dict[str, Any]): Dictionary of solver options
    """
    HOMEDIR, AFDIR, REYNDIR, ANGLEDIRS = db.foilsDB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )
    setup_open_foam(
        HOMEDIR,
        AFDIR,
        REYNDIR,
        BASEOPENFOAM,
        airfoil.file_name,
        reynolds,
        mach,
        angles,
        solver_options,
    )
    max_iter: int = solver_options["max_iterations"]

    job = Thread(target=run_angles, args=(REYNDIR, ANGLEDIRS))
    job_monitor = Thread(target=parallel_monitor, args=(ANGLEDIRS, angles, max_iter))

    # Start Jobs
    job.start()
    job_monitor.start()

    # Join
    job.join()
    job_monitor.join()
