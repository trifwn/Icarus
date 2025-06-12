from multiprocessing import Pool
from subprocess import call
from threading import Thread
from typing import Any

from tqdm.auto import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers import runOFscript
from ICARUS.computation.solvers.OpenFoam.analyses.monitor_progress import (
    parallel_monitor,
)
from ICARUS.computation.solvers.OpenFoam.analyses.monitor_progress import serial_monitor
from ICARUS.computation.solvers.OpenFoam.files.setup_case import setup_open_foam
from ICARUS.core.types import FloatArray
from ICARUS.database import Database


def run_angle(
    directory: str,
) -> None:
    """Function to run OpenFoam for a given angle given it is already setup

    Args:
        ANGLEDIR (float): ANGLE DIRECTORY

    """
    call(["/bin/bash", "-c", f"{runOFscript}"], cwd=directory)


def run_angles(
    case_directories: list[str],
) -> None:
    """Function to run multiple Openfoam Simulations (many AoAs) after they
    are already setup

    Args:
        case_directories (list[str]): List of directories where the cases are setup

    """
    with Pool(processes=CPU_TO_USE - 2) as pool:
        pool.starmap(run_angle, case_directories)


def angles_serial(
    airfoil: Airfoil,
    angles: list[float] | FloatArray,
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    """Runs OpenFoam for multiple angles in serial (same subproccess)

    Args:
        airfoil (Airfoil): Airfoil Object
        angles (list[float] | FloatArray): List of angles to run
        reynolds (float): Reynolds Number
        mach (float): Mach Number
        solver_options (dict[str, Any]): Solver Options in a dictionary

    """
    DB = Database.get_instance()
    HOMEDIR, AFDIR, REYNDIR, ANGLEDIRS = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    setup_open_foam(
        AFDIR,
        REYNDIR,
        airfoil.file_name,
        reynolds,
        mach,
        angles,
        solver_options,
    )
    max_iter: int = solver_options["max_iterations"]

    progress_bars = []
    for pos, angle_dir in enumerate(ANGLEDIRS):
        job = Thread(target=run_angle, args=(angle_dir,))

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


def angles_parallel(
    airfoil: Airfoil,
    angles: list[float] | FloatArray,
    reynolds: float,
    mach: float,
    solver_options: dict[str, Any],
) -> None:
    """Runs OpenFoam for multiple angles in parallel (different subproccesses)

    Args:
        airfoil (Airfoil): Airfoil Object
        angles (list[float] | FloatArray): List of angles
        reynolds (float): Reynolds Number
        mach (float): Mach Number
        solver_options (dict[str, Any]): Dictionary of solver options

    """
    DB = Database.get_instance()
    HOMEDIR, AFDIR, REYNDIR, ANGLEDIRS = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )
    setup_open_foam(
        AFDIR,
        REYNDIR,
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
