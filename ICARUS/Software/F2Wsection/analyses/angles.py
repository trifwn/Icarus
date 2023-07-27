import os
from multiprocessing import Pool
from threading import Lock
from threading import Thread
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame
from tqdm.auto import tqdm

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database.db import DB
from ICARUS.Software.F2Wsection.analyses.monitor_progress import parallel_monitor
from ICARUS.Software.F2Wsection.analyses.monitor_progress import serial_monitor
from ICARUS.Software.F2Wsection.files_interface import sequential_run
from ICARUS.Software.F2Wsection.post_process.polars import make_polars
from ICARUS.Software.F2Wsection.utils import separate_angles


def f2w_single_reynolds(
    db: DB,
    airfoil: AirfoilD,
    reynolds: float,
    mach: float,
    all_angles: list[float] | ndarray[Any, dtype[floating[Any]]],
    solver_options: dict[str, Any],
) -> None:
    HOMEDIR, _, REYNDIR, _ = db.foilsDB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=all_angles,
    )

    nangles, pangles = separate_angles(all_angles)

    jobs: list[Thread] = []
    for name, angles in zip(["pos", "neg"], [pangles, nangles]):
        job = Thread(
            target=sequential_run,
            kwargs={
                "CASEDIR": REYNDIR,
                "HOMEDIR": HOMEDIR,
                "airfile": airfoil.file_name,
                "name": name,
                "angles": angles,
                "reynolds": reynolds,
                "mach": mach,
                "solver_options": solver_options,
            },
        )
        jobs.append(job)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()
    os.chdir(HOMEDIR)


def run_single_reynolds(
    db: DB,
    airfoil: AirfoilD,
    reynolds: float,
    mach: float,
    all_angles: list[float],
    solver_options: dict[str, Any],
    position: int = 1,
) -> None:
    HOMEDIR, _, REYNDIR, _ = db.foilsDB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=all_angles,
    )

    nangles, pangles = separate_angles(all_angles)
    max_iter: int = solver_options["max_iter"]
    reyn_str: str = np.format_float_scientific(reynolds, sign=False, precision=2).zfill(
        8,
    )

    job = Thread(
        target=f2w_single_reynolds,
        kwargs={
            "db": db,
            "airfoil": airfoil,
            "reynolds": reynolds,
            "mach": mach,
            "all_angles": all_angles,
            "solver_options": solver_options,
        },
    )
    jobs: list[Thread] = [job]

    # Create a lock to synchronize progress bar updates
    progress_bar_lock = Lock()

    # Create a list to store progress bars
    progress_bars = []

    for name, angles in zip(["pos", "neg"], [pangles, nangles]):
        i = 0
        if name == "neg":
            i = 1
        pbar = tqdm(
            total=max_iter,
            desc=f"\t\t{reyn_str}-{name}-0.0 Progress:",
            position=2 * position + i,
            leave=True,
            colour="#003366",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)
        job_monitor = Thread(
            target=serial_monitor,
            kwargs={
                "progress_bars": progress_bars,
                "REYNDIR": REYNDIR,
                "reyn_str": reyn_str,
                "name": name,
                "position": 2 * position + i,
                "lock": progress_bar_lock,
                "max_iter": max_iter,
                "last": min(angles) if name == "neg" else max(angles),
                "refresh_progress": 2,
            },
        )

        jobs.append(job_monitor)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()


def run_multiple_reynolds_parallel(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:
    REYNDIRS: list[str] = []
    for reyn in reynolds:
        _, _, REYNDIR, _ = db.foilsDB.generate_airfoil_directories(
            airfoil=airfoil,
            reynolds=reyn,
            angles=angles,
        )
        REYNDIRS.append(REYNDIR)

    max_iter: float = solver_options["max_iter"]

    def run() -> None:
        with Pool(12) as pool:
            args_list = [
                (
                    db,
                    airfoil,
                    reyn,
                    mach,
                    angles,
                    solver_options,
                )
                for reyn in reynolds
            ]
            _ = pool.starmap(f2w_single_reynolds, args_list)

    job = Thread(target=run, args=())
    job_monitor = Thread(
        target=parallel_monitor,
        args=(REYNDIRS, reynolds, max_iter, max(angles), min(angles)),
    )

    # Start
    job.start()
    job_monitor.start()

    # Join
    job.join()
    job_monitor.join()


def run_multiple_reynolds_sequentially(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:

    for i, reyn in enumerate(reynolds):
        run_single_reynolds(db, airfoil, reyn, mach, angles, solver_options, i)


def process_f2w_run(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float] | float,
) -> dict[str, DataFrame]:

    polars: dict[str, DataFrame] = {}

    if isinstance(reynolds, float):
        reynolds = [reynolds]

    for reyn in reynolds:
        reynolds_str: str = np.format_float_scientific(
            reyn,
            sign=False,
            precision=3,
        )

        CASEDIR: str = os.path.join(
            db.foilsDB.DATADIR,
            f"NACA{airfoil.name}",
            f"Reynolds_{reynolds_str.replace('+', '')}",
        )

        polars[reynolds_str] = make_polars(CASEDIR, db.HOMEDIR)

    return polars
