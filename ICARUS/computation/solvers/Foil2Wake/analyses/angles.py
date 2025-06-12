import os
from multiprocessing import Pool
from threading import Lock
from threading import Thread
from typing import Any

import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers.Foil2Wake.analyses.monitor_progress import (
    parallel_monitor,
)
from ICARUS.computation.solvers.Foil2Wake.analyses.monitor_progress import (
    serial_monitor,
)
from ICARUS.computation.solvers.Foil2Wake.files_interface import sequential_run
from ICARUS.computation.solvers.Foil2Wake.post_process.polars import make_polars
from ICARUS.computation.solvers.Foil2Wake.utils import separate_angles
from ICARUS.core.types import FloatArray
from ICARUS.database import Database


def f2w_single_reynolds(
    airfoil: Airfoil,
    reynolds: float,
    mach: float,
    angles: list[float] | FloatArray,
    solver_options: dict[str, Any],
) -> None:
    DB = Database.get_instance()
    _, _, REYNDIR, _ = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    nangles, pangles = separate_angles(angles)

    jobs: list[Thread] = []
    for name, angles in zip(["pos", "neg"], [pangles, nangles]):
        job = Thread(
            target=sequential_run,
            kwargs={
                "CASEDIR": REYNDIR,
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


def run_single_reynolds(
    airfoil: Airfoil,
    reynolds: float,
    mach: float,
    angles: list[float],
    solver_options: dict[str, Any],
    position: int = 1,
) -> None:
    DB = Database.get_instance()
    HOMEDIR, _, REYNDIR, _ = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )

    nangles, pangles = separate_angles(angles)
    max_iter: int = solver_options["max_iter"]
    reyn_str: str = np.format_float_scientific(
        reynolds,
        sign=False,
        precision=3,
        min_digits=3,
    ).zfill(
        8,
    )

    job = Thread(
        target=f2w_single_reynolds,
        kwargs={
            "airfoil": airfoil,
            "reynolds": reynolds,
            "mach": mach,
            "angles": angles,
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
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:
    REYNDIRS: list[str] = []
    DB = Database.get_instance()
    for reyn in reynolds:
        _, _, REYNDIR, _ = DB.generate_airfoil_directories(
            airfoil=airfoil,
            reynolds=reyn,
            angles=angles,
        )
        REYNDIRS.append(REYNDIR)

    max_iter: float = solver_options["max_iter"]

    def run() -> None:
        with Pool(CPU_TO_USE) as pool:
            args_list = [
                (
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
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, float],
) -> None:
    for i, reyn in enumerate(reynolds):
        print("Running Reynolds number: ", reyn)
        run_single_reynolds(airfoil, reyn, mach, angles, solver_options, i)


def process_f2w_run(
    airfoil: Airfoil,
    reynolds: list[float] | float,
) -> dict[str, DataFrame]:
    polars: dict[str, DataFrame] = {}
    DB = Database.get_instance()
    if isinstance(reynolds, float):
        reynolds_list: list[float] = [reynolds]
    elif isinstance(reynolds, list):
        reynolds_list = reynolds
    elif isinstance(reynolds, np.ndarray):
        reynolds_list = reynolds.tolist()
    else:
        raise TypeError(
            f"reynolds must be either a float or a list of floats. Got {type(reynolds)}",
        )

    for reyn in reynolds_list:
        reynolds_str: str = np.format_float_scientific(
            reyn,
            sign=False,
            precision=3,
            min_digits=3,
        ).replace("+", "")

        CASEDIR: str = os.path.join(
            DB.DB2D,
            f"{airfoil.name.upper()}",
            f"Reynolds_{reynolds_str}",
        )

        polars[reynolds_str] = make_polars(CASEDIR, DB.HOMEDIR)
    resutls_folder = os.path.join(DB.DB2D, airfoil.name.upper())
    DB.load_airfoil_data(resutls_folder)
    return polars
