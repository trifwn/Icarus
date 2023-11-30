from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from typing import Optional

import numpy as np
from tqdm.auto import tqdm
from ICARUS import CPU_TO_USE

from ICARUS.Input_Output.F2Wsection.post_process.progress import latest_time


def serial_monitor(
    progress_bars: list[tqdm],
    REYNDIR: str,
    reyn_str: str,
    name: str,
    position: int,
    lock: Optional[Lock],
    max_iter: int,
    last: float,
    refresh_progress: float = 2,
) -> None:
    sleep(5 + (position + 1) / 10)
    angle_prev: float = 0

    while True:
        sleep(refresh_progress)
        time, angle, error = latest_time(REYNDIR, f"{name}.out")

        if angle is None:
            angle = angle_prev
        else:
            angle_prev = angle
            progress_bars[position].desc = f"\t\t{reyn_str}-{name}-{angle} Progress"

        if error:
            progress_bars[position].write(f"Analysis encountered Error at {reyn_str} {angle}")
            break

        if time is None:
            continue

        if lock:
            with lock:
                progress_bars[position].n = int(time)
                progress_bars[position].refresh(nolock=True)
        else:
            progress_bars[position].n = int(time)
            progress_bars[position].refresh(nolock=True)

        if time >= max_iter and abs(angle) >= abs(last):
            break


def parallel_monitor(
    REYNDIRS: list[str],
    reynolds: list[float],
    max_iter: int,
    max_angle: int,
    min_angle: int,
    refresh_progress: float = 2,
) -> None:
    # Create a lock to synchronize progress bar updates
    progress_bar_lock = Lock()

    # Create a list to store progress bars
    progress_bars = []

    with ThreadPoolExecutor(max_workers= min(2 * len(reynolds), CPU_TO_USE)) as executor:
        for i, reyn in enumerate(reynolds):
            reyn_str: str = np.format_float_scientific(reyn, sign=False, precision=3, min_digits=3).zfill(8)
            for j, name in enumerate(["pos", "neg"]):
                pbar = tqdm(
                    total=max_iter,
                    desc=f"\t\t{reyn_str}-{name}-0.0 Progress:",
                    position=2 * i + j,
                    leave=True,
                    colour="#cc3300",
                    bar_format="{l_bar}{bar:30}{r_bar}",
                )

                progress_bars.append(pbar)

                # Start the progress update in parallel using ThreadPoolExecutor
                executor.submit(
                    serial_monitor,
                    progress_bars=progress_bars,
                    REYNDIR=REYNDIRS[i],
                    reyn_str=reyn_str,
                    name=name,
                    position=2 * i + j,
                    lock=progress_bar_lock,
                    max_iter=max_iter,
                    last=min_angle if name == "neg" else max_angle,
                    refresh_progress=refresh_progress,
                )

    for pbar in progress_bars:
        pbar.close()
