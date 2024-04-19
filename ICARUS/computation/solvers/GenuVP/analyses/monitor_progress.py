from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from threading import Lock
from time import sleep
from typing import NoReturn

from tqdm.auto import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.computation.solvers.GenuVP.post_process.progress import latest_time
from ICARUS.core.types import FloatArray


def serial_monitor(
    progress_bars: list[tqdm[NoReturn]],
    CASEDIR: str,
    position: int,
    lock: Lock | None,
    max_iter: int,
    refresh_progress: float,
    genu_version: int,
    stop_event: Event = Event(),
) -> None:
    sleep(1 + (position + 1) / 10)

    while not stop_event.is_set():
        sleep(refresh_progress)
        time, error = latest_time(CASEDIR, genu_version)

        if error:
            progress_bars[position].write("Analysis encountered Error")
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

        if time >= max_iter:
            break


def parallel_monitor(
    CASEDIRS: list[str],
    variables: list[str] | list[float] | FloatArray,
    max_iter: int,
    genu_version: int,
    refresh_progress: float = 0.2,
    stop_event: Event = Event(),
) -> None:
    # Create a lock to synchronize progress bar updates
    progress_bar_lock = Lock()

    # Create a list to store progress bars
    progress_bars = []

    with ThreadPoolExecutor(max_workers=min(len(variables), CPU_TO_USE)) as executor:
        for i, var in enumerate(variables):
            pbar = tqdm(
                total=max_iter,
                desc=f"\t\t{var} Progress:",
                position=i,
                leave=True,
                colour="#cc3300",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            progress_bars.append(pbar)

            # Start the progress update in parallel using ThreadPoolExecutor
            executor.submit(
                serial_monitor,
                progress_bars=progress_bars,
                CASEDIR=CASEDIRS[i],
                position=i,
                lock=progress_bar_lock,
                max_iter=max_iter,
                refresh_progress=refresh_progress,
                genu_version=genu_version,
                stop_event=stop_event,
            )

    for pbar in progress_bars:
        pbar.close()
