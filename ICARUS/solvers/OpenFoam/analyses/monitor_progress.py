from threading import Lock
from time import sleep

from ICARUS.solvers.OpenFoam.post_process.progress import latest_time


def serial_monitor(
    ANGLEDIR: str,
    position: int,
    lock: Lock | None,
    max_iter: int,
    refresh_progress: float,
) -> None:
    sleep((position + 1) / 100)

    while True:
        sleep(refresh_progress)
        time: int | None = latest_time(ANGLEDIR)

        if time is None:
            continue

        # if lock:
        #     with lock:
        #         progress_bars[position].n = int(time)
        #         progress_bars[position].refresh(nolock=True)
        # else:
        #     progress_bars[position].n = int(time)
        #     progress_bars[position].refresh(nolock=True)

        if time >= max_iter:
            break
