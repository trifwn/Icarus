from __future__ import annotations

from time import sleep

from .progress import latest_time


def serial_monitor(
    CASEDIR: str,
    position: int,
    max_iter: int,
    refresh_progress: float,
    gnvp_version: int,
) -> None:
    sleep(1 + (position + 1) / 10)

    # while not stop_event.is_set():
    sleep(refresh_progress)
    time, error = latest_time(CASEDIR, gnvp_version)

    # if error:
    #     progress_bars[position].write("Analysis encountered Error")
    #     break

    # if time is None:
    #     continue

    # if lock:
    #     with lock:
    #         progress_bars[position].n = int(time)
    #         progress_bars[position].refresh(nolock=True)
    # else:
    #     progress_bars[position].n = int(time)
    #     progress_bars[position].refresh(nolock=True)

    # if time >= max_iter:
    #     break
