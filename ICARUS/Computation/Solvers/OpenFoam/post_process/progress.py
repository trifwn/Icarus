import os
import re

from ICARUS.Core.file_tail import tail


def latest_time(ANGLEDIR: str) -> int | None:
    filename: str = os.path.join(ANGLEDIR, "log")
    try:
        with open(filename, "rb") as f:
            data_b: list[bytes] = tail(f, 300)
            data: list[str] = [line.decode() for line in data_b]
    except FileNotFoundError:
        return None
    times = list(filter(lambda x: re.search(r"^Time =", x), data))
    if isinstance(times, list):
        try:
            latest: str = times[-1]
            latest_t: int = int(latest[7:])
            return latest_t
        except IndexError:
            return None
    elif isinstance(times, str):
        latest = times
        latest_t = int(latest[7:])
        return latest_t
    elif len(times) == 0:
        return None
