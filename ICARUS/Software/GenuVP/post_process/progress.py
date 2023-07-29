import os
import re
from typing import Optional

from ICARUS.Core.tail import tail


def latest_time(
    CASEDIR: str,
) -> tuple[Optional[int], bool]:
    """Get the latest iteration of GNVP3

    Args:
        CASEDIR (str): Directory where it is run
        name (str): pos.out or neg.out depending on run

    Returns:
        Tuple[Optional[int], Optional[float], bool]: Tuple containing IBLM iteration, the angle where the simulation is, and an error flag.
    """

    filename: str = os.path.join(CASEDIR, 'gnvp.out')
    try:
        with open(filename, 'rb') as f:
            data_b: list[bytes] = tail(f, 300)
        data: list[str] = [line.decode() for line in data_b]
    except FileNotFoundError:
        return None, False

    # ERROR
    error: bool = any(re.search(r"forrtl", x) for x in data)

    # ITERATION
    matches: list[str] = [x.split('   ')[1] for x in data if re.search(r"^ Time step  NTIME =", x)]
    times: list[int] = [int(x) for x in matches]
    try:
        latest_t: int = max(times)
        return latest_t, error
    except ValueError:
        return None, error
